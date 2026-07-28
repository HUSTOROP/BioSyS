"""Neural network modules for tactile Young's modulus estimation.

The model combines:

1. a ResNet-18 image encoder;
2. force-guided cross-attention;
3. a deformable spatial convolution and temporal convolution; and
4. a fully connected regression head.

The public class used by the training entry point is :class:`PhysiNet`.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.ops import DeformConv2d


class TactileVisualAttention(nn.Module):
    """Fuse a spatial visual feature map with force/width measurements."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )

        hidden_dim = max(embed_dim // 2, 1)
        self.embed_dim = embed_dim
        self.physical_projection = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, visual_features: Tensor, physical_values: Tensor) -> Tensor:
        """Return the force-guided visual feature map.

        Args:
            visual_features: Tensor with shape ``(B, C, H, W)``.
            physical_values: Tensor with shape ``(B, 2)`` containing the
                normalized force and width for each image.
        """
        if visual_features.ndim != 4:
            raise ValueError(
                "visual_features must have shape (B, C, H, W), "
                f"got {tuple(visual_features.shape)}."
            )
        if physical_values.ndim != 2 or physical_values.shape[1] != 2:
            raise ValueError(
                "physical_values must have shape (B, 2), "
                f"got {tuple(physical_values.shape)}."
            )

        batch_size, channels, height, width = visual_features.shape
        if channels != self.embed_dim:
            raise ValueError(
                f"Expected {self.embed_dim} visual channels, got {channels}."
            )
        if physical_values.shape[0] != batch_size:
            raise ValueError("Visual and physical batch sizes do not match.")

        visual_tokens = visual_features.flatten(2).transpose(1, 2)
        physical_token = self.physical_projection(physical_values).unsqueeze(1)
        attention_output, _ = self.cross_attention(
            visual_tokens,
            physical_token,
            physical_token,
            need_weights=False,
        )
        fused_tokens = self.norm1(visual_tokens + attention_output)
        fused_tokens = self.norm2(fused_tokens + self.feed_forward(fused_tokens))
        return (
            fused_tokens.transpose(1, 2)
            .contiguous()
            .reshape(batch_size, channels, height, width)
        )


def _adapt_first_convolution(
    convolution: nn.Conv2d,
    in_channels: int,
    copy_pretrained_weights: bool,
) -> nn.Conv2d:
    """Create a first convolution compatible with non-RGB inputs."""
    new_convolution = nn.Conv2d(
        in_channels=in_channels,
        out_channels=convolution.out_channels,
        kernel_size=convolution.kernel_size,
        stride=convolution.stride,
        padding=convolution.padding,
        bias=convolution.bias is not None,
    )

    if not copy_pretrained_weights:
        return new_convolution

    with torch.no_grad():
        old_weights = convolution.weight
        if in_channels == 1:
            new_convolution.weight.copy_(old_weights.mean(dim=1, keepdim=True))
        else:
            repeats = math.ceil(in_channels / old_weights.shape[1])
            expanded = old_weights.repeat(1, repeats, 1, 1)[:, :in_channels]
            expanded *= old_weights.shape[1] / in_channels
            new_convolution.weight.copy_(expanded)
    return new_convolution


class Encoder2DResNetCrossAttention(nn.Module):
    """ResNet-18 spatial encoder with physical cross-attention."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 256,
        attention_heads: int = 4,
        attention_dropout: float = 0.1,
        pretrained_backbone: bool = True,
        allow_pretrained_fallback: bool = False,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")

        weights = ResNet18_Weights.DEFAULT if pretrained_backbone else None
        try:
            backbone = resnet18(weights=weights)
        except Exception as exc:
            if not (pretrained_backbone and allow_pretrained_fallback):
                raise RuntimeError(
                    "Could not load the pretrained ResNet-18 weights. "
                    "Connect to the internet once, cache the weights, or set "
                    "model.pretrained_backbone=false."
                ) from exc
            warnings.warn(
                "Pretrained ResNet-18 weights were unavailable; using random "
                "initialization because allow_pretrained_fallback=true.",
                RuntimeWarning,
                stacklevel=2,
            )
            backbone = resnet18(weights=None)

        if in_channels != 3:
            backbone.conv1 = _adapt_first_convolution(
                backbone.conv1,
                in_channels=in_channels,
                copy_pretrained_weights=pretrained_backbone,
            )

        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.projection = nn.Sequential(
            nn.Conv2d(512, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.fusion = TactileVisualAttention(
            embed_dim=embed_dim,
            num_heads=attention_heads,
            dropout=attention_dropout,
        )

    def forward(self, images: Tensor, physical_values: Tensor) -> Tensor:
        features = self.backbone(images)
        features = self.projection(features)
        return self.fusion(features, physical_values)


class RheologyDynamicModuleDeform(nn.Module):
    """Deformable spatial and dynamic temporal feature aggregation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("Channel counts must be positive.")
        if temporal_kernel_size <= 0:
            raise ValueError("temporal_kernel_size must be positive.")

        self.in_channels = in_channels
        self.temporal_kernel_size = temporal_kernel_size
        kernel_size = 3
        offset_channels = 2 * kernel_size * kernel_size

        self.offset_convolution = nn.Conv2d(
            in_channels,
            offset_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
            bias=True,
        )
        nn.init.zeros_(self.offset_convolution.weight)
        nn.init.zeros_(self.offset_convolution.bias)

        self.spatial_convolution = DeformConv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
            bias=False,
        )
        self.spatial_activation = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.temporal_convolution = nn.Sequential(
            nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=(temporal_kernel_size, 1, 1),
                bias=False,
            ),
            nn.BatchNorm3d(in_channels),
        )

        gate_hidden_dim = max(in_channels // 4, 1)
        self.dynamic_gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(in_channels, gate_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden_dim, in_channels),
            nn.Sigmoid(),
        )
        self.output_projection = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, features: Tensor) -> Tensor:
        """Aggregate features with shape ``(B, C, T, H, W)``."""
        if features.ndim != 5:
            raise ValueError(
                "features must have shape (B, C, T, H, W), "
                f"got {tuple(features.shape)}."
            )

        batch_size, channels, time_steps, height, width = features.shape
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {channels}.")
        if time_steps != self.temporal_kernel_size:
            raise ValueError(
                f"Expected {self.temporal_kernel_size} frames, got {time_steps}."
            )

        frame_features = (
            features.permute(0, 2, 1, 3, 4)
            .contiguous()
            .reshape(batch_size * time_steps, channels, height, width)
        )
        offsets = self.offset_convolution(frame_features)
        frame_features = self.spatial_convolution(frame_features, offsets)
        frame_features = self.spatial_activation(frame_features)

        spatial_features = (
            frame_features.reshape(
                batch_size,
                time_steps,
                channels,
                height,
                width,
            )
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )
        temporal_features = self.temporal_convolution(spatial_features)
        gate = self.dynamic_gate(features).reshape(
            batch_size,
            channels,
            1,
            1,
            1,
        )
        output = self.output_projection(temporal_features * gate)
        return self.global_pool(output).flatten(1)


class DecoderFC(nn.Module):
    """Fully connected regression decoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (256, 128),
        output_dim: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive.")

        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            if hidden_dim <= 0:
                raise ValueError("All hidden dimensions must be positive.")
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features).squeeze(-1)


class PhysiNet(nn.Module):
    """Estimate Young's modulus from a tactile image sequence and force.

    Args:
        in_channels: Number of image channels per frame.
        time_steps: Number of tactile frames per sample.
        image_embed_dim: Projected ResNet feature dimension.
        pretrained_backbone: Load ImageNet ResNet-18 weights when ``True``.
        use_hertz_residual: Concatenate a scalar Hertz estimate when ``True``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        time_steps: int = 3,
        image_embed_dim: int = 256,
        attention_heads: int = 4,
        attention_dropout: float = 0.1,
        decoder_hidden_dims: Sequence[int] = (256, 128),
        decoder_dropout: float = 0.2,
        pretrained_backbone: bool = True,
        allow_pretrained_fallback: bool = False,
        use_hertz_residual: bool = False,
    ) -> None:
        super().__init__()
        if time_steps <= 0:
            raise ValueError("time_steps must be positive.")

        self.in_channels = in_channels
        self.time_steps = time_steps
        self.use_hertz_residual = use_hertz_residual
        self.encoder = Encoder2DResNetCrossAttention(
            in_channels=in_channels,
            embed_dim=image_embed_dim,
            attention_heads=attention_heads,
            attention_dropout=attention_dropout,
            pretrained_backbone=pretrained_backbone,
            allow_pretrained_fallback=allow_pretrained_fallback,
        )
        self.temporal_module = RheologyDynamicModuleDeform(
            in_channels=image_embed_dim,
            out_channels=image_embed_dim,
            temporal_kernel_size=time_steps,
        )

        physical_feature_dim = time_steps * 2
        decoder_input_dim = (
            image_embed_dim + physical_feature_dim + int(use_hertz_residual)
        )
        self.decoder = DecoderFC(
            input_dim=decoder_input_dim,
            hidden_dims=decoder_hidden_dims,
            dropout=decoder_dropout,
        )

    def _validate_physical_input(
        self,
        values: Tensor,
        name: str,
        batch_size: int,
    ) -> Tensor:
        if values.ndim == 3 and values.shape[-1] == 1:
            values = values.squeeze(-1)
        expected_shape = (batch_size, self.time_steps)
        if tuple(values.shape) != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {tuple(values.shape)}."
            )
        return values

    def forward(
        self,
        images: Tensor,
        forces: Tensor,
        widths: Tensor,
        hertz_estimate: Tensor | None = None,
    ) -> Tensor:
        """Return one normalized modulus estimate per sample."""
        if images.ndim == 4:
            batch_size, combined_channels, height, width = images.shape
            expected_channels = self.time_steps * self.in_channels
            if combined_channels != expected_channels:
                raise ValueError(
                    f"Flattened images need {expected_channels} channels, "
                    f"got {combined_channels}."
                )
            images = images.reshape(
                batch_size,
                self.time_steps,
                self.in_channels,
                height,
                width,
            )
        elif images.ndim != 5:
            raise ValueError(
                "images must have shape (B, T, C, H, W) or "
                f"(B, T*C, H, W), got {tuple(images.shape)}."
            )

        batch_size, time_steps, channels, height, width = images.shape
        if time_steps != self.time_steps or channels != self.in_channels:
            raise ValueError(
                "Unexpected image sequence shape: expected "
                f"T={self.time_steps}, C={self.in_channels}; "
                f"got T={time_steps}, C={channels}."
            )

        forces = self._validate_physical_input(
            forces,
            "forces",
            batch_size,
        )
        widths = self._validate_physical_input(
            widths,
            "widths",
            batch_size,
        )

        flattened_images = images.reshape(
            batch_size * time_steps,
            channels,
            height,
            width,
        )
        physical_values = torch.stack((forces, widths), dim=-1).reshape(
            batch_size * time_steps,
            2,
        )
        spatial_features = self.encoder(
            flattened_images,
            physical_values,
        )

        _, feature_channels, feature_height, feature_width = spatial_features.shape
        temporal_features = (
            spatial_features.reshape(
                batch_size,
                time_steps,
                feature_channels,
                feature_height,
                feature_width,
            )
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )
        aggregated_features = self.temporal_module(temporal_features)
        physical_skip = torch.cat((forces, widths), dim=1)
        decoder_input = torch.cat(
            (aggregated_features, physical_skip),
            dim=1,
        )

        if self.use_hertz_residual:
            if hertz_estimate is None:
                hertz_estimate = images.new_zeros((batch_size, 1))
            else:
                hertz_estimate = hertz_estimate.reshape(batch_size, 1)
            decoder_input = torch.cat(
                (decoder_input, hertz_estimate),
                dim=1,
            )
        return self.decoder(decoder_input)


# Compatibility aliases for code that imported the original class names.
Encoder2D_ResNet_CrossAttn = Encoder2DResNetCrossAttention
RheologyDynamicModule_Deform = RheologyDynamicModuleDeform


def _smoke_test() -> None:
    """Run a small, offline forward-pass check."""
    model = PhysiNet(
        in_channels=3,
        time_steps=3,
        image_embed_dim=32,
        pretrained_backbone=False,
    )
    model.eval()
    images = torch.randn(2, 3, 3, 64, 64)
    forces = torch.rand(2, 3)
    widths = torch.zeros(2, 3)
    with torch.no_grad():
        output = model(images, forces, widths)
    assert output.shape == (2,)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print("Model smoke test passed.")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Parameters: {parameter_count:,}")


if __name__ == "__main__":
    _smoke_test()
