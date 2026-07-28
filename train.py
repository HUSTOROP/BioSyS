"""Train and evaluate the HUST-SOFT tactile modulus model.

Example:
    python train.py --config configs/default.yaml

The script expects a CSV file whose rows describe tactile frames. Frames with
the same ``group_id`` form one temporal sample.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from BioSyS import PhysiNet

LOGGER = logging.getLogger("hust_soft")
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate a YAML configuration file."""
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("The configuration root must be a YAML mapping.")
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    """Fail early when required configuration fields are missing or invalid."""
    required_sections = (
        "data",
        "model",
        "training",
        "evaluation",
        "output",
        "wandb",
    )
    missing_sections = [
        section for section in required_sections if section not in config
    ]
    if missing_sections:
        raise KeyError("Missing configuration sections: " + ", ".join(missing_sections))

    positive_values = {
        "data.n_frames": config["data"].get("n_frames"),
        "data.image_height": config["data"].get("image_height"),
        "data.image_width": config["data"].get("image_width"),
        "data.max_force": config["data"].get("max_force"),
        "model.image_embed_dim": config["model"].get("image_embed_dim"),
        "training.epochs": config["training"].get("epochs"),
        "training.batch_size": config["training"].get("batch_size"),
        "training.learning_rate": config["training"].get("learning_rate"),
    }
    invalid = [
        name
        for name, value in positive_values.items()
        if not isinstance(value, (int, float)) or value <= 0
    ]
    if invalid:
        raise ValueError(
            "These configuration values must be positive: " + ", ".join(invalid)
        )

    validation_ratio = config["data"].get("validation_ratio")
    if not isinstance(validation_ratio, (int, float)) or not (
        0.0 < validation_ratio < 1.0
    ):
        raise ValueError("data.validation_ratio must be between 0 and 1.")

    minimum_modulus = config["data"].get("min_modulus")
    maximum_modulus = config["data"].get("max_modulus")
    if minimum_modulus is None or maximum_modulus is None:
        raise KeyError("data.min_modulus and data.max_modulus are required.")
    if maximum_modulus <= minimum_modulus:
        raise ValueError("data.max_modulus must exceed data.min_modulus.")

    class_values = config["evaluation"].get("class_values_mpa")
    if not isinstance(class_values, list) or not class_values:
        raise ValueError("evaluation.class_values_mpa must be a non-empty list.")


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve a configuration path relative to the repository root."""
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def resolve_image_path(data_root: Path, raw_path: Any) -> Path:
    """Resolve Windows- or POSIX-style CSV image paths cross-platform."""
    if pd.isna(raw_path):
        raise ValueError("Encountered an empty image_path value.")

    normalized = str(raw_path).strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    supplied_path = Path(normalized).expanduser()

    if supplied_path.is_absolute():
        return supplied_path

    candidates = (
        data_root / supplied_path,
        data_root.parent / supplied_path,
        PROJECT_ROOT / supplied_path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def seed_everything(seed: int, deterministic: bool) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable data splits and training."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(
            deterministic,
            warn_only=True,
        )


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker from PyTorch's deterministic worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_device(requested_device: str) -> torch.device:
    """Resolve ``auto`` and validate explicitly requested CUDA devices."""
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"Device '{requested_device}' was requested, but CUDA is unavailable."
        )
    return device


def build_sequence_groups(
    csv_path: Path,
    data_config: Mapping[str, Any],
) -> list[list[dict[str, Any]]]:
    """Read the CSV and build one fixed-length sequence per ``group_id``."""
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset CSV not found: {csv_path}\n"
            "Update data.root/data.csv_filename in the configuration."
        )

    columns = data_config["columns"]
    required_logical_columns = (
        "group_id",
        "image_path",
        "force",
        "modulus",
        "shore_a",
    )
    missing_column_settings = [
        name for name in required_logical_columns if name not in columns
    ]
    if missing_column_settings:
        raise KeyError(
            "Missing data.columns settings: " + ", ".join(missing_column_settings)
        )

    dataframe = pd.read_csv(csv_path)
    required_csv_columns = {columns[name] for name in required_logical_columns}
    missing_csv_columns = sorted(required_csv_columns.difference(dataframe.columns))
    if missing_csv_columns:
        raise ValueError(
            f"CSV {csv_path} is missing columns: " + ", ".join(missing_csv_columns)
        )

    group_column = columns["group_id"]
    force_column = columns["force"]
    modulus_column = columns["modulus"]
    shore_column = columns["shore_a"]
    number_of_frames = int(data_config["n_frames"])
    groups: list[list[dict[str, Any]]] = []
    skipped_short_groups = 0

    for group_id, group_frame in dataframe.groupby(
        group_column,
        sort=False,
        dropna=False,
    ):
        if pd.isna(group_id):
            raise ValueError("group_id contains a missing value.")
        if len(group_frame) < number_of_frames:
            skipped_short_groups += 1
            continue

        modulus_values = pd.to_numeric(
            group_frame[modulus_column],
            errors="raise",
        ).to_numpy(dtype=np.float64)
        shore_values = pd.to_numeric(
            group_frame[shore_column],
            errors="raise",
        ).to_numpy(dtype=np.float64)
        if not np.allclose(modulus_values, modulus_values[0]):
            raise ValueError(
                f"group_id={group_id!r} contains inconsistent modulus labels."
            )
        if not np.allclose(shore_values, shore_values[0]):
            raise ValueError(
                f"group_id={group_id!r} contains inconsistent Shore A labels."
            )

        ordered = group_frame.assign(
            __force_numeric=pd.to_numeric(
                group_frame[force_column],
                errors="raise",
            )
        ).sort_values("__force_numeric")
        selected = ordered.tail(number_of_frames).drop(columns="__force_numeric")
        groups.append(selected.to_dict(orient="records"))

    if not groups:
        raise ValueError(
            f"No valid {number_of_frames}-frame sequences were found in {csv_path}."
        )
    if len(groups) < 2:
        raise ValueError(
            "At least two valid sequences are required for a train/validation split."
        )
    if skipped_short_groups:
        LOGGER.warning(
            "Skipped %d groups with fewer than %d frames.",
            skipped_short_groups,
            number_of_frames,
        )
    LOGGER.info("Loaded %d valid tactile sequences.", len(groups))
    return groups


def validate_image_files(
    groups: Sequence[Sequence[Mapping[str, Any]]],
    data_root: Path,
    image_column: str,
) -> None:
    """Check every referenced image before training starts."""
    missing_paths: list[Path] = []
    for group in groups:
        for frame in group:
            image_path = resolve_image_path(
                data_root,
                frame[image_column],
            )
            if not image_path.is_file():
                missing_paths.append(image_path)
                if len(missing_paths) >= 10:
                    break
        if len(missing_paths) >= 10:
            break

    if missing_paths:
        formatted = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(
            "Images referenced by the CSV could not be found. "
            "First unresolved paths:\n"
            f"{formatted}\n"
            "Store image_path values relative to data.root or update data.root."
        )


class TactileSequenceDataset(Dataset):
    """Load a fixed-length tactile image and force sequence."""

    def __init__(
        self,
        groups: Sequence[Sequence[Mapping[str, Any]]],
        data_root: Path,
        data_config: Mapping[str, Any],
        augment: bool,
    ) -> None:
        self.groups = groups
        self.data_root = data_root
        self.data_config = data_config
        self.augment = augment
        self.number_of_frames = int(data_config["n_frames"])
        self.image_height = int(data_config["image_height"])
        self.image_width = int(data_config["image_width"])
        self.maximum_force = float(data_config["max_force"])
        self.columns = data_config["columns"]

    def __len__(self) -> int:
        augmentation_factor = 4 if self.augment else 1
        return len(self.groups) * augmentation_factor

    def __getitem__(
        self,
        index: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if self.augment:
            group_index, augmentation_mode = divmod(index, 4)
        else:
            group_index, augmentation_mode = index, 0

        group = self.groups[group_index]
        image_sequence = torch.empty(
            (
                self.number_of_frames,
                3,
                self.image_height,
                self.image_width,
            ),
            dtype=torch.float32,
        )
        forces = torch.empty(self.number_of_frames, dtype=torch.float32)

        for frame_index, frame in enumerate(group):
            image_path = resolve_image_path(
                self.data_root,
                frame[self.columns["image_path"]],
            )
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image = image.resize(
                    (self.image_width, self.image_height),
                    resample=Image.Resampling.BILINEAR,
                )
                image_array = np.asarray(image, dtype=np.float32).copy() / 255.0

            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
            if augmentation_mode == 1:
                image_tensor = torch.flip(image_tensor, dims=(2,))
            elif augmentation_mode == 2:
                image_tensor = torch.flip(image_tensor, dims=(1,))
            elif augmentation_mode == 3:
                image_tensor = torch.flip(image_tensor, dims=(1, 2))
            image_sequence[frame_index] = image_tensor

            force = float(frame[self.columns["force"]])
            forces[frame_index] = force / self.maximum_force

        # Width measurements were not present in the original HUST-SOFT CSV.
        # Keeping this input at zero preserves the published model interface.
        widths = torch.zeros(self.number_of_frames, dtype=torch.float32)
        modulus = torch.tensor(
            float(group[0][self.columns["modulus"]]),
            dtype=torch.float32,
        )
        shore_a = torch.tensor(
            float(group[0][self.columns["shore_a"]]),
            dtype=torch.float32,
        )
        return image_sequence, forces, widths, modulus, shore_a


def split_groups(
    groups: Sequence[Sequence[Mapping[str, Any]]],
    data_config: Mapping[str, Any],
    seed: int,
) -> tuple[list[Any], list[Any]]:
    """Split at sequence level to prevent frames from leaking across sets."""
    validation_ratio = float(data_config["validation_ratio"])
    stratify = None
    if data_config.get("stratified_split", False):
        modulus_column = data_config["columns"]["modulus"]
        labels = [float(group[0][modulus_column]) for group in groups]
        unique_labels, counts = np.unique(labels, return_counts=True)
        validation_count = math.ceil(len(groups) * validation_ratio)
        if np.all(counts >= 2) and validation_count >= len(unique_labels):
            stratify = labels
        else:
            LOGGER.warning(
                "Stratification was requested but class counts are too small; "
                "using a seeded random split instead."
            )

    train_groups, validation_groups = train_test_split(
        list(groups),
        test_size=validation_ratio,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    LOGGER.info(
        "Split sequences into %d training and %d validation samples.",
        len(train_groups),
        len(validation_groups),
    )
    return train_groups, validation_groups


def torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    """Load checkpoints on both newer and older supported PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


class ModulusTrainer:
    """Own the data loaders, optimization loop, checkpoints, and plots."""

    def __init__(
        self,
        config: dict[str, Any],
        device: torch.device,
        resume_path: Path | None = None,
    ) -> None:
        self.config = config
        self.device = device
        self.resume_path = resume_path
        self.data_config = config["data"]
        self.training_config = config["training"]
        self.evaluation_config = config["evaluation"]

        self.data_root = resolve_project_path(self.data_config["root"])
        csv_value = Path(self.data_config["csv_filename"])
        self.csv_path = (
            csv_value.resolve()
            if csv_value.is_absolute()
            else self.data_root / csv_value
        )
        self.output_root = resolve_project_path(config["output"]["root"])
        self.run_directory = self.output_root / config["output"]["run_name"]
        self.run_directory.mkdir(parents=True, exist_ok=True)

        self.minimum_modulus = float(self.data_config["min_modulus"])
        self.maximum_modulus = float(self.data_config["max_modulus"])
        self.class_values = np.asarray(
            self.evaluation_config["class_values_mpa"],
            dtype=np.float64,
        )

        model_config = config["model"]
        self.model = PhysiNet(
            in_channels=3,
            time_steps=int(self.data_config["n_frames"]),
            image_embed_dim=int(model_config["image_embed_dim"]),
            attention_heads=int(model_config["attention_heads"]),
            attention_dropout=float(model_config["attention_dropout"]),
            decoder_hidden_dims=tuple(
                int(value) for value in model_config["decoder_hidden_dims"]
            ),
            decoder_dropout=float(model_config["decoder_dropout"]),
            pretrained_backbone=bool(model_config["pretrained_backbone"]),
            allow_pretrained_fallback=bool(
                model_config.get("allow_pretrained_fallback", False)
            ),
            use_hertz_residual=False,
        ).to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.training_config["learning_rate"]),
        )
        self.loss_function = nn.MSELoss()
        self.best_mape = math.inf
        self.start_epoch = 0
        self.wandb_run: Any = None
        self.wandb_module: Any = None

        if resume_path is not None:
            self._resume(resume_path)
        self._initialize_wandb()
        self._save_resolved_config()

    def _initialize_wandb(self) -> None:
        wandb_config = self.config["wandb"]
        if not wandb_config.get("enabled", False):
            return
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "W&B logging was enabled. Install it with `pip install wandb` "
                "or set wandb.enabled=false."
            ) from exc

        self.wandb_module = wandb
        self.wandb_run = wandb.init(
            project=wandb_config["project"],
            name=self.config["output"]["run_name"],
            config=self.config,
        )

    def _save_resolved_config(self) -> None:
        config_path = self.run_directory / "resolved_config.yaml"
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                self.config,
                handle,
                sort_keys=False,
                allow_unicode=True,
            )

    def _resume(self, checkpoint_path: Path) -> None:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch_load(checkpoint_path, self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.best_mape = float(checkpoint.get("best_mape", math.inf))
        LOGGER.info(
            "Resumed %s at epoch %d.",
            checkpoint_path,
            self.start_epoch + 1,
        )

    def _normalize(self, modulus: Tensor) -> Tensor:
        scale = self.maximum_modulus - self.minimum_modulus
        return (modulus - self.minimum_modulus) / scale

    def _unnormalize(self, normalized_modulus: Tensor) -> Tensor:
        scale = self.maximum_modulus - self.minimum_modulus
        return normalized_modulus * scale + self.minimum_modulus

    def _create_loader(
        self,
        dataset: Dataset,
        shuffle: bool,
    ) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(int(self.training_config["seed"]))
        number_of_workers = int(self.training_config["num_workers"])
        return DataLoader(
            dataset,
            batch_size=int(self.training_config["batch_size"]),
            shuffle=shuffle,
            num_workers=number_of_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=number_of_workers > 0,
            worker_init_fn=seed_worker if number_of_workers > 0 else None,
            generator=generator,
            drop_last=False,
        )

    def load_data(self) -> None:
        groups = build_sequence_groups(self.csv_path, self.data_config)
        validate_image_files(
            groups,
            self.data_root,
            self.data_config["columns"]["image_path"],
        )
        train_groups, validation_groups = split_groups(
            groups,
            self.data_config,
            int(self.training_config["seed"]),
        )

        self.train_groups = train_groups
        self.validation_groups = validation_groups
        self.train_dataset = TactileSequenceDataset(
            train_groups,
            self.data_root,
            self.data_config,
            augment=bool(self.data_config["augmentation"]),
        )
        self.clean_train_dataset = TactileSequenceDataset(
            train_groups,
            self.data_root,
            self.data_config,
            augment=False,
        )
        self.validation_dataset = TactileSequenceDataset(
            validation_groups,
            self.data_root,
            self.data_config,
            augment=False,
        )
        self.train_loader = self._create_loader(
            self.train_dataset,
            shuffle=True,
        )
        self.clean_train_loader = self._create_loader(
            self.clean_train_dataset,
            shuffle=False,
        )
        self.validation_loader = self._create_loader(
            self.validation_dataset,
            shuffle=False,
        )

    def _move_batch(
        self,
        batch: Sequence[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        images, forces, widths, modulus, shore_a = batch
        non_blocking = self.device.type == "cuda"
        return (
            images.to(self.device, non_blocking=non_blocking),
            forces.to(self.device, non_blocking=non_blocking),
            widths.to(self.device, non_blocking=non_blocking),
            modulus.to(self.device, non_blocking=non_blocking),
            shore_a,
        )

    def _regularization_term(self) -> Tensor:
        coefficient = float(self.training_config["l2_lambda"])
        if coefficient == 0.0:
            return torch.zeros((), device=self.device)
        norms = [
            parameter.norm(p=2)
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        return coefficient * torch.stack(norms).sum()

    def train_epoch(self) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_absolute_error = 0.0
        total_squared_error = 0.0
        sample_count = 0

        progress = tqdm(
            self.train_loader,
            desc="Train",
            leave=False,
        )
        for raw_batch in progress:
            images, forces, widths, modulus, _ = self._move_batch(raw_batch)
            self.optimizer.zero_grad(set_to_none=True)
            normalized_targets = self._normalize(modulus)
            normalized_predictions = self.model(images, forces, widths)
            regression_loss = self.loss_function(
                normalized_predictions,
                normalized_targets,
            )
            loss = regression_loss + self._regularization_term()
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                predictions = self._unnormalize(normalized_predictions)
                errors = predictions - modulus
                batch_size = images.shape[0]
                total_loss += loss.item() * batch_size
                total_absolute_error += errors.abs().sum().item()
                total_squared_error += errors.square().sum().item()
                sample_count += batch_size

        if sample_count == 0:
            raise RuntimeError("The training DataLoader produced no samples.")
        return {
            "loss": total_loss / sample_count,
            "mae": total_absolute_error / sample_count,
            "rmse": math.sqrt(total_squared_error / sample_count),
        }

    def evaluate_loader(
        self,
        loader: DataLoader,
        description: str,
        collect_predictions: bool = False,
    ) -> tuple[dict[str, float], dict[str, np.ndarray] | None]:
        self.model.eval()
        total_loss = 0.0
        total_absolute_error = 0.0
        total_absolute_percentage_error = 0.0
        total_squared_error = 0.0
        sample_count = 0
        predictions_list: list[float] = []
        labels_list: list[float] = []
        shores_list: list[float] = []

        with torch.no_grad():
            for raw_batch in tqdm(loader, desc=description, leave=False):
                images, forces, widths, modulus, shore_a = self._move_batch(raw_batch)
                normalized_targets = self._normalize(modulus)
                normalized_predictions = self.model(images, forces, widths)
                loss = self.loss_function(
                    normalized_predictions,
                    normalized_targets,
                )
                predictions = self._unnormalize(normalized_predictions)
                errors = predictions - modulus
                batch_size = images.shape[0]

                total_loss += loss.item() * batch_size
                total_absolute_error += errors.abs().sum().item()
                total_absolute_percentage_error += (
                    (errors.abs() / modulus.abs().clamp_min(1e-6)).sum().item()
                )
                total_squared_error += errors.square().sum().item()
                sample_count += batch_size

                if collect_predictions:
                    predictions_list.extend(predictions.detach().cpu().tolist())
                    labels_list.extend(modulus.detach().cpu().tolist())
                    shores_list.extend(shore_a.detach().cpu().tolist())

        if sample_count == 0:
            raise RuntimeError(f"The {description} DataLoader was empty.")
        metrics = {
            "loss": total_loss / sample_count,
            "mae": total_absolute_error / sample_count,
            "mape": total_absolute_percentage_error / sample_count,
            "rmse": math.sqrt(total_squared_error / sample_count),
        }
        if not collect_predictions:
            return metrics, None
        arrays = {
            "predictions": np.asarray(predictions_list, dtype=np.float64),
            "labels": np.asarray(labels_list, dtype=np.float64),
            "shores": np.asarray(shores_list, dtype=np.float64),
        }
        return metrics, arrays

    def _checkpoint_payload(
        self,
        epoch: int,
        metrics: Mapping[str, float],
    ) -> dict[str, Any]:
        return {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_mape": self.best_mape,
            "metrics": dict(metrics),
            "config": self.config,
        }

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: Mapping[str, float],
    ) -> None:
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            self._checkpoint_payload(epoch, metrics),
            temporary_path,
        )
        temporary_path.replace(path)

    def _append_metrics(
        self,
        epoch: int,
        train_metrics: Mapping[str, float],
        validation_metrics: Mapping[str, float],
    ) -> None:
        metrics_path = self.run_directory / "metrics.csv"
        fieldnames = (
            "epoch",
            "train_loss",
            "train_mae_mpa",
            "train_rmse_mpa",
            "validation_loss",
            "validation_mae_mpa",
            "validation_rmse_mpa",
            "validation_mape",
        )
        write_header = not metrics_path.exists()
        with metrics_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_metrics["loss"],
                    "train_mae_mpa": train_metrics["mae"],
                    "train_rmse_mpa": train_metrics["rmse"],
                    "validation_loss": validation_metrics["loss"],
                    "validation_mae_mpa": validation_metrics["mae"],
                    "validation_rmse_mpa": validation_metrics["rmse"],
                    "validation_mape": validation_metrics["mape"],
                }
            )

    def make_performance_plot(self, epoch: int | None = None) -> Path:
        """Plot clean training and validation predictions in MPa."""
        _, train_data = self.evaluate_loader(
            self.clean_train_loader,
            "Clean train predictions",
            collect_predictions=True,
        )
        _, validation_data = self.evaluate_loader(
            self.validation_loader,
            "Validation predictions",
            collect_predictions=True,
        )
        assert train_data is not None and validation_data is not None

        plt.style.use("default")
        figure, axis = plt.subplots(figsize=(10, 8))
        maximum_axis = self.maximum_modulus
        axis.plot(
            [self.minimum_modulus, maximum_axis],
            [self.minimum_modulus, maximum_axis],
            "k--",
            alpha=0.6,
            label="Ideal (y = x)",
        )
        axis.scatter(
            train_data["labels"],
            train_data["predictions"],
            color="0.55",
            marker="x",
            alpha=0.35,
            s=28,
            label="Training samples",
        )

        all_shores = np.unique(validation_data["shores"])
        colors = plt.get_cmap("viridis")(
            np.linspace(0.05, 0.95, max(len(all_shores), 1))
        )
        for color, shore_value in zip(colors, all_shores, strict=True):
            mask = np.isclose(validation_data["shores"], shore_value)
            axis.scatter(
                validation_data["labels"][mask],
                validation_data["predictions"][mask],
                color=color,
                marker="o",
                alpha=0.75,
                s=34,
                edgecolors="white",
                linewidths=0.4,
                label=f"Validation: Shore A {shore_value:g}",
            )

        axis.set_xlim(self.minimum_modulus, maximum_axis)
        axis.set_ylim(self.minimum_modulus, maximum_axis)
        axis.set_xlabel("Ground-truth Young's modulus (MPa)")
        axis.set_ylabel("Predicted Young's modulus (MPa)")
        axis.set_title("Young's modulus estimation")
        axis.grid(linestyle="--", alpha=0.4)
        axis.legend(fontsize=8, ncols=2)
        figure.tight_layout()

        suffix = f"_epoch_{epoch:03d}" if epoch is not None else ""
        output_path = self.run_directory / f"regression_plot{suffix}.png"
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        LOGGER.info("Saved regression plot to %s.", output_path)
        return output_path

    def evaluate_classification(self) -> dict[str, Any]:
        """Snap validation predictions to known modulus classes and plot them."""
        _, data = self.evaluate_loader(
            self.validation_loader,
            "Classification evaluation",
            collect_predictions=True,
        )
        assert data is not None
        true_indices = np.abs(
            data["labels"][:, None] - self.class_values[None, :]
        ).argmin(axis=1)
        predicted_indices = np.abs(
            data["predictions"][:, None] - self.class_values[None, :]
        ).argmin(axis=1)
        accuracy = float(np.mean(true_indices == predicted_indices))
        labels = np.arange(len(self.class_values))
        matrix = confusion_matrix(
            true_indices,
            predicted_indices,
            labels=labels,
        )

        tick_labels = [f"{value:.2f}" for value in self.class_values]
        figure, axis = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=tick_labels,
            yticklabels=tick_labels,
            ax=axis,
        )
        axis.set_title(f"Validation confusion matrix (accuracy: {accuracy:.2%})")
        axis.set_xlabel("Predicted Young's modulus class (MPa)")
        axis.set_ylabel("Ground-truth Young's modulus class (MPa)")
        axis.tick_params(axis="x", rotation=45)
        axis.tick_params(axis="y", rotation=0)
        figure.tight_layout()

        image_path = self.run_directory / "confusion_matrix.png"
        figure.savefig(image_path, dpi=300, bbox_inches="tight")
        plt.close(figure)

        result = {
            "validation_accuracy": accuracy,
            "correct_predictions": int(np.sum(true_indices == predicted_indices)),
            "sample_count": len(true_indices),
            "class_values_mpa": self.class_values.tolist(),
            "confusion_matrix": matrix.tolist(),
        }
        result_path = self.run_directory / "classification_metrics.json"
        with result_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        LOGGER.info(
            "Validation classification accuracy: %.2f%% (%d/%d).",
            accuracy * 100.0,
            result["correct_predictions"],
            result["sample_count"],
        )

        if self.wandb_run is not None:
            self.wandb_module.log(
                {
                    "validation_classification_accuracy": accuracy,
                    "confusion_matrix": self.wandb_module.Image(str(image_path)),
                }
            )
        return result

    def dry_run(self) -> None:
        """Validate data loading and one model forward pass without training."""
        self.load_data()
        raw_batch = next(iter(self.train_loader))
        images, forces, widths, modulus, _ = self._move_batch(raw_batch)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(images, forces, widths)
            loss = self.loss_function(
                predictions,
                self._normalize(modulus),
            )
        LOGGER.info("Dry run passed.")
        LOGGER.info("Image batch shape: %s", tuple(images.shape))
        LOGGER.info("Prediction shape: %s", tuple(predictions.shape))
        LOGGER.info("Batch loss: %.6f", loss.item())

    def train(self) -> None:
        self.load_data()
        metrics_path = self.run_directory / "metrics.csv"
        if self.resume_path is None and metrics_path.exists():
            metrics_path.unlink()

        total_epochs = int(self.training_config["epochs"])
        plot_every = int(self.training_config["plot_every"])
        LOGGER.info(
            "Training on %s for %d epochs. Outputs: %s",
            self.device,
            total_epochs,
            self.run_directory,
        )

        for epoch in range(self.start_epoch, total_epochs):
            train_metrics = self.train_epoch()
            validation_metrics, _ = self.evaluate_loader(
                self.validation_loader,
                "Validation",
            )
            LOGGER.info(
                "Epoch %d/%d | "
                "train loss %.5f, MAE %.4f, RMSE %.4f MPa | "
                "val loss %.5f, MAE %.4f, RMSE %.4f MPa, MAPE %.2f%%",
                epoch + 1,
                total_epochs,
                train_metrics["loss"],
                train_metrics["mae"],
                train_metrics["rmse"],
                validation_metrics["loss"],
                validation_metrics["mae"],
                validation_metrics["rmse"],
                validation_metrics["mape"] * 100.0,
            )

            self._append_metrics(epoch, train_metrics, validation_metrics)
            is_best = validation_metrics["mape"] < self.best_mape
            if is_best:
                self.best_mape = validation_metrics["mape"]
            self._save_checkpoint(
                self.run_directory / "last_checkpoint.pt",
                epoch,
                validation_metrics,
            )
            if is_best:
                self._save_checkpoint(
                    self.run_directory / "best_checkpoint.pt",
                    epoch,
                    validation_metrics,
                )
                torch.save(
                    self.model.state_dict(),
                    self.run_directory / "best_model_state_dict.pt",
                )
                LOGGER.info(
                    "Saved a new best model (MAPE %.2f%%).",
                    self.best_mape * 100.0,
                )

            if self.wandb_run is not None:
                self.wandb_module.log(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_metrics["loss"],
                        "train_mae_mpa": train_metrics["mae"],
                        "train_rmse_mpa": train_metrics["rmse"],
                        "validation_loss": validation_metrics["loss"],
                        "validation_mae_mpa": validation_metrics["mae"],
                        "validation_rmse_mpa": validation_metrics["rmse"],
                        "validation_mape_percent": (validation_metrics["mape"] * 100.0),
                    }
                )

            should_plot = plot_every > 0 and (
                (epoch + 1) % plot_every == 0 or epoch + 1 == total_epochs
            )
            if should_plot:
                self.make_performance_plot(epoch=epoch + 1)

        best_checkpoint_path = self.run_directory / "best_checkpoint.pt"
        if not best_checkpoint_path.is_file():
            raise RuntimeError("Training ended without a best checkpoint.")
        best_checkpoint = torch_load(best_checkpoint_path, self.device)
        self.model.load_state_dict(best_checkpoint["model_state"])
        self.make_performance_plot()
        self.evaluate_classification()
        LOGGER.info("Training finished. Best MAPE: %.2f%%.", self.best_mape * 100)

    def finish(self) -> None:
        if self.wandb_run is not None:
            self.wandb_run.finish()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the HUST-SOFT tactile modulus model."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument("--data-dir", type=str, help="Override data.root.")
    parser.add_argument(
        "--csv",
        type=str,
        help="Override data.csv_filename.",
    )
    parser.add_argument("--epochs", type=int, help="Override training.epochs.")
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override training.batch_size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Override training.num_workers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Override training.device (for example: auto, cpu, cuda:0).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Override output.run_name.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output.root.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not load ImageNet ResNet-18 weights.",
    )
    parser.add_argument(
        "--allow-pretrained-fallback",
        action="store_true",
        help="Use random weights if pretrained weights cannot be downloaded.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume from a checkpoint.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check CSV/images and run one forward pass without training.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def apply_command_line_overrides(
    config: dict[str, Any],
    arguments: argparse.Namespace,
) -> None:
    """Apply only overrides explicitly supplied by the user."""
    if arguments.data_dir is not None:
        config["data"]["root"] = arguments.data_dir
    if arguments.csv is not None:
        config["data"]["csv_filename"] = arguments.csv
    if arguments.epochs is not None:
        config["training"]["epochs"] = arguments.epochs
    if arguments.batch_size is not None:
        config["training"]["batch_size"] = arguments.batch_size
    if arguments.num_workers is not None:
        config["training"]["num_workers"] = arguments.num_workers
    if arguments.device is not None:
        config["training"]["device"] = arguments.device
    if arguments.run_name is not None:
        config["output"]["run_name"] = arguments.run_name
    if arguments.output_dir is not None:
        config["output"]["root"] = arguments.output_dir
    if arguments.no_pretrained:
        config["model"]["pretrained_backbone"] = False
    if arguments.allow_pretrained_fallback:
        config["model"]["allow_pretrained_fallback"] = True
    if arguments.wandb:
        config["wandb"]["enabled"] = True


def main() -> None:
    arguments = parse_arguments()
    logging.basicConfig(
        level=getattr(logging, arguments.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    config = load_config(arguments.config.resolve())
    apply_command_line_overrides(config, arguments)
    validate_config(config)

    seed = int(config["training"]["seed"])
    deterministic = bool(config["training"]["deterministic"])
    seed_everything(seed, deterministic)
    device = resolve_device(config["training"]["device"])
    LOGGER.info("Using device: %s", device)

    resume_path = (
        arguments.resume.expanduser().resolve()
        if arguments.resume is not None
        else None
    )
    trainer = ModulusTrainer(config, device, resume_path=resume_path)
    try:
        if arguments.dry_run:
            trainer.dry_run()
        else:
            trainer.train()
    finally:
        trainer.finish()


if __name__ == "__main__":
    main()
