"""Dataset-format tests using a generated temporary data directory."""

from pathlib import Path

from scripts.create_demo_dataset import create_dataset
from train import (
    TactileSequenceDataset,
    build_sequence_groups,
    load_config,
    split_groups,
    validate_image_files,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_demo_data_pipeline(tmp_path: Path) -> None:
    csv_path = create_dataset(
        output_directory=tmp_path,
        groups_per_class=4,
        image_size=64,
        seed=27,
        overwrite=False,
    )
    config = load_config(PROJECT_ROOT / "configs" / "demo.yaml")
    data_config = config["data"]
    groups = build_sequence_groups(csv_path, data_config)
    validate_image_files(
        groups,
        tmp_path,
        data_config["columns"]["image_path"],
    )
    train_groups, validation_groups = split_groups(
        groups,
        data_config,
        seed=27,
    )
    dataset = TactileSequenceDataset(
        train_groups,
        tmp_path,
        data_config,
        augment=False,
    )
    images, forces, widths, modulus, shore_a = dataset[0]

    assert images.shape == (3, 3, 64, 64)
    assert forces.shape == (3,)
    assert widths.shape == (3,)
    assert modulus.ndim == 0
    assert shore_a.ndim == 0
    assert len(validation_groups) > 0
