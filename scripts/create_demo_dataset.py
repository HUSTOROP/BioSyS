"""Create a tiny synthetic dataset for an end-to-end installation check.

This dataset is only a software smoke test. It must not be used to report
scientific results.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "demo_data"


def make_tactile_image(
    size: int,
    force: float,
    modulus: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a simple RGB contact pattern correlated with force/modulus."""
    coordinates = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    radius = np.sqrt(x_grid**2 + y_grid**2)

    contact_radius = 0.20 + 0.012 * force + 0.006 * modulus
    indentation = np.clip(1.0 - radius / contact_radius, 0.0, 1.0)
    ring = np.exp(-((radius - contact_radius) ** 2) / 0.0025)
    noise = rng.normal(0.0, 2.0, size=(size, size)).astype(np.float32)

    red = 78.0 + 105.0 * indentation + 24.0 * ring + noise
    green = 92.0 + 82.0 * indentation + noise
    blue = 106.0 + 58.0 * indentation - 18.0 * ring + noise
    image = np.stack((red, green, blue), axis=-1)
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def create_dataset(
    output_directory: Path,
    groups_per_class: int,
    image_size: int,
    seed: int,
    overwrite: bool,
) -> Path:
    csv_path = output_directory / "tactile_dataset_demo.csv"
    image_directory = output_directory / "images"
    if csv_path.exists() and not overwrite:
        raise FileExistsError(
            f"{csv_path} already exists. Pass --overwrite to regenerate it."
        )

    image_directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    class_definitions = (
        (31.0, 1.19),
        (49.0, 2.37),
        (73.0, 6.37),
    )
    force_levels = (10.0, 20.0, 30.0)
    rows: list[dict[str, str | float]] = []

    for class_index, (shore_a, modulus) in enumerate(class_definitions):
        for sample_index in range(groups_per_class):
            group_id = f"class_{class_index}_sample_{sample_index:02d}"
            for frame_index, force in enumerate(force_levels):
                image = make_tactile_image(
                    size=image_size,
                    force=force,
                    modulus=modulus,
                    rng=rng,
                )
                filename = f"{group_id}_force_{frame_index:02d}.png"
                image_path = image_directory / filename
                Image.fromarray(image, mode="RGB").save(image_path)
                rows.append(
                    {
                        "group_id": group_id,
                        "image_path": f"images/{filename}",
                        "force_n": force,
                        "youngs_modulus_mpa": modulus,
                        "shore_a": shore_a,
                    }
                )

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "group_id",
                "image_path",
                "force_n",
                "youngs_modulus_mpa",
                "shore_a",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a synthetic HUST-SOFT-format demo dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument("--groups-per-class", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.groups_per_class < 2:
        raise ValueError("--groups-per-class must be at least 2.")
    if arguments.image_size < 32:
        raise ValueError("--image-size must be at least 32.")
    csv_path = create_dataset(
        output_directory=arguments.output_dir.expanduser().resolve(),
        groups_per_class=arguments.groups_per_class,
        image_size=arguments.image_size,
        seed=arguments.seed,
        overwrite=arguments.overwrite,
    )
    print(f"Demo dataset created: {csv_path}")
    print("This dataset is for software testing only.")


if __name__ == "__main__":
    main()
