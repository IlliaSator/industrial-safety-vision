"""Create train/val/test splits for YOLO datasets."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from industrial_safety_vision.data.dataset_validation import IMAGE_EXTENSIONS


def split_yolo_dataset(
    source_images: str | Path,
    source_labels: str | Path,
    output_root: str | Path,
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    if round(train_ratio + val_ratio + test_ratio, 6) != 1.0:
        raise ValueError("Split ratios must sum to 1.0")

    source_images = Path(source_images)
    source_labels = Path(source_labels)
    output_root = Path(output_root)
    images = sorted(
        path for path in source_images.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    random.Random(seed).shuffle(images)

    train_end = int(len(images) * train_ratio)
    val_end = train_end + int(len(images) * val_ratio)
    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }

    for split, split_images in splits.items():
        image_out = output_root / "images" / split
        label_out = output_root / "labels" / split
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)
        for image_path in split_images:
            shutil.copy2(image_path, image_out / image_path.name)
            label_path = source_labels / f"{image_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, label_out / label_path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split YOLO images and labels into train/val/test."
    )
    parser.add_argument("--images", required=True, help="Source image directory.")
    parser.add_argument("--labels", required=True, help="Source label directory.")
    parser.add_argument("--output", default="data/processed", help="Output YOLO dataset root.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_yolo_dataset(
        args.images,
        args.labels,
        args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(f"Dataset split written to {args.output}")


if __name__ == "__main__":
    main()
