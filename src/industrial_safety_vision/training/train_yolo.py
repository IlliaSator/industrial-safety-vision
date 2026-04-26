"""YOLO training CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def train_from_config(
    config_path: str | Path,
    *,
    dataset_path: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    image_size: int | None = None,
    model_checkpoint: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    dataset_yaml = dataset_path or config.get("dataset", {}).get("data_yaml")
    if not dataset_yaml or not Path(dataset_yaml).exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")

    checkpoint = model_checkpoint or config.get("model", {}).get("checkpoint", "yolov8n.pt")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        msg = "Ultralytics is required for training. Install project dependencies."
        raise RuntimeError(msg) from exc

    model = YOLO(checkpoint)
    results = model.train(
        data=dataset_yaml,
        imgsz=image_size or config.get("model", {}).get("image_size", 640),
        epochs=epochs or config.get("training", {}).get("epochs", 50),
        batch=batch_size or config.get("training", {}).get("batch_size", 16),
        device=device or config.get("training", {}).get("device", "auto"),
        project=config.get("dataset", {}).get("project_dir", "runs/train"),
    )
    metrics = _extract_metrics(results)
    metrics_path = Path(
        config.get("outputs", {}).get("metrics_path", "reports/training_metrics.json")
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _extract_metrics(results: Any) -> dict[str, Any]:
    if hasattr(results, "results_dict"):
        return dict(results.results_dict)
    if isinstance(results, dict):
        return results
    return {"status": "completed", "note": "Ultralytics did not expose metrics in a dict format."}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO safety detector.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_from_config(
        args.config,
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        model_checkpoint=args.model,
        device=args.device,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
