"""YOLO evaluation CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def evaluate_from_config(
    config_path: str | Path,
    *,
    model_checkpoint: str | None = None,
    dataset_path: str | None = None,
    split: str = "val",
    device: str | None = None,
) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    dataset_yaml = dataset_path or config.get("dataset", {}).get("data_yaml")
    checkpoint = model_checkpoint or config.get("model", {}).get("checkpoint")
    if not dataset_yaml or not Path(dataset_yaml).exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")
    if not checkpoint:
        raise FileNotFoundError("Model checkpoint is required for evaluation.")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is required for evaluation. Install project dependencies.") from exc

    model = YOLO(checkpoint)
    results = model.val(data=dataset_yaml, split=split, device=device or config.get("training", {}).get("device"))
    metrics = _extract_detection_metrics(results)
    output_path = Path("reports/evaluation_metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _extract_detection_metrics(results: Any) -> dict[str, Any]:
    result_dict = dict(getattr(results, "results_dict", {}) or {})
    box = getattr(results, "box", None)
    metrics = {
        "map50": getattr(box, "map50", result_dict.get("metrics/mAP50(B)")),
        "map50_95": getattr(box, "map", result_dict.get("metrics/mAP50-95(B)")),
        "precision": result_dict.get("metrics/precision(B)"),
        "recall": result_dict.get("metrics/recall(B)"),
        "raw": result_dict,
    }
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a YOLO safety detector.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_from_config(
        args.config,
        model_checkpoint=args.model,
        dataset_path=args.dataset,
        split=args.split,
        device=args.device,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
