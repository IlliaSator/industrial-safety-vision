"""Dataset YAML helpers for Ultralytics entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def materialize_ultralytics_dataset_yaml(
    dataset_yaml: str | Path,
    *,
    output_path: str | Path = "reports/ultralytics_dataset.yaml",
) -> Path:
    """Write a dataset YAML whose `path` is absolute.

    Some public YOLO datasets use `path: .`. Ultralytics may resolve that
    relative to the current process instead of the YAML file location, so local
    training can fail when the dataset is outside the repository root.
    """

    dataset_yaml = Path(dataset_yaml)
    payload: dict[str, Any] = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8")) or {}
    dataset_root = Path(payload.get("path", "."))
    if not dataset_root.is_absolute():
        dataset_root = (dataset_yaml.parent / dataset_root).resolve()
    payload["path"] = str(dataset_root)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output_path
