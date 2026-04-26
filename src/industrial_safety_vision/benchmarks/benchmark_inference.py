"""Inference latency benchmark utilities."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from industrial_safety_vision.inference.detector import YOLODetector
from industrial_safety_vision.inference.mock_detector import MockSafetyDetector
from industrial_safety_vision.inference.onnx_detector import ONNXDetector
from industrial_safety_vision.safety.danger_zone import DangerZone
from industrial_safety_vision.safety.rules import (
    DangerZoneRuleConfig,
    PPERuleConfig,
    SafetyRulesConfig,
    SafetyRulesEngine,
    VehicleProximityRuleConfig,
)
from industrial_safety_vision.tracking.tracker import SimpleIoUTracker
from industrial_safety_vision.utils.image_io import read_image


def benchmark_callable(
    fn: Callable[[], Any],
    *,
    warmup_runs: int,
    benchmark_runs: int,
) -> dict[str, float]:
    for _ in range(warmup_runs):
        fn()
    latencies = []
    for _ in range(benchmark_runs):
        started = time.perf_counter()
        fn()
        latencies.append((time.perf_counter() - started) * 1000.0)
    mean_latency = statistics.mean(latencies)
    return {
        "mean_latency_ms": mean_latency,
        "p50_latency_ms": statistics.median(latencies),
        "p95_latency_ms": _percentile(latencies, 95),
        "fps": 1000.0 / mean_latency if mean_latency else 0.0,
    }


def run_benchmark(
    model_path: str | Path | None = None,
    *,
    onnx_model_path: str | Path | None = None,
    image_path: str | Path | None = None,
    input_size: int = 640,
    warmup_runs: int = 5,
    benchmark_runs: int = 20,
    device: str = "cpu",
    output_path: str | Path = "reports/benchmark_results.json",
    markdown_report_path: str | Path | None = "docs/benchmark_report.md",
    mock: bool = False,
) -> dict[str, Any]:
    frame = _load_benchmark_frame(image_path=image_path, input_size=input_size)
    results: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "mode": "mock_pipeline" if mock else "real_model",
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "input_size": input_size,
        "warmup_runs": warmup_runs,
        "benchmark_runs": benchmark_runs,
        "backends": [],
    }

    if mock:
        detector = MockSafetyDetector(include_helmet=False, moving_person=True)
        tracker = SimpleIoUTracker(iou_threshold=0.1, max_missed_frames=3)
        safety_engine = _build_mock_safety_engine()
        mock_metrics = benchmark_callable(
            lambda: _run_mock_pipeline(frame, detector, tracker, safety_engine),
            warmup_runs=warmup_runs,
            benchmark_runs=benchmark_runs,
        )
        results["backends"].append(
            {
                "backend": "mock_detector_tracking_rules",
                "device": "cpu",
                "model_size_mb": None,
                **mock_metrics,
            }
        )
        return _write_benchmark_outputs(results, output_path, markdown_report_path)

    if model_path is None:
        raise FileNotFoundError("A real model path is required unless --mock is used.")
    model_path = Path(model_path)
    model_size_mb = _file_size_mb(model_path) if model_path.exists() else None

    detector = YOLODetector(str(model_path), device=device)
    pytorch_metrics = benchmark_callable(
        lambda: detector.predict_frame(frame),
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs,
    )
    results["backends"].append(
        {
            "backend": "pytorch_ultralytics",
            "device": device,
            "model_size_mb": model_size_mb,
            "model": str(model_path),
            **pytorch_metrics,
        }
    )

    if onnx_model_path is not None and Path(onnx_model_path).exists():
        onnx_detector = ONNXDetector(onnx_model_path, input_size=input_size)
        onnx_metrics = benchmark_callable(
            lambda: onnx_detector.run_raw(frame),
            warmup_runs=warmup_runs,
            benchmark_runs=benchmark_runs,
        )
        results["backends"].append(
            {
                "backend": "onnxruntime",
                "device": "cpu",
                "model_size_mb": _file_size_mb(Path(onnx_model_path)),
                **onnx_metrics,
            }
        )

    return _write_benchmark_outputs(results, output_path, markdown_report_path)


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, round((percentile / 100) * (len(sorted_values) - 1)))
    return sorted_values[index]


def _file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def _load_benchmark_frame(image_path: str | Path | None, *, input_size: int) -> np.ndarray:
    if image_path is not None:
        return read_image(image_path)
    return np.zeros((input_size, input_size, 3), dtype=np.uint8)


def _build_mock_safety_engine() -> SafetyRulesEngine:
    return SafetyRulesEngine(
        SafetyRulesConfig(
            missing_helmet=PPERuleConfig(consecutive_frames=3, cooldown_frames=20),
            missing_vest=PPERuleConfig(enabled=False),
            danger_zone=DangerZoneRuleConfig(
                consecutive_frames=3,
                cooldown_frames=20,
                zones=[
                    DangerZone(
                        "benchmark_zone",
                        [(320, 190), (610, 190), (628, 342), (300, 342)],
                    )
                ],
            ),
            vehicle_proximity=VehicleProximityRuleConfig(enabled=False),
        )
    )


def _run_mock_pipeline(
    frame: np.ndarray,
    detector: MockSafetyDetector,
    tracker: SimpleIoUTracker,
    safety_engine: SafetyRulesEngine,
) -> None:
    detections = detector.predict_frame(frame)
    tracks = tracker.update(detections)
    safety_engine.evaluate(tracks=tracks, detections=detections, frame_index=0)


def _write_benchmark_outputs(
    results: dict[str, Any],
    output_path: str | Path,
    markdown_report_path: str | Path | None,
) -> dict[str, Any]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if markdown_report_path is not None:
        _write_markdown_report(results, markdown_report_path)
    return results


def _write_markdown_report(results: dict[str, Any], report_path: str | Path) -> None:
    rows = []
    for backend in results["backends"]:
        rows.append(
            "| {backend} | {device} | {input_size} | {mean:.3f} ms | {p50:.3f} ms | "
            "{p95:.3f} ms | {fps:.2f} | {size} |".format(
                backend=backend["backend"],
                device=backend["device"],
                input_size=results["input_size"],
                mean=backend["mean_latency_ms"],
                p50=backend["p50_latency_ms"],
                p95=backend["p95_latency_ms"],
                fps=backend["fps"],
                size=backend["model_size_mb"] if backend["model_size_mb"] is not None else "n/a",
            )
        )
    mode_note = (
        "This is a mock pipeline benchmark, not neural network inference."
        if results["mode"] == "mock_pipeline"
        else "This benchmark used a real model checkpoint."
    )
    real_model = next(
        (backend.get("model", "models/best.pt") for backend in results["backends"]),
        "models/best.pt",
    )
    content = f"""# Benchmark Report

Generated: `{results["timestamp"]}`

Mode: `{results["mode"]}`

{mode_note}

Hardware:

- Platform: `{results["hardware"]["platform"]}`
- Processor: `{results["hardware"]["processor"]}`
- Python: `{results["hardware"]["python"]}`

| Backend | Device | Input size | Mean latency | P50 latency | P95 latency | FPS | Model size MB |
| --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

Benchmark configuration:

- Warmup runs: `{results["warmup_runs"]}`
- Benchmark runs: `{results["benchmark_runs"]}`

Real model benchmark command:

```bash
python scripts/run_benchmark.py --model {real_model} --image docs/assets/demo_input.jpg
```

Mock pipeline benchmark command:

```bash
python scripts/run_benchmark.py --mock --image docs/assets/demo_input.jpg
```
"""
    Path(report_path).write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch and optional ONNX inference.")
    parser.add_argument("--model", default=None, help="PyTorch YOLO checkpoint path.")
    parser.add_argument("--onnx-model", default=None, help="Optional ONNX model path.")
    parser.add_argument("--image", default=None, help="Optional benchmark image path.")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--benchmark-runs", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="reports/benchmark_results.json")
    parser.add_argument(
        "--markdown-report",
        default="docs/benchmark_report.md",
        help="Markdown benchmark report path. Use an empty value to skip writing it.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Benchmark deterministic mock pipeline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_benchmark(
        args.model,
        onnx_model_path=args.onnx_model,
        image_path=args.image,
        input_size=args.input_size,
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs,
        device=args.device,
        output_path=args.output,
        markdown_report_path=args.markdown_report or None,
        mock=args.mock,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
