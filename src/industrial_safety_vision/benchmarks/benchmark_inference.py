"""Inference latency benchmark utilities."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from industrial_safety_vision.inference.detector import YOLODetector
from industrial_safety_vision.inference.onnx_detector import ONNXDetector


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
    model_path: str | Path,
    *,
    onnx_model_path: str | Path | None = None,
    input_size: int = 640,
    warmup_runs: int = 5,
    benchmark_runs: int = 20,
    device: str = "cpu",
    output_path: str | Path = "reports/benchmark_results.json",
) -> dict[str, Any]:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"PyTorch model file not found: {model_path}")

    frame = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    results: dict[str, Any] = {
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

    detector = YOLODetector(model_path, device=device)
    pytorch_metrics = benchmark_callable(
        lambda: detector.predict_frame(frame),
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs,
    )
    results["backends"].append(
        {
            "backend": "pytorch_ultralytics",
            "device": device,
            "model_size_mb": _file_size_mb(model_path),
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

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, round((percentile / 100) * (len(sorted_values) - 1)))
    return sorted_values[index]


def _file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch and optional ONNX inference.")
    parser.add_argument("--model", required=True, help="PyTorch YOLO checkpoint path.")
    parser.add_argument("--onnx-model", default=None, help="Optional ONNX model path.")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--benchmark-runs", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="reports/benchmark_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_benchmark(
        args.model,
        onnx_model_path=args.onnx_model,
        input_size=args.input_size,
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs,
        device=args.device,
        output_path=args.output,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
