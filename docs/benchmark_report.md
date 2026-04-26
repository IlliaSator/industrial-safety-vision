# Benchmark Report

Benchmarks are generated from actual local runs only. No placeholder metrics should be interpreted as performance claims.

Run:

```bash
python scripts/run_benchmark.py --model models/best.pt --onnx-model models/best.onnx --input-size 640
```

Output:

- `reports/benchmark_results.json`

| Backend | Device | Input size | Mean latency | P50 latency | P95 latency | FPS | Model size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PyTorch / Ultralytics | Run locally | Run locally | Run locally | Run locally | Run locally | Run locally | Run locally |
| ONNX Runtime | Run locally | Run locally | Run locally | Run locally | Run locally | Run locally | Run locally |

Record hardware, model checkpoint, input size, warmup runs, benchmark runs, and whether the run used CPU, CUDA, or another accelerator.
