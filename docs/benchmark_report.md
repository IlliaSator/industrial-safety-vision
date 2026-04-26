# Benchmark Report

Generated: `2026-04-26T15:12:30.771980+00:00`

Mode: `mock_pipeline`

This is a mock pipeline benchmark, not neural network inference.

Hardware:

- Platform: `Windows-11-10.0.26200-SP0`
- Processor: `AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD`
- Python: `3.12.8`

| Backend | Device | Input size | Mean latency | P50 latency | P95 latency | FPS | Model size MB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mock_detector_tracking_rules | cpu | 640 | 0.036 ms | 0.033 ms | 0.048 ms | 28011.21 | n/a |

Benchmark configuration:

- Warmup runs: `3`
- Benchmark runs: `10`

Real model benchmark command:

```bash
python scripts/run_benchmark.py --model models/best.pt --image docs/assets/demo_input.jpg
```

Mock pipeline benchmark command:

```bash
python scripts/run_benchmark.py --mock --image docs/assets/demo_input.jpg
```
