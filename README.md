# Industrial Safety Vision

[![CI](https://github.com/IlliaSator/industrial-safety-vision/actions/workflows/ci.yml/badge.svg)](https://github.com/IlliaSator/industrial-safety-vision/actions/workflows/ci.yml)

Industrial Safety Vision is a production-style computer vision project for real-time industrial workplace safety monitoring. It combines YOLO-style object detection, worker tracking, a safety-rule engine, temporal alert smoothing, FastAPI serving, Docker packaging, dataset validation, training/evaluation scripts, ONNX export hooks, and benchmark tooling.

This repository currently supports two explicit modes:

1. **Demo/mock mode**: validates the full application pipeline without model weights.
2. **Real model mode**: runs YOLO inference with a pretrained COCO model or a custom PPE checkpoint.

Full PPE safety mode requires a custom model trained on classes such as `person`, `helmet`, `safety_vest`, `forklift`, and `vehicle`. No trained PPE checkpoint or fake metrics are committed.

## Status

| Component | Status |
| --- | --- |
| Package compile check | Passing |
| Unit tests | Passing |
| FastAPI health check | Working in mock mode |
| Image demo | Working in deterministic mock mode |
| Synthetic video demo | Working in deterministic mock mode |
| Real YOLO inference | Supported when a checkpoint/model name is provided |
| Custom PPE training | Smoke-tested on RF100 construction-safety dataset |
| ONNX export | Supported when a model checkpoint exists |
| Benchmarking | Working; mock pipeline benchmark included |
| Docker | Configured for mock API mode by default |

## Demo

Synthetic input:

![Synthetic input](docs/assets/demo_input.jpg)

Deterministic annotated output:

![Annotated output](docs/assets/demo_input_annotated.jpg)

Expected safety overlay:

![Expected overlay](docs/assets/demo_expected_overlay.jpg)

Example alert JSON:

```json
[
  {
    "alert_type": "danger_zone_violation",
    "severity": "critical",
    "track_id": 1,
    "frame_index": 26,
    "message": "Worker #1 entered danger zone 'default_loading_zone'.",
    "metadata": {
      "zone": "default_loading_zone",
      "mode": "synthetic_demo"
    }
  }
]
```

Generate demo assets:

```bash
python scripts/generate_demo_assets.py
```

Run image demo without model weights:

```bash
python scripts/run_image_demo.py --image docs/assets/demo_input.jpg --output data/outputs --mock
```

Run synthetic video pipeline demo without model weights:

```bash
python scripts/run_video_demo.py --synthetic --output data/outputs/synthetic_annotated.gif --max-frames 30
```

Synthetic mode demonstrates system behavior: drawing, tracking, safety rules, alert smoothing, and output serialization. It is not a neural-network accuracy demo.

## Architecture

```mermaid
flowchart LR
    A[Image / Video / Webcam] --> B[Detector: YOLO or Mock]
    B --> C[SimpleIoU Tracker]
    C --> D[Safety Rules Engine]
    B --> D
    D --> E[Temporal Smoothing + Cooldown]
    E --> F[Structured Alerts JSON/CSV]
    B --> G[Annotated Image/Video]
    C --> G
    D --> G
    F --> H[FastAPI / Reports]
```

## What Makes This More Than A YOLO Demo

The project focuses on the ML engineering layer around object detection:

- structured domain objects instead of raw model outputs
- swappable detector interface
- real-time video loop with FPS/latency accounting
- worker tracking
- PPE association logic
- danger-zone and vehicle-proximity rules
- temporal smoothing and cooldown
- API service with metrics
- Docker deployment
- reproducible train/eval/data validation scripts
- ONNX export entrypoint
- benchmark scripts
- model card and error-analysis documentation

## Quickstart

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m compileall src tests scripts
python -m pytest -q
ruff check .
```

If GNU Make is available:

```bash
make test
make lint
make download-data
make validate-data
make demo-image
make demo-video
make benchmark
```

## FastAPI

Start in mock mode:

```bash
$env:INDUSTRIAL_SAFETY_MOCK_DETECTOR="true"
python -m uvicorn industrial_safety_vision.api.main:app --host 0.0.0.0 --port 8000
```

Linux/macOS:

```bash
INDUSTRIAL_SAFETY_MOCK_DETECTOR=true python -m uvicorn industrial_safety_vision.api.main:app --host 0.0.0.0 --port 8000
```

Smoke checks:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/model/info
curl http://localhost:8000/metrics
curl -F "file=@docs/assets/demo_input.jpg" http://localhost:8000/predict/image
```

Swagger UI:

```text
http://localhost:8000/docs
```

## Real Model Inference

COCO smoke test:

```bash
python scripts/run_image_demo.py --image docs/assets/demo_input.jpg --model yolov8n.pt --output data/outputs
```

Custom PPE model:

```bash
python scripts/run_image_demo.py --image docs/assets/demo_input.jpg --model models/best.pt --output data/outputs
python scripts/run_video_demo.py --input data/samples/demo.mp4 --output data/outputs/annotated_demo.mp4 --model models/best.pt
```

`models/best.pt` is intentionally ignored by git. Put local checkpoints under `models/`.

## Dataset And Training

Recommended public starter dataset:

- Hugging Face: [`LibreYOLO/construction-safety-gsnvb`](https://huggingface.co/datasets/LibreYOLO/construction-safety-gsnvb)
- Original Roboflow Universe dataset: [`construction-safety-gsnvb`](https://universe.roboflow.com/roboflow-100/construction-safety-gsnvb/dataset/1)
- License: CC-BY-4.0
- Classes: `helmet`, `no-helmet`, `no-vest`, `person`, `vest`
- Splits validated locally: 997 train images, 119 validation images, 90 test images

Download locally:

```bash
python -m pip install huggingface_hub
python scripts/download_ppe_dataset.py --output data/raw/construction-safety-gsnvb
```

Expected YOLO dataset layout:

```text
data/processed/
  dataset.yaml
  images/train
  images/val
  images/test
  labels/train
  labels/val
  labels/test
```

The validator also supports the common Ultralytics/Roboflow layout with `train/images`, `train/labels`, `valid/images`, and `test/images`.

Validate dataset:

```bash
python -m industrial_safety_vision.data.dataset_validation \
  --dataset-yaml data/raw/construction-safety-gsnvb/data.yaml \
  --report reports/dataset_summary_construction_safety.json
```

Smoke train on the recommended dataset:

```bash
python -m industrial_safety_vision.training.train_yolo \
  --config configs/train.yaml \
  --dataset data/raw/construction-safety-gsnvb/data.yaml \
  --model yolo11n.pt \
  --epochs 1 \
  --batch-size 2 \
  --image-size 320 \
  --device 0
```

Evaluate:

```bash
python -m industrial_safety_vision.training.evaluate_yolo \
  --config configs/train.yaml \
  --dataset data/raw/construction-safety-gsnvb/data.yaml \
  --model runs/.../weights/best.pt \
  --split val \
  --device 0
```

Detection metrics such as mAP@0.5 and mAP@0.5:0.95 differ from classification accuracy because a prediction must get both the class and bounding-box localization right.

One local smoke-training run was completed on an NVIDIA MX450 with `yolo11n.pt`, 1 epoch, `imgsz=320`, `batch=2`. These numbers prove the training/evaluation pipeline runs; they are not production-quality model claims:

| Metric | Value |
| --- | ---: |
| Precision | 0.660 |
| Recall | 0.457 |
| mAP@0.5 | 0.448 |
| mAP@0.5:0.95 | 0.217 |

## Benchmark

The project now includes both a mock-pipeline benchmark and a real YOLO CPU benchmark.

| Backend | Device | Input size | Mean latency | P95 latency | FPS | Mode |
| --- | --- | --- | --- | --- | --- | --- |
| mock_detector_tracking_rules | CPU | 640 | 0.036 ms | 0.048 ms | 28011.21 | mock_pipeline |
| pytorch_ultralytics `yolo11n.pt` | CPU | 640 | 69.728 ms | 75.878 ms | 14.34 | real_model |

Reproduce:

```bash
python scripts/run_benchmark.py --mock --image docs/assets/demo_input.jpg --warmup-runs 3 --benchmark-runs 10
```

Real model benchmark:

```bash
python scripts/run_benchmark.py --model yolo11n.pt --image docs/assets/demo_input.jpg --device cpu
```

## ONNX Export

```bash
python scripts/export_onnx.py --model models/best.pt --output-dir models --imgsz 640
```

If the checkpoint is missing, the script exits with a clear message. ONNX files are ignored by git.

## Docker

```bash
docker build -t industrial-safety-vision:local .
docker compose up api
```

The compose service runs in mock mode by default so `/health`, `/model/info`, and `/metrics` work even without model weights.

For a heavier image with YOLO/OpenCV runtime dependencies:

```bash
docker build --build-arg INSTALL_ML_DEPS=true -t industrial-safety-vision:ml .
```

## Current Limitations

- No custom PPE model is included.
- Demo mode uses deterministic synthetic/mock detections.
- COCO pretrained models do not provide reliable helmet/vest classes.
- PPE association is based on bounding-box overlap and can fail with overlapping people.
- Danger zones require camera/site calibration.
- Safety-critical deployment requires human validation, monitoring, audit logs, and a reviewed escalation process.

## Roadmap

- Train/evaluate a custom PPE model on a documented dataset
- ByteTrack or DeepSORT integration
- RTSP stream support
- Perspective calibration for vehicle proximity
- ONNX/TensorRT optimized inference profile
- Model monitoring and active-learning loop
- Human-in-the-loop alert review
