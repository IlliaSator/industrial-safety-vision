PYTHON ?= python
MOCK ?= true

export INDUSTRIAL_SAFETY_MOCK_DETECTOR ?= $(MOCK)

.PHONY: install test lint format train evaluate validate-data generate-demo demo-image demo-video api docker-build docker-run export-onnx benchmark

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

lint:
	ruff check .

format:
	ruff check . --fix

train:
	$(PYTHON) -m industrial_safety_vision.training.train_yolo --config configs/train.yaml

evaluate:
	$(PYTHON) -m industrial_safety_vision.training.evaluate_yolo --config configs/train.yaml

validate-data:
	$(PYTHON) -m industrial_safety_vision.data.dataset_validation --config configs/train.yaml

generate-demo:
	$(PYTHON) scripts/generate_demo_assets.py

demo-image: generate-demo
	$(PYTHON) scripts/run_image_demo.py --image docs/assets/demo_input.jpg --output data/outputs --mock

demo-video: generate-demo
	$(PYTHON) scripts/run_video_demo.py --synthetic --output data/outputs/synthetic_annotated.gif --max-frames 30

api:
	$(PYTHON) -m uvicorn industrial_safety_vision.api.main:app --host 0.0.0.0 --port 8000

docker-build:
	docker build -t industrial-safety-vision:latest .

docker-run:
	docker run --rm -p 8000:8000 -e INDUSTRIAL_SAFETY_MOCK_DETECTOR=true -v ./models:/app/models -v ./data:/app/data industrial-safety-vision:latest

export-onnx:
	$(PYTHON) scripts/export_onnx.py --model models/best.pt --output-dir models

benchmark:
	$(PYTHON) scripts/run_benchmark.py --mock --output reports/benchmark_results.json
