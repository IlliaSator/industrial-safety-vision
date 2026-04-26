.PHONY: install test lint train evaluate validate-data demo-image demo-video api docker-build docker-run export-onnx benchmark

install:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .

train:
	python -m industrial_safety_vision.training.train_yolo --config configs/train.yaml

evaluate:
	python -m industrial_safety_vision.training.evaluate_yolo --config configs/train.yaml

validate-data:
	python -m industrial_safety_vision.data.dataset_validation --config configs/train.yaml

demo-image:
	python scripts/run_image_demo.py --image data/samples/sample.jpg --output data/outputs

demo-video:
	python scripts/run_video_demo.py --input data/samples/demo.mp4 --output data/outputs/annotated_demo.mp4

api:
	uvicorn industrial_safety_vision.api.main:app --host 0.0.0.0 --port 8000

docker-build:
	docker build -t industrial-safety-vision:latest .

docker-run:
	docker run --rm -p 8000:8000 -v ./models:/app/models -v ./data:/app/data industrial-safety-vision:latest

export-onnx:
	python scripts/export_onnx.py --model models/best.pt --output models/best.onnx

benchmark:
	python scripts/run_benchmark.py --model models/best.pt
