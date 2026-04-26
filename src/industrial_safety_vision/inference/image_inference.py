"""Image inference pipeline."""

from __future__ import annotations

from pathlib import Path

from industrial_safety_vision.inference.detector import YOLODetector
from industrial_safety_vision.inference.mock_detector import MockSafetyDetector
from industrial_safety_vision.utils.image_io import read_image, write_image
from industrial_safety_vision.visualization.draw import draw_detections


def run_image_inference(
    image_path: str | Path,
    output_dir: str | Path,
    *,
    model_path: str | Path = "models/best.pt",
    confidence: float = 0.35,
    iou: float = 0.45,
    device: str = "auto",
    mock: bool = False,
) -> Path:
    """Run detector on one image and save an annotated copy."""

    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = read_image(image_path)

    detector = (
        MockSafetyDetector()
        if mock
        else YOLODetector(model_path, confidence=confidence, iou=iou, device=device)
    )
    detections = detector.predict_frame(frame)
    annotated = draw_detections(frame, detections)

    output_path = output_dir / f"{image_path.stem}_annotated{image_path.suffix}"
    write_image(output_path, annotated)
    return output_path
