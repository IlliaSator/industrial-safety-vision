"""Image inference pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2

from industrial_safety_vision.inference.detector import YOLODetector
from industrial_safety_vision.visualization.draw import draw_detections


def run_image_inference(
    image_path: str | Path,
    output_dir: str | Path,
    *,
    model_path: str | Path = "models/best.pt",
    confidence: float = 0.35,
    iou: float = 0.45,
    device: str = "auto",
) -> Path:
    """Run detector on one image and save an annotated copy."""

    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = cv2.imread(str(image_path))
    if frame is None:
        msg = f"Failed to read image: {image_path}"
        raise ValueError(msg)

    detector = YOLODetector(model_path, confidence=confidence, iou=iou, device=device)
    detections = detector.predict_frame(frame)
    annotated = draw_detections(frame, detections)

    output_path = output_dir / f"{image_path.stem}_annotated{image_path.suffix}"
    cv2.imwrite(str(output_path), annotated)
    return output_path
