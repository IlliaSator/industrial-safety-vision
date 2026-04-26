"""Video inference pipeline with latency/FPS accounting."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from industrial_safety_vision.core import Detection
from industrial_safety_vision.inference.detector import YOLODetector
from industrial_safety_vision.utils.video_io import build_video_writer, open_video_capture
from industrial_safety_vision.visualization.draw import draw_detections
from industrial_safety_vision.visualization.report import write_json


class FrameDetector(Protocol):
    def predict_frame(self, frame: np.ndarray) -> list[Detection]:
        """Return detections for a decoded frame."""


@dataclass
class VideoInferenceSummary:
    processed_frames: int
    average_latency_ms: float
    fps: float
    output_path: str | None = None
    alerts_path: str | None = None
    summary_path: str | None = None
    total_alerts: int = 0
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "processed_frames": self.processed_frames,
            "average_latency_ms": self.average_latency_ms,
            "fps": self.fps,
            "output_path": self.output_path,
            "alerts_path": self.alerts_path,
            "summary_path": self.summary_path,
            "total_alerts": self.total_alerts,
            "metadata": self.metadata,
        }


def process_frame_sequence(
    frames: list[np.ndarray],
    detector: FrameDetector,
) -> tuple[list[np.ndarray], VideoInferenceSummary]:
    """Process an in-memory frame sequence, useful for fast unit tests."""

    annotated_frames: list[np.ndarray] = []
    latencies_ms: list[float] = []
    started = time.perf_counter()
    for frame in frames:
        frame_started = time.perf_counter()
        detections = detector.predict_frame(frame)
        annotated_frames.append(draw_detections(frame, detections))
        latencies_ms.append((time.perf_counter() - frame_started) * 1000.0)

    elapsed = max(time.perf_counter() - started, 1e-12)
    processed = len(frames)
    summary = VideoInferenceSummary(
        processed_frames=processed,
        average_latency_ms=sum(latencies_ms) / processed if processed else 0.0,
        fps=processed / elapsed if processed else 0.0,
    )
    return annotated_frames, summary


def run_video_inference(
    input_source: str | int,
    output_path: str | Path,
    *,
    model_path: str | Path = "models/best.pt",
    confidence: float = 0.35,
    iou: float = 0.45,
    device: str = "auto",
    frame_skip: int = 1,
    max_frames: int | None = None,
) -> VideoInferenceSummary:
    """Run detection frame by frame on a video file or webcam index."""

    detector = YOLODetector(model_path, confidence=confidence, iou=iou, device=device)
    capture = open_video_capture(input_source)
    import cv2

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = build_video_writer(output_path, fps=source_fps, frame_size=(width, height))

    frame_skip = max(1, frame_skip)
    processed = 0
    frame_index = 0
    latencies_ms: list[float] = []
    started = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_skip != 0:
                frame_index += 1
                continue

            frame_started = time.perf_counter()
            detections = detector.predict_frame(frame)
            annotated = draw_detections(frame, detections)
            latencies_ms.append((time.perf_counter() - frame_started) * 1000.0)
            writer.write(annotated)
            processed += 1
            frame_index += 1
            if max_frames is not None and processed >= max_frames:
                break
    finally:
        capture.release()
        writer.release()

    elapsed = max(time.perf_counter() - started, 1e-12)
    summary_path = Path("data/outputs/video_summary.json")
    summary = VideoInferenceSummary(
        processed_frames=processed,
        average_latency_ms=sum(latencies_ms) / processed if processed else 0.0,
        fps=processed / elapsed if processed else 0.0,
        output_path=str(output_path),
        summary_path=str(summary_path),
        metadata={"frame_skip": frame_skip, "source_fps": source_fps},
    )
    write_json(summary_path, summary.to_dict())
    return summary
