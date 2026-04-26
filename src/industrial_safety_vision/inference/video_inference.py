"""Video inference pipeline with latency/FPS accounting."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from industrial_safety_vision.core import Detection
from industrial_safety_vision.inference.detector import YOLODetector
from industrial_safety_vision.safety.alert_types import Alert
from industrial_safety_vision.safety.rules import SafetyRulesEngine, load_safety_rules_config
from industrial_safety_vision.tracking.tracker import SimpleIoUTracker, load_tracker_from_config
from industrial_safety_vision.tracking.track_types import Track
from industrial_safety_vision.utils.video_io import build_video_writer, open_video_capture
from industrial_safety_vision.visualization.draw import (
    draw_alerts,
    draw_danger_zones,
    draw_detections,
    draw_tracks,
)
from industrial_safety_vision.visualization.report import write_json, write_rows_csv


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
    *,
    tracker: SimpleIoUTracker | None = None,
    safety_engine: SafetyRulesEngine | None = None,
) -> tuple[list[np.ndarray], VideoInferenceSummary]:
    """Process an in-memory frame sequence, useful for fast unit tests."""

    annotated_frames: list[np.ndarray] = []
    latencies_ms: list[float] = []
    all_alerts: list[Alert] = []
    started = time.perf_counter()
    for frame_index, frame in enumerate(frames):
        frame_started = time.perf_counter()
        detections = detector.predict_frame(frame)
        tracks: list[Track] = tracker.update(detections) if tracker is not None else []
        alerts = (
            safety_engine.evaluate(tracks=tracks, detections=detections, frame_index=frame_index)
            if safety_engine is not None
            else []
        )
        annotated = draw_tracks(frame, tracks) if tracks else draw_detections(frame, detections)
        annotated = draw_alerts(annotated, alerts)
        annotated_frames.append(annotated)
        all_alerts.extend(alerts)
        latencies_ms.append((time.perf_counter() - frame_started) * 1000.0)

    elapsed = max(time.perf_counter() - started, 1e-12)
    processed = len(frames)
    summary = VideoInferenceSummary(
        processed_frames=processed,
        average_latency_ms=sum(latencies_ms) / processed if processed else 0.0,
        fps=processed / elapsed if processed else 0.0,
        total_alerts=len(all_alerts),
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
    tracking_config_path: str | Path = "configs/tracking.yaml",
    safety_config_path: str | Path = "configs/safety_rules.yaml",
    enable_tracking: bool = True,
    enable_safety: bool = True,
) -> VideoInferenceSummary:
    """Run detection frame by frame on a video file or webcam index."""

    detector = YOLODetector(model_path, confidence=confidence, iou=iou, device=device)
    tracker = load_tracker_from_config(tracking_config_path) if enable_tracking else None
    safety_engine = SafetyRulesEngine(load_safety_rules_config(safety_config_path)) if enable_safety else None
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
    alerts: list[Alert] = []
    started = time.perf_counter()
    zones = (
        [(zone.name, zone.polygon) for zone in safety_engine.config.danger_zone.zones]
        if safety_engine is not None
        else []
    )

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
            tracks = tracker.update(detections) if tracker is not None else []
            frame_alerts = (
                safety_engine.evaluate(tracks=tracks, detections=detections, frame_index=frame_index)
                if safety_engine is not None
                else []
            )
            annotated = draw_tracks(frame, tracks) if tracks else draw_detections(frame, detections)
            if zones:
                annotated = draw_danger_zones(annotated, zones)
            annotated = draw_alerts(annotated, frame_alerts)
            alerts.extend(frame_alerts)
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
    alerts_path = Path("data/outputs/video_alerts.json")
    alerts_csv_path = Path("data/outputs/video_alerts.csv")
    alert_rows = [alert.to_dict() for alert in alerts]
    write_json(alerts_path, alert_rows)
    write_rows_csv(alerts_csv_path, alert_rows)
    summary = VideoInferenceSummary(
        processed_frames=processed,
        average_latency_ms=sum(latencies_ms) / processed if processed else 0.0,
        fps=processed / elapsed if processed else 0.0,
        output_path=str(output_path),
        alerts_path=str(alerts_path),
        summary_path=str(summary_path),
        total_alerts=len(alerts),
        metadata={"frame_skip": frame_skip, "source_fps": source_fps},
    )
    write_json(summary_path, summary.to_dict())
    return summary
