import numpy as np

from industrial_safety_vision.core import BoundingBox, Detection
from industrial_safety_vision.inference.video_inference import process_frame_sequence


class FakeDetector:
    def predict_frame(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        return [Detection(0, "person", 0.9, BoundingBox(1, 1, w / 2, h / 2))]


def test_process_frame_sequence_with_synthetic_frames() -> None:
    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]

    annotated, summary = process_frame_sequence(frames, FakeDetector())

    assert len(annotated) == 3
    assert summary.processed_frames == 3
    assert summary.average_latency_ms >= 0
    assert summary.fps > 0
    assert annotated[0].sum() > 0
