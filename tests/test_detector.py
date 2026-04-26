import numpy as np
import pytest

from industrial_safety_vision.inference.detector import YOLODetector


class FakeBoxes:
    xyxy = np.array([[0, 1, 20, 31], [50, 60, 90, 140]], dtype=float)
    cls = np.array([0, 2], dtype=float)
    conf = np.array([0.91, 0.82], dtype=float)


class FakeResult:
    boxes = FakeBoxes()


class FakeYOLO:
    names = {0: "person", 2: "helmet"}

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def predict(self, **kwargs: object) -> list[FakeResult]:
        self.calls.append(kwargs)
        return [FakeResult()]


def test_detector_returns_structured_detections() -> None:
    fake_model = FakeYOLO()
    detector = YOLODetector(
        "fake.pt",
        confidence=0.4,
        iou=0.5,
        device="cpu",
        model_factory=lambda _: fake_model,
    )

    detections = detector.predict_frame(np.zeros((128, 128, 3), dtype=np.uint8))

    assert len(detections) == 2
    assert detections[0].class_name == "person"
    assert detections[0].bbox.width == 20
    assert detections[1].class_name == "helmet"
    assert fake_model.calls[0]["conf"] == 0.4
    assert fake_model.calls[0]["iou"] == 0.5


def test_detector_rejects_empty_frame() -> None:
    detector = YOLODetector("fake.pt", model_factory=lambda _: FakeYOLO())

    with pytest.raises(ValueError, match="empty frame"):
        detector.predict_frame(np.array([]))
