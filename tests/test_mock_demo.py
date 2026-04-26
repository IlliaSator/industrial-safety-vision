import numpy as np

from industrial_safety_vision.inference.image_inference import run_image_inference
from industrial_safety_vision.inference.mock_detector import MockSafetyDetector
from industrial_safety_vision.utils.image_io import write_image


def test_mock_detector_returns_safety_classes() -> None:
    detector = MockSafetyDetector()

    detections = detector.predict_frame(np.zeros((240, 320, 3), dtype=np.uint8))

    assert {detection.class_name for detection in detections} == {
        "person",
        "helmet",
        "safety_vest",
        "forklift",
    }
    assert detector.model_info()["backend"] == "mock_safety_detector"


def test_image_inference_mock_mode_writes_output(tmp_path) -> None:
    image_path = tmp_path / "input.jpg"
    output_dir = tmp_path / "outputs"
    write_image(image_path, np.zeros((120, 160, 3), dtype=np.uint8))

    output_path = run_image_inference(image_path, output_dir, mock=True)

    assert output_path.exists()
    assert output_path.name == "input_annotated.jpg"
