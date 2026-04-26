"""Video input/output helpers built on OpenCV."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def open_video_capture(input_source: str | int):
    cv2 = _require_cv2()
    capture = cv2.VideoCapture(input_source)
    if not capture.isOpened():
        msg = f"Failed to open video source: {input_source}"
        raise ValueError(msg)
    return capture


def build_video_writer(
    output_path: str | Path,
    *,
    fps: float,
    frame_size: tuple[int, int],
    codec: str = "mp4v",
):
    cv2 = _require_cv2()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
    if not writer.isOpened():
        msg = f"Failed to create video writer: {output_path}"
        raise ValueError(msg)
    return writer


def read_frames(
    capture,
    *,
    frame_skip: int = 1,
    max_frames: int | None = None,
) -> list[np.ndarray]:
    """Read frames from a capture for small tests or bounded demos."""

    frames: list[np.ndarray] = []
    frame_index = 0
    frame_skip = max(1, frame_skip)
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % frame_skip == 0:
            frames.append(frame)
            if max_frames is not None and len(frames) >= max_frames:
                break
        frame_index += 1
    return frames


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment-specific
        msg = "OpenCV is required for video I/O. Install opencv-python to use video demos."
        raise RuntimeError(msg) from exc
    return cv2
