"""Image I/O helpers with OpenCV primary path and Pillow fallback."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


def read_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    try:
        import cv2

        image = cv2.imread(str(path))
        if image is not None:
            return image
    except ImportError:
        pass

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def write_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2

        if cv2.imwrite(str(path), image):
            return
    except ImportError:
        pass

    Image.fromarray(_as_uint8(image)).save(path)


def decode_image_bytes(content: bytes) -> np.ndarray:
    try:
        import cv2

        image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            return image
    except ImportError:
        pass

    with Image.open(BytesIO(content)) as image:
        return np.asarray(image.convert("RGB"))


def _as_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)
