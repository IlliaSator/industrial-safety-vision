"""Synthetic image/video fixtures for deterministic demo mode."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def generate_synthetic_frame(
    *,
    width: int = 640,
    height: int = 360,
    frame_index: int = 0,
    total_frames: int = 30,
    annotated: bool = False,
) -> np.ndarray:
    """Generate a simple industrial-safety scene as an RGB array."""

    image = Image.new("RGB", (width, height), (236, 239, 242))
    draw = ImageDraw.Draw(image)

    draw.rectangle((0, int(height * 0.72), width, height), fill=(195, 201, 205))
    zone = [
        (int(width * 0.52), int(height * 0.55)),
        (int(width * 0.95), int(height * 0.55)),
        (int(width * 0.98), int(height * 0.95)),
        (int(width * 0.48), int(height * 0.95)),
    ]
    draw.polygon(zone, fill=(235, 214, 214), outline=(190, 65, 65))

    progress = frame_index / max(total_frames - 1, 1)
    person_x = int(width * (0.16 + progress * 0.42))
    person_y = int(height * 0.34)
    person_w = int(width * 0.12)
    person_h = int(height * 0.45)
    person = (person_x, person_y, person_x + person_w, person_y + person_h)

    helmet = (
        person_x + int(person_w * 0.25),
        person_y - int(person_h * 0.12),
        person_x + int(person_w * 0.75),
        person_y + int(person_h * 0.05),
    )
    vest = (
        person_x + int(person_w * 0.18),
        person_y + int(person_h * 0.25),
        person_x + int(person_w * 0.82),
        person_y + int(person_h * 0.62),
    )
    vehicle = (
        int(width * 0.68),
        int(height * 0.47),
        int(width * 0.9),
        int(height * 0.73),
    )

    draw.rectangle(vehicle, fill=(222, 142, 55), outline=(120, 70, 30), width=3)
    draw.rectangle(person, fill=(70, 115, 170), outline=(25, 60, 105), width=3)
    draw.rectangle(helmet, fill=(245, 204, 52), outline=(110, 90, 20), width=2)
    draw.rectangle(vest, fill=(235, 235, 72), outline=(100, 100, 20), width=2)

    if annotated:
        draw.rectangle(person, outline=(54, 170, 90), width=4)
        draw.rectangle(helmet, outline=(255, 215, 0), width=4)
        draw.rectangle(vest, outline=(255, 190, 0), width=4)
        draw.rectangle(vehicle, outline=(230, 95, 40), width=4)
        draw.line(zone + [zone[0]], fill=(210, 40, 40), width=4)
        if progress > 0.7:
            draw.rectangle(person, outline=(220, 25, 25), width=6)

    return np.asarray(image)


def generate_synthetic_frames(frame_count: int = 30) -> list[np.ndarray]:
    return [
        generate_synthetic_frame(frame_index=index, total_frames=frame_count, annotated=False)
        for index in range(frame_count)
    ]


def save_gif(path: str | Path, frames: list[np.ndarray], *, duration_ms: int = 80) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = [Image.fromarray(frame.astype(np.uint8)) for frame in frames]
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )


def write_demo_assets(output_dir: str | Path = "docs/assets") -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = output_dir / "demo_input.jpg"
    overlay_path = output_dir / "demo_expected_overlay.jpg"
    alerts_path = output_dir / "alerts_example.json"

    Image.fromarray(generate_synthetic_frame()).save(input_path, quality=90)
    Image.fromarray(generate_synthetic_frame(frame_index=26, annotated=True)).save(
        overlay_path,
        quality=90,
    )
    alerts = [
        {
            "alert_type": "danger_zone_violation",
            "severity": "critical",
            "track_id": 1,
            "frame_index": 26,
            "message": "Worker #1 entered danger zone 'default_loading_zone'.",
            "metadata": {"zone": "default_loading_zone", "mode": "synthetic_demo"},
        }
    ]
    alerts_path.write_text(json.dumps(alerts, indent=2), encoding="utf-8")

    return {
        "input": str(input_path),
        "expected_overlay": str(overlay_path),
        "alerts_example": str(alerts_path),
    }
