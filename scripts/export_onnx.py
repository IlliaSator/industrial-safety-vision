from __future__ import annotations

import argparse
from pathlib import Path


def export_onnx(model_path: str | Path, output_path: str | Path, *, image_size: int = 640) -> Path:
    model_path = Path(model_path)
    output_path = Path(output_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}. ONNX export requires a real trained model."
        )
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is required for ONNX export.") from exc

    model = YOLO(str(model_path))
    exported_path = Path(model.export(format="onnx", imgsz=image_size))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if exported_path.resolve() != output_path.resolve():
        exported_path.replace(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a YOLO checkpoint to ONNX.")
    parser.add_argument("--model", required=True, help="Input YOLO checkpoint path.")
    parser.add_argument("--output", required=True, help="Output ONNX path.")
    parser.add_argument("--image-size", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = export_onnx(args.model, args.output, image_size=args.image_size)
    print(f"ONNX model exported to {output}")


if __name__ == "__main__":
    main()
