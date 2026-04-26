from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()


def export_onnx(
    model_path: str | Path,
    output_path: str | Path,
    *,
    image_size: int = 640,
    opset: int | None = None,
) -> Path:
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
    export_kwargs = {"format": "onnx", "imgsz": image_size}
    if opset is not None:
        export_kwargs["opset"] = opset
    exported_path = Path(model.export(**export_kwargs))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if exported_path.resolve() != output_path.resolve():
        exported_path.replace(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a YOLO checkpoint to ONNX.")
    parser.add_argument("--model", required=True, help="Input YOLO checkpoint path.")
    parser.add_argument("--output", default=None, help="Output ONNX path.")
    parser.add_argument("--output-dir", default="models", help="Directory for exported ONNX file.")
    parser.add_argument("--imgsz", "--image-size", dest="image_size", type=int, default=640)
    parser.add_argument("--opset", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or Path(args.output_dir) / f"{Path(args.model).stem}.onnx"
    try:
        exported = export_onnx(args.model, output, image_size=args.image_size, opset=args.opset)
    except Exception as exc:
        raise SystemExit(f"ONNX export skipped: {exc}") from exc
    print(f"ONNX model exported to {exported}")


if __name__ == "__main__":
    main()
