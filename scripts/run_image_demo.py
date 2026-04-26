from __future__ import annotations

import argparse
from pathlib import Path

from industrial_safety_vision.inference.image_inference import run_image_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO image inference demo.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--output", default="data/outputs", help="Directory for annotated image.")
    parser.add_argument("--model", default="models/best.pt", help="YOLO model path or model name.")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--device", default="auto", help="cpu, cuda, cuda:0, or auto.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = run_image_inference(
        image_path=args.image,
        output_dir=args.output,
        model_path=args.model,
        confidence=args.conf,
        iou=args.iou,
        device=args.device,
    )
    print(f"Annotated image saved to {Path(output_path)}")


if __name__ == "__main__":
    main()
