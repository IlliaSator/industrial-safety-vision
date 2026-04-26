from __future__ import annotations

import argparse

from industrial_safety_vision.inference.video_inference import run_video_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run video safety inference demo.")
    parser.add_argument("--input", required=True, help="Path to video file or webcam index.")
    parser.add_argument("--output", default="data/outputs/annotated_demo.mp4", help="Output video path.")
    parser.add_argument("--model", default="models/best.pt", help="YOLO model path or model name.")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--device", default="auto", help="cpu, cuda, cuda:0, or auto.")
    parser.add_argument("--frame-skip", type=int, default=1, help="Process every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum processed frames.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_source: str | int = int(args.input) if str(args.input).isdigit() else args.input
    summary = run_video_inference(
        input_source=input_source,
        output_path=args.output,
        model_path=args.model,
        confidence=args.conf,
        iou=args.iou,
        device=args.device,
        frame_skip=args.frame_skip,
        max_frames=args.max_frames,
    )
    print(summary.to_dict())


if __name__ == "__main__":
    main()
