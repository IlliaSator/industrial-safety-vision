from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run video safety inference demo.")
    parser.add_argument("--input", default=None, help="Path to video file or webcam index.")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run deterministic synthetic pipeline demo without YOLO weights.",
    )
    parser.add_argument(
        "--output",
        default="data/outputs/annotated_demo.mp4",
        help="Output video path.",
    )
    parser.add_argument("--model", default="models/best.pt", help="YOLO model path or model name.")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--device", default="auto", help="cpu, cuda, cuda:0, or auto.")
    parser.add_argument("--frame-skip", type=int, default=1, help="Process every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum processed frames.")
    return parser.parse_args()


def main() -> None:
    from industrial_safety_vision.inference.video_inference import run_video_inference

    args = parse_args()
    if args.synthetic:
        summary = run_synthetic_demo(args.output, max_frames=args.max_frames or 30)
        print(json.dumps(summary, indent=2))
        return

    if args.input is None:
        raise SystemExit("--input is required unless --synthetic is used.")
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


def run_synthetic_demo(output_path: str, *, max_frames: int) -> dict[str, object]:
    from industrial_safety_vision.demo.synthetic import generate_synthetic_frames, save_gif
    from industrial_safety_vision.inference.mock_detector import MockSafetyDetector
    from industrial_safety_vision.safety.alert_types import Alert
    from industrial_safety_vision.safety.danger_zone import DangerZone
    from industrial_safety_vision.safety.rules import (
        DangerZoneRuleConfig,
        PPERuleConfig,
        SafetyRulesConfig,
        SafetyRulesEngine,
        VehicleProximityRuleConfig,
    )
    from industrial_safety_vision.tracking.tracker import SimpleIoUTracker
    from industrial_safety_vision.visualization.draw import (
        draw_alerts,
        draw_danger_zones,
        draw_tracks,
    )
    from industrial_safety_vision.visualization.report import write_json, write_rows_csv

    frames = generate_synthetic_frames(max_frames)
    detector = MockSafetyDetector(include_helmet=False, include_vest=True, moving_person=True)
    tracker = SimpleIoUTracker(iou_threshold=0.1, max_missed_frames=3)
    safety_engine = SafetyRulesEngine(
        SafetyRulesConfig(
            missing_helmet=PPERuleConfig(consecutive_frames=3, cooldown_frames=20),
            missing_vest=PPERuleConfig(enabled=False),
            danger_zone=DangerZoneRuleConfig(
                consecutive_frames=3,
                cooldown_frames=20,
                zones=[
                    DangerZone(
                        "synthetic_loading_zone",
                        [(320, 190), (610, 190), (628, 342), (300, 342)],
                    )
                ],
            ),
            vehicle_proximity=VehicleProximityRuleConfig(enabled=False),
        )
    )

    annotated_frames = []
    alerts: list[Alert] = []
    latencies_ms: list[float] = []
    zones = [(zone.name, zone.polygon) for zone in safety_engine.config.danger_zone.zones]
    started = time.perf_counter()

    for frame_index, frame in enumerate(frames):
        frame_started = time.perf_counter()
        detections = detector.predict_frame(frame)
        tracks = tracker.update(detections)
        frame_alerts = safety_engine.evaluate(
            tracks=tracks,
            detections=detections,
            frame_index=frame_index,
        )
        annotated = draw_tracks(frame, tracks)
        annotated = draw_danger_zones(annotated, zones)
        annotated = draw_alerts(annotated, frame_alerts)
        annotated_frames.append(annotated)
        alerts.extend(frame_alerts)
        latencies_ms.append((time.perf_counter() - frame_started) * 1000.0)

    output = Path(output_path)
    save_gif(output, annotated_frames)
    alerts_path = Path("data/outputs/synthetic_video_alerts.json")
    alerts_csv_path = Path("data/outputs/synthetic_video_alerts.csv")
    summary_path = Path("data/outputs/synthetic_video_summary.json")
    alert_rows = [alert.to_dict() for alert in alerts]
    write_json(alerts_path, alert_rows)
    write_rows_csv(alerts_csv_path, alert_rows)

    elapsed = max(time.perf_counter() - started, 1e-12)
    summary = {
        "mode": "mock_pipeline",
        "processed_frames": len(frames),
        "average_latency_ms": sum(latencies_ms) / len(latencies_ms),
        "fps": len(frames) / elapsed,
        "output_path": str(output),
        "alerts_path": str(alerts_path),
        "summary_path": str(summary_path),
        "total_alerts": len(alerts),
    }
    write_json(summary_path, summary)
    return summary


if __name__ == "__main__":
    main()
