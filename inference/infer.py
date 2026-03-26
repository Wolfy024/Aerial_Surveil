#!/usr/bin/env python3
"""Run YOLO inference on a video and save annotated output."""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO detection on video.")
    parser.add_argument("--weights", required=True, help="Path to best.pt weights")
    parser.add_argument("--source", required=True, help="Path to input video file")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size (default: 1280, matching your training)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--device", default=None, help="cuda device or cpu")
    parser.add_argument("--project", default="runs/infer", help="Output directory")
    parser.add_argument("--name", default="exp", help="Experiment name")
    parser.add_argument("--line-width", type=int, default=2, help="Bounding box line width")
    parser.add_argument("--show-labels", action="store_true", default=True)
    parser.add_argument("--show-conf", action="store_true", default=True)
    parser.add_argument("--save-txt", action="store_true", help="Also save detections as .txt labels")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    weights = Path(args.weights)
    source = Path(args.source)

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not source.exists():
        raise FileNotFoundError(f"Video not found: {source}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics not installed") from exc

    model = YOLO(str(weights))

    results = model.predict(
        source=str(source),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=args.project,
        name=args.name,
        save=True,           # saves the annotated video
        save_txt=args.save_txt,
        line_width=args.line_width,
        show_labels=args.show_labels,
        show_conf=args.show_conf,
        stream=True,         # memory-efficient frame-by-frame processing
    )

    # Consume the generator to process all frames
    frame_count = 0
    for r in results:
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"  Processed {frame_count} frames...")

    print(f"\n[ok] Inference complete — {frame_count} frames processed.")
    print(f"[ok] Annotated video saved to: {args.project}/{args.name}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())