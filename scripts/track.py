#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO tracking on a source stream.")
    parser.add_argument("--source", required=True, help="Video path, image path, dir, or stream")
    parser.add_argument("--weights", required=True, help="Model weights path")
    parser.add_argument("--tracker", default="configs/bytetrack.yaml", help="Tracker config yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default=None, help="cuda device string or cpu")
    parser.add_argument("--save", action="store_true", help="Save rendered output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    tracker_path = Path(args.tracker)
    if not tracker_path.exists():
        raise FileNotFoundError(f"Tracker config not found: {tracker_path}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed. Run: pip install -r requirements.txt"
        ) from exc

    model = YOLO(str(weights_path))
    kwargs = {
        "source": args.source,
        "tracker": str(tracker_path),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "save": args.save,
    }
    if args.device is not None:
        kwargs["device"] = args.device
    model.track(**kwargs)
    print("[ok] Tracking run finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
