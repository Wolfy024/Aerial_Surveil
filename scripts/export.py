#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO model to deployment formats.")
    parser.add_argument("--weights", required=True, help="Weights path")
    parser.add_argument("--format", default="onnx", help="Export format, e.g. onnx/engine")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--device", default=None, help="cuda device string or cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed. Run: pip install -r requirements.txt"
        ) from exc

    model = YOLO(str(weights_path))
    kwargs = {
        "format": args.format,
        "imgsz": args.imgsz,
        "dynamic": args.dynamic,
        "half": args.half,
    }
    if args.device is not None:
        kwargs["device"] = args.device
    model.export(**kwargs)
    print("[ok] Export finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
