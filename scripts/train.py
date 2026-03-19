#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO detection model.")
    parser.add_argument("--data", required=True, help="Dataset yaml path")
    parser.add_argument(
        "--model",
        default="yolo26m.pt",
        help="Model name or weights path (default: yolo26m.pt)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="exp")
    parser.add_argument("--device", default=None, help="cuda device string or cpu")
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--freeze", type=int, default=0)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--augment", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed. Run: pip install -r requirements.txt"
        ) from exc

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "freeze": args.freeze,
        "patience": args.patience,
        "augment": args.augment,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device
    if args.lr0 is not None:
        train_kwargs["lr0"] = args.lr0

    model.train(**train_kwargs)
    print("[ok] Training finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
