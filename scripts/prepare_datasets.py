#!/usr/bin/env python3
from __future__ import annotations

import argparse
import imghdr
import os
import struct
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
CLASS_NAMES = ["person", "vehicle", "drone", "bicycle", "aerial_other"]
# VisDrone DET category IDs:
# 1:pedestrian 2:people 3:bicycle 4:car 5:van 6:truck 7:tricycle 8:awning-tricycle 9:bus 10:motor 11:others
VISDRONE_CLASS_MAP = {
    1: 0,
    2: 0,
    3: 3,
    4: 1,
    5: 1,
    6: 1,
    7: 1,
    8: 1,
    9: 1,
    10: 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset from VisDrone DET.")
    parser.add_argument("--aod4", required=True, help="Path to AOD-4 dataset directory")
    parser.add_argument("--hituav", required=True, help="Path to HIT-UAV dataset directory")
    parser.add_argument(
        "--visdrone", required=True, help="Path to VisDrone dataset directory"
    )
    parser.add_argument("--out", required=True, help="Output directory for generated files")
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of symlinking (slower, more disk usage)",
    )
    parser.add_argument(
        "--dataset-yaml",
        default="configs/dataset.yaml",
        help="Where to write dataset YAML",
    )
    return parser.parse_args()


def image_size(path: Path) -> tuple[int, int]:
    kind = imghdr.what(path)
    if kind == "png":
        with path.open("rb") as f:
            f.read(16)
            width, height = struct.unpack(">II", f.read(8))
            return int(width), int(height)
    if kind == "jpeg":
        with path.open("rb") as f:
            f.read(2)
            while True:
                marker_start = f.read(1)
                if not marker_start:
                    break
                if marker_start != b"\xFF":
                    continue
                marker = f.read(1)
                while marker == b"\xFF":
                    marker = f.read(1)
                if marker in {b"\xC0", b"\xC1", b"\xC2", b"\xC3", b"\xC5", b"\xC6", b"\xC7", b"\xC9", b"\xCA", b"\xCB", b"\xCD", b"\xCE", b"\xCF"}:
                    f.read(3)
                    h, w = struct.unpack(">HH", f.read(4))
                    return int(w), int(h)
                size = struct.unpack(">H", f.read(2))[0]
                f.seek(size - 2, os.SEEK_CUR)
    raise RuntimeError(f"Unsupported image format for size parsing: {path}")


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def convert_visdrone_annotation(txt_path: Path, img_w: int, img_h: int) -> list[str]:
    yolo_lines: list[str] = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 6:
            continue
        x, y, w, h, score, cat = parts[:6]
        cat_id = int(cat)
        if cat_id not in VISDRONE_CLASS_MAP:
            continue
        if int(score) == 0:
            continue
        cls = VISDRONE_CLASS_MAP[cat_id]
        bx = float(x)
        by = float(y)
        bw = float(w)
        bh = float(h)
        cx = (bx + bw / 2.0) / img_w
        cy = (by + bh / 2.0) / img_h
        nw = bw / img_w
        nh = bh / img_h
        if nw <= 0 or nh <= 0:
            continue
        yolo_lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return yolo_lines


def process_split(split_root: Path, out_dir: Path, copy_images: bool) -> tuple[int, int]:
    img_dir = split_root / "images"
    ann_dir = split_root / "annotations"
    if not img_dir.exists() or not ann_dir.exists():
        return 0, 0

    out_img = out_dir / "images"
    out_lbl = out_dir / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    image_count = 0
    label_count = 0
    for img_path in sorted(img_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        stem = img_path.stem
        ann_path = ann_dir / f"{stem}.txt"
        if not ann_path.exists():
            continue

        dest_img = out_img / img_path.name
        if not dest_img.exists():
            if copy_images:
                dest_img.write_bytes(img_path.read_bytes())
            else:
                dest_img.symlink_to(img_path.resolve())

        w, h = image_size(img_path)
        yolo_lines = convert_visdrone_annotation(ann_path, w, h)
        (out_lbl / f"{stem}.txt").write_text(
            "\n".join(yolo_lines) + ("\n" if yolo_lines else ""),
            encoding="utf-8",
        )
        image_count += 1
        if yolo_lines:
            label_count += 1
    return image_count, label_count


def main() -> int:
    args = parse_args()
    visdrone_root = Path(args.visdrone)
    if not visdrone_root.exists() or not visdrone_root.is_dir():
        raise FileNotFoundError(f"VisDrone directory not found: {visdrone_root}")
    out_dir = Path(args.out)

    train_root = visdrone_root / "VisDrone2019-DET-train" / "VisDrone2019-DET-train"
    val_root = visdrone_root / "VisDrone2019-DET-val" / "VisDrone2019-DET-val"
    if not train_root.exists() or not val_root.exists():
        raise FileNotFoundError(
            "Expected VisDrone train/val directories were not found under --visdrone"
        )

    train_images, train_labeled = process_split(
        train_root, out_dir / "train", args.copy_images
    )
    val_images, val_labeled = process_split(val_root, out_dir / "val", args.copy_images)
    if train_images == 0 or val_images == 0:
        raise RuntimeError("No images processed from VisDrone train/val splits.")

    dataset_yaml_path = Path(args.dataset_yaml)
    dataset_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_yaml_path.write_text(
        "\n".join(
            [
                f"path: {out_dir}",
                "train: train/images",
                "val: val/images",
                "test: val/images",
                f"nc: {len(CLASS_NAMES)}",
                "names:",
                *[f"  - {name}" for name in CLASS_NAMES],
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"[ok] Wrote dataset YAML: {dataset_yaml_path}")
    print(
        f"[ok] Prepared VisDrone YOLO dataset at: {out_dir} "
        f"(train images={train_images}, labeled={train_labeled}; "
        f"val images={val_images}, labeled={val_labeled})"
    )
    print("[note] AOD-4 and HIT-UAV conversion is not implemented yet in this script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
