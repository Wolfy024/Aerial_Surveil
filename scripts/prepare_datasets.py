#!/usr/bin/env python3
"""
Prepare a unified YOLO dataset from AOD-4, HIT-UAV, and VisDrone-DET.

Unified class map (5 classes):
  0  person
  1  vehicle
  2  drone
  3  bicycle
  4  aerial_other

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AOD-4  (confirmed layout)
  <aod4_root>/
    Images/
      train/   *.jpg
      valid/   *.jpg
      test/    *.jpg
    Annotations/
      YOLOv8 format/
        train/   *.txt
        valid/   *.txt
        test/    *.txt

  AOD-4 YOLOv8 class IDs  →  unified
    0 drone        →  2  drone
    1 airplane     →  4  aerial_other
    2 helicopter   →  4  aerial_other
    3 bird         →  4  aerial_other

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HIT-UAV  (HIT-UAV-Infrared-Thermal-Dataset-main)
  <hituav_root>/
    yolo_labels/   *.txt  flat dir (class IDs: 0=person 1=car 2=bicycle
                                    3=OtherVehicle  4=DontCare[skip])
    <images_dir>/  *.jpg  searched as: images/ JPEGImages/ VideoFrames/ frames/

  No train/val split on disk — split 90/10 by sorted filename.

  HIT-UAV class IDs  →  unified
    0 person         →  0  person
    1 car            →  1  vehicle
    2 bicycle        →  3  bicycle
    3 OtherVehicle   →  1  vehicle
    4 DontCare       →  skipped

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VisDrone-DET
  <visdrone_root>/
    VisDrone2019-DET-{train,val}/VisDrone2019-DET-{train,val}/
      images/        *.jpg
      annotations/   *.txt   CSV: x,y,w,h,score,cat_id,trunc,occl
  Flat layout (no double-nested dir) also accepted.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

import argparse
import os
import shutil
import struct
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ── Unified class IDs ─────────────────────────────────────────────────────────
CLASS_NAMES   = ["person", "vehicle", "drone", "bicycle", "aerial_other"]
CLASS_PERSON  = 0
CLASS_VEHICLE = 1
CLASS_DRONE   = 2
CLASS_BICYCLE = 3
CLASS_OTHER   = 4

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# ── Per-dataset class maps ────────────────────────────────────────────────────

AOD4_CLASS_MAP: dict[int, int] = {
    0: CLASS_DRONE,    # drone
    1: CLASS_OTHER,    # airplane  → aerial_other
    2: CLASS_OTHER,    # helicopter → aerial_other
    3: CLASS_OTHER,    # bird      → aerial_other
}

HITUAV_CLASS_MAP: dict[int, int] = {
    0: CLASS_PERSON,   # person
    1: CLASS_VEHICLE,  # car
    2: CLASS_BICYCLE,  # bicycle
    3: CLASS_VEHICLE,  # OtherVehicle → vehicle
    # 4 DontCare → omitted = skipped
}

VISDRONE_CLASS_MAP: dict[int, int] = {
    1:  CLASS_PERSON,   # pedestrian
    2:  CLASS_PERSON,   # people
    3:  CLASS_BICYCLE,  # bicycle
    4:  CLASS_VEHICLE,  # car
    5:  CLASS_VEHICLE,  # van
    6:  CLASS_VEHICLE,  # truck
    7:  CLASS_VEHICLE,  # tricycle
    8:  CLASS_VEHICLE,  # awning-tricycle
    9:  CLASS_VEHICLE,  # bus
    10: CLASS_VEHICLE,  # motor
    # 11 others → uncomment to include: 11: CLASS_OTHER,
}

# ── Utilities ─────────────────────────────────────────────────────────────────

def image_size(path: Path) -> tuple[int, int]:
    """Return (width, height) by reading image header bytes — no PIL required."""
    with path.open("rb") as f:
        header = f.read(24)
    if header[:8] == b"\x89PNG\r\n\x1a\n":
        w, h = struct.unpack(">II", header[16:24])
        return int(w), int(h)
    if header[:2] == b"\xFF\xD8":
        with path.open("rb") as f:
            f.read(2)
            while True:
                b = f.read(1)
                if not b:
                    break
                if b != b"\xFF":
                    continue
                marker = f.read(1)
                while marker == b"\xFF":
                    marker = f.read(1)
                if marker in {
                    b"\xC0", b"\xC1", b"\xC2", b"\xC3",
                    b"\xC5", b"\xC6", b"\xC7",
                    b"\xC9", b"\xCA", b"\xCB",
                    b"\xCD", b"\xCE", b"\xCF",
                }:
                    f.read(3)
                    h, w = struct.unpack(">HH", f.read(4))
                    return int(w), int(h)
                seg = struct.unpack(">H", f.read(2))[0]
                f.seek(seg - 2, os.SEEK_CUR)
    raise RuntimeError(f"Unsupported image format: {path}")


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    if dst.exists():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def _make_split_dirs(split_out: Path) -> tuple[Path, Path]:
    img_dir = split_out / "images"
    lbl_dir = split_out / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    return img_dir, lbl_dir


def _run_parallel(
    tasks: list,
    desc: str,
    workers: int,
) -> tuple[int, int]:
    imgs = lbld = 0
    with ProcessPoolExecutor(max_workers=workers) as exe:
        futs = {exe.submit(fn, *args): None for fn, *args in tasks}
        it: object
        if HAS_TQDM:
            it = tqdm(as_completed(futs), total=len(futs), desc=desc, unit="img")
        else:
            it = as_completed(futs)
            print(f"  Processing {desc} ({len(tasks)} images) …")
        for fut in it:
            try:
                ok, has_lbl = fut.result()
            except Exception as exc:
                print(f"  [warn] worker error: {exc}")
                continue
            if ok:
                imgs += 1
            if has_lbl:
                lbld += 1
    return imgs, lbld


# ── Per-image worker functions (must be module-level for multiprocessing) ─────

def _worker_remap_yolo(
    img_path: Path,
    lbl_src_dir: Path,
    out_img: Path,
    out_lbl: Path,
    class_map: dict[int, int],
    copy_images: bool,
) -> tuple[bool, bool]:
    """Re-map YOLO class IDs and write to out_lbl. Used for AOD-4 and HIT-UAV."""
    lbl_path = lbl_src_dir / f"{img_path.stem}.txt"
    if not lbl_path.exists():
        return False, False

    link_or_copy(img_path, out_img / img_path.name, copy_images)

    out_lines: list[str] = []
    for raw in lbl_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) < 5:
            continue
        try:
            src_cls = int(parts[0])
            cx, cy, nw, nh = (
                float(parts[1]), float(parts[2]),
                float(parts[3]), float(parts[4]),
            )
        except ValueError:
            continue
        if src_cls not in class_map:
            continue  # drop DontCare, unknown classes, etc.
        out_lines.append(
            f"{class_map[src_cls]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
        )

    (out_lbl / f"{img_path.stem}.txt").write_text(
        "\n".join(out_lines) + ("\n" if out_lines else ""),
        encoding="utf-8",
    )
    return True, bool(out_lines)


def _worker_visdrone(
    img_path: Path,
    ann_dir: Path,
    out_img: Path,
    out_lbl: Path,
    copy_images: bool,
) -> tuple[bool, bool]:
    """Convert VisDrone CSV annotation to YOLO format."""
    ann_path = ann_dir / f"{img_path.stem}.txt"
    if not ann_path.exists():
        return False, False

    link_or_copy(img_path, out_img / img_path.name, copy_images)

    try:
        w, h = image_size(img_path)
    except Exception as exc:
        print(f"  [warn] image_size failed for {img_path.name}: {exc}")
        return True, False

    out_lines: list[str] = []
    for raw in ann_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split(",")
        if len(parts) < 6:
            continue
        try:
            bx    = float(parts[0])
            by    = float(parts[1])
            bw    = float(parts[2])
            bh    = float(parts[3])
            score = int(parts[4])
            cat   = int(parts[5])
        except ValueError:
            continue
        if score == 0 or cat not in VISDRONE_CLASS_MAP:
            continue
        cx = clamp01((bx + bw / 2.0) / w)
        cy = clamp01((by + bh / 2.0) / h)
        nw = clamp01(bw / w)
        nh = clamp01(bh / h)
        if nw <= 0 or nh <= 0:
            continue
        out_lines.append(
            f"{VISDRONE_CLASS_MAP[cat]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
        )

    (out_lbl / f"{img_path.stem}.txt").write_text(
        "\n".join(out_lines) + ("\n" if out_lines else ""),
        encoding="utf-8",
    )
    return True, bool(out_lines)


# ── Dataset processors ────────────────────────────────────────────────────────

def process_aod4(
    root: Path,
    out_dir: Path,
    copy_images: bool,
    workers: int,
) -> dict[str, tuple[int, int]]:
    """
    AOD-4 layout (two variants handled):

    Variant A — flat labels:
      <root>/Images/{train,valid,test}/
      <root>/Annotations/YOLOv8 format/{train,valid,test}/*.txt

    Variant B — nested labels (Roboflow/Mendeley download):
      <root>/Images/{train,valid,test}/
      <root>/Annotations/YOLOv8 format/{train,valid,test}/labels/*.txt

    'valid' is remapped to canonical 'val'.
    """
    # ── Locate annotation base ────────────────────────────────────────────────
    ann_root = root / "Annotations"
    ann_base: Optional[Path] = None

    if ann_root.exists():
        # Try known exact names first
        for candidate_name in (
            "YOLOv8 format",
            "YOLOv8format",
            "YOLOv8_format",
        ):
            p = ann_root / candidate_name
            if p.exists():
                ann_base = p
                break

        # Fallback: scan all subdirs of Annotations/ and pick the first
        # that contains at least one of the expected split folders
        if ann_base is None:
            for subdir in sorted(ann_root.iterdir()):
                if subdir.is_dir() and any(
                    (subdir / s).is_dir() for s in ("train", "valid", "test")
                ):
                    ann_base = subdir
                    print(f"  [info] AOD-4: using annotation folder '{subdir.name}'")
                    break

    # Last resort: flat labels/ next to Images/
    if ann_base is None:
        flat = root / "labels"
        if flat.exists():
            ann_base = flat

    if ann_base is None:
        print(
            f"  [warn] AOD-4: YOLOv8 annotation folder not found.\n"
            f"         Searched under: {ann_root}\n"
            f"         Expected a subfolder containing train/ valid/ or test/ dirs."
        )
        return {}

    print(f"  [info] AOD-4: annotations → {ann_base}")

    # ── Locate image base ─────────────────────────────────────────────────────
    img_base: Optional[Path] = None
    for candidate in (root / "Images", root / "images"):
        if candidate.is_dir():
            img_base = candidate
            break

    if img_base is None:
        print(f"  [warn] AOD-4: images folder not found under {root} (tried Images/, images/)")
        return {}

    print(f"  [info] AOD-4: images      → {img_base}")

    # ── Process splits ────────────────────────────────────────────────────────
    # AOD-4 uses 'valid' on disk; remap to canonical 'val'
    split_map = {"train": "train", "valid": "val", "test": "test"}
    results: dict[str, tuple[int, int]] = {}

    for disk_split, out_split in split_map.items():
        img_dir = img_base / disk_split

        # Handle both flat and nested-labels layouts:
        #   Variant A: ann_base/train/*.txt
        #   Variant B: ann_base/train/labels/*.txt
        _nested = ann_base / disk_split / "labels"
        lbl_dir = _nested if _nested.is_dir() else ann_base / disk_split

        if not img_dir.is_dir():
            print(f"  [info] AOD-4: skipping '{disk_split}' — {img_dir} not found")
            continue
        if not lbl_dir.is_dir():
            print(f"  [info] AOD-4: skipping '{disk_split}' — {lbl_dir} not found")
            continue

        print(f"  [info] AOD-4: {disk_split} labels → {lbl_dir}")

        out_img, out_lbl = _make_split_dirs(out_dir / out_split)
        img_paths = sorted(
            p for p in img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if not img_paths:
            print(f"  [warn] AOD-4: no images found in {img_dir}")
            continue

        tasks = [
            (_worker_remap_yolo, p, lbl_dir, out_img, out_lbl, AOD4_CLASS_MAP, copy_images)
            for p in img_paths
        ]
        results[out_split] = _run_parallel(tasks, f"AOD-4/{out_split}", workers)

    if not results:
        print(
            f"  [warn] AOD-4: no valid splits processed.\n"
            f"         img_base={img_base}\n"
            f"         ann_base={ann_base}"
        )
    return results

def process_hituav(
    root: Path,
    out_dir: Path,
    copy_images: bool,
    workers: int,
    val_fraction: float = 0.10,
) -> dict[str, tuple[int, int]]:
    """
    HIT-UAV — supports two layouts:

    Layout A  (Kaggle / pre-split):
      <root>/images/{train,val,test}/  *.jpg
      <root>/labels/{train,val,test}/  *.txt
      Uses the existing split as-is; val_fraction is ignored.

    Layout B  (original flat):
      <root>/yolo_labels/  *.txt  (flat, no split)
      <root>/<img_dir>/    *.jpg  (searched: images/ JPEGImages/ VideoFrames/ frames/ imgs/)
      Split into train/val by sorted filename (default 90/10).

    HIT-UAV class IDs → unified:
      0 person       →  0  person
      1 car          →  1  vehicle
      2 bicycle      →  3  bicycle
      3 OtherVehicle →  1  vehicle
      4 DontCare     →  skipped
    """

    # ── Layout A detection ────────────────────────────────────────────────────
    kaggle_img_root = root / "images"
    kaggle_lbl_root = root / "labels"
    is_kaggle_layout = (
        kaggle_img_root.is_dir()
        and kaggle_lbl_root.is_dir()
        and any((kaggle_img_root / s).is_dir() for s in ("train", "val", "test"))
        and any((kaggle_lbl_root / s).is_dir() for s in ("train", "val", "test"))
    )

    if is_kaggle_layout:
        print("  [info] HIT-UAV: detected pre-split layout (Kaggle style)")
        results: dict[str, tuple[int, int]] = {}
        for disk_split in ("train", "val", "test"):
            img_dir = kaggle_img_root / disk_split
            lbl_dir = kaggle_lbl_root / disk_split
            if not img_dir.is_dir() or not lbl_dir.is_dir():
                continue
            out_img, out_lbl = _make_split_dirs(out_dir / disk_split)
            img_paths = sorted(
                p for p in img_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            )
            if not img_paths:
                print(f"  [warn] HIT-UAV: no images found in {img_dir}")
                continue
            tasks = [
                (_worker_remap_yolo, p, lbl_dir, out_img, out_lbl, HITUAV_CLASS_MAP, copy_images)
                for p in img_paths
            ]
            results[disk_split] = _run_parallel(tasks, f"HIT-UAV/{disk_split}", workers)
        if not results:
            print(f"  [warn] HIT-UAV: pre-split layout detected but no valid splits found under {root}")
        return results

    # ── Layout B  (original flat yolo_labels/) ────────────────────────────────
    lbl_dir = root / "yolo_labels"
    if not lbl_dir.exists():
        print(
            f"  [warn] HIT-UAV: could not detect a known layout under {root}.\n"
            f"         Expected either:\n"
            f"           Layout A (Kaggle): images/{{train,val,test}}/ + labels/{{train,val,test}}/\n"
            f"           Layout B (flat):   yolo_labels/ + images/ (or JPEGImages/ etc.)"
        )
        return {}

    print("  [info] HIT-UAV: detected flat layout (original / yolo_labels style)")

    # Search for image directory
    img_dir: Optional[Path] = None
    for candidate in ("images", "JPEGImages", "VideoFrames", "frames", "imgs"):
        p = root / candidate
        if p.is_dir():
            if any(f.suffix.lower() in IMAGE_EXTS for f in p.iterdir() if f.is_file()):
                img_dir = p
                break

    if img_dir is None:
        print(
            f"  [warn] HIT-UAV: images directory not found under {root}.\n"
            f"         Tried: images/, JPEGImages/, VideoFrames/, frames/, imgs/\n"
            f"         Rename your images folder to one of the above, or symlink it."
        )
        return {}

    # Only keep images that have a matching label
    all_imgs = sorted(
        p for p in img_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
        and (lbl_dir / f"{p.stem}.txt").exists()
    )

    if not all_imgs:
        print(f"  [warn] HIT-UAV: no image/label pairs found in {img_dir} + {lbl_dir}")
        return {}

    cut = max(1, int(len(all_imgs) * (1.0 - val_fraction)))
    split_imgs = {"train": all_imgs[:cut], "val": all_imgs[cut:]}

    results = {}
    for split_name, imgs in split_imgs.items():
        if not imgs:
            continue
        out_img, out_lbl = _make_split_dirs(out_dir / split_name)
        tasks = [
            (_worker_remap_yolo, p, lbl_dir, out_img, out_lbl, HITUAV_CLASS_MAP, copy_images)
            for p in imgs
        ]
        results[split_name] = _run_parallel(tasks, f"HIT-UAV/{split_name}", workers)

    return results


def process_visdrone(
    root: Path,
    out_dir: Path,
    copy_images: bool,
    workers: int,
) -> dict[str, tuple[int, int]]:
    """
    VisDrone-DET: nested or flat layout.
    Handles both VisDrone2019-DET-{split}/{split}/ and VisDrone2019-DET-{split}/ directly.
    """
    def _locate(split: str) -> Optional[Path]:
        nested = root / f"VisDrone2019-DET-{split}" / f"VisDrone2019-DET-{split}"
        flat   = root / f"VisDrone2019-DET-{split}"
        if nested.is_dir() and (nested / "images").is_dir():
            return nested
        if flat.is_dir() and (flat / "images").is_dir():
            return flat
        return None

    results: dict[str, tuple[int, int]] = {}
    for split_name in ("train", "val"):
        split_root = _locate(split_name)
        if split_root is None:
            print(f"  [warn] VisDrone: {split_name} split not found under {root}")
            continue

        img_dir = split_root / "images"
        ann_dir = split_root / "annotations"
        if not img_dir.is_dir() or not ann_dir.is_dir():
            print(f"  [warn] VisDrone: missing images/ or annotations/ in {split_root}")
            continue

        out_img, out_lbl = _make_split_dirs(out_dir / split_name)
        img_paths = sorted(
            p for p in img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        tasks = [
            (_worker_visdrone, p, ann_dir, out_img, out_lbl, copy_images)
            for p in img_paths
        ]
        results[split_name] = _run_parallel(tasks, f"VisDrone/{split_name}", workers)

    return results


# ── Summary helpers ───────────────────────────────────────────────────────────

def _print_summary(name: str, results: dict[str, tuple[int, int]]) -> None:
    for split, (imgs, lbld) in sorted(results.items()):
        print(f"  [{name:10s}/{split:5s}]  images={imgs:6d}  labeled={lbld:6d}")


def _total(results: dict[str, tuple[int, int]]) -> tuple[int, int]:
    return sum(v[0] for v in results.values()), sum(v[1] for v in results.values())


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build a unified YOLO dataset from AOD-4, HIT-UAV, and VisDrone-DET.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--aod4",     required=True, help="AOD-4 root  (contains Images/ and Annotations/)")
    ap.add_argument("--hituav",   required=True, help="HIT-UAV root (contains yolo_labels/ and images/)")
    ap.add_argument("--visdrone", required=True, help="VisDrone-DET root")
    ap.add_argument("--out",      required=True, help="Output dataset root")
    ap.add_argument(
        "--copy-images", action="store_true",
        help="Copy images instead of symlinking (use when crossing filesystems)",
    )
    ap.add_argument(
        "--dataset-yaml", default="configs/dataset.yaml",
        help="Path to write the YOLO dataset YAML",
    )
    ap.add_argument(
        "--workers", type=int, default=os.cpu_count() or 4,
        help="Parallel worker processes",
    )
    ap.add_argument(
        "--hituav-val-fraction", type=float, default=0.10,
        help="Fraction of HIT-UAV images reserved for validation",
    )
    ap.add_argument("--skip-aod4",     action="store_true", help="Skip AOD-4")
    ap.add_argument("--skip-hituav",   action="store_true", help="Skip HIT-UAV")
    ap.add_argument("--skip-visdrone", action="store_true", help="Skip VisDrone")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    workers = max(1, args.workers)

    print(f"Workers  : {workers}")
    print(f"Output   : {out_dir.resolve()}\n")

    all_results: dict[str, dict[str, tuple[int, int]]] = {}

    if not args.skip_aod4:
        print("=== AOD-4 ===")
        root = Path(args.aod4)
        if root.is_dir():
            r = process_aod4(root, out_dir, args.copy_images, workers)
            all_results["AOD-4"] = r
            _print_summary("AOD-4", r)
        else:
            print(f"  [error] Not found: {root}")

    if not args.skip_hituav:
        print("\n=== HIT-UAV ===")
        root = Path(args.hituav)
        if root.is_dir():
            r = process_hituav(
                root, out_dir, args.copy_images, workers,
                val_fraction=args.hituav_val_fraction,
            )
            all_results["HIT-UAV"] = r
            _print_summary("HIT-UAV", r)
        else:
            print(f"  [error] Not found: {root}")

    if not args.skip_visdrone:
        print("\n=== VisDrone-DET ===")
        root = Path(args.visdrone)
        if root.is_dir():
            r = process_visdrone(root, out_dir, args.copy_images, workers)
            all_results["VisDrone"] = r
            _print_summary("VisDrone", r)
        else:
            print(f"  [error] Not found: {root}")

    # ── Dataset YAML ──────────────────────────────────────────────────────────
    def _rel(split: str) -> str:
        return f"{split}/images"

    yaml_path = Path(args.dataset_yaml)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        "\n".join([
            f"path: {out_dir.resolve()}",
            f"train: {_rel('train')}",
            f"val:   {_rel('val')}",
            f"test:  {_rel('test')}",
            f"nc: {len(CLASS_NAMES)}",
            "names:",
            *[f"  - {n}" for n in CLASS_NAMES],
            "",
        ]),
        encoding="utf-8",
    )
    print(f"\n[ok] Dataset YAML → {yaml_path}")

    # ── Grand summary ─────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print(f"  {'Dataset':<12}  {'images':>8}  {'labeled':>8}")
    print("─" * 50)
    grand_imgs = grand_lbld = 0
    for name, res in all_results.items():
        imgs, lbld = _total(res)
        grand_imgs += imgs
        grand_lbld += lbld
        print(f"  {name:<12}  {imgs:>8}  {lbld:>8}")
    print("─" * 50)
    print(f"  {'TOTAL':<12}  {grand_imgs:>8}  {grand_lbld:>8}")
    print("─" * 50)
    print(f"\n[ok] Unified dataset ready at: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())