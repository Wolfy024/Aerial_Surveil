# Border Surveillance Baseline (YOLO + Tracking)

Runnable starter for dataset preparation, YOLO training, object tracking, and model export.

## What is in this repo right now

- Dataset folders under `data/`
- Dataset download helper: `scripts/download_datasets.sh`
- Bootstrapped Python entrypoints:
  - `scripts/prepare_datasets.py`
  - `scripts/train.py`
  - `scripts/track.py`
  - `scripts/export.py`
- Config files under `configs/`

## Project layout

```text
Surveil/
├── README.md
├── requirements.txt
├── configs/
│   ├── dataset.yaml
│   ├── rgb_dataset.yaml
│   ├── thermal_dataset.yaml
│   ├── bytetrack.yaml
│   └── hyp.yaml
├── scripts/
│   ├── download_datasets.sh
│   ├── prepare_datasets.py
│   ├── train.py
│   ├── track.py
│   └── export.py
└── data/
```

## Prerequisites

- Python 3.10+
- Linux/macOS shell
- Optional but recommended: CUDA-capable GPU

System packages for dataset downloader:

```bash
sudo apt-get update
sudo apt-get install -y curl unzip tar
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset download (optional)

`scripts/download_datasets.sh` is URL-driven. It will skip datasets whose URLs are unset.

```bash
chmod +x scripts/download_datasets.sh
VISDRONE_URL="https://example.com/visdrone.zip" \
HITUAV_URL="https://example.com/hituav.zip" \
bash scripts/download_datasets.sh
```

Supported env vars:

- `VISDRONE_URL`
- `HITUAV_URL`
- `THERMAL_VISION_URL`
- `MONET_URL`
- `AOD4_URL`

## Prepare datasets (baseline indexing + YAML)

This baseline script validates dataset directories, scans for images, creates train/val/test index lists, and writes a dataset YAML. It does not perform deep annotation conversion yet.

```bash
python scripts/prepare_datasets.py \
  --aod4 "data/AOD 4 Dataset for Air Borne Object Detection" \
  --hituav "data/HIT-UAV-Infrared-Thermal-Dataset-main" \
  --visdrone "data/VisDrone" \
  --out datasets/merged
```

Expected generated files:

- `datasets/merged/train/images/`
- `datasets/merged/train/labels/`
- `datasets/merged/val/images/`
- `datasets/merged/val/labels/`
- `configs/dataset.yaml` (or custom `--dataset-yaml`)

## Train

```bash
python scripts/train.py \
  --data configs/dataset.yaml \
  --model yolo26m.pt \
  --epochs 10 \
  --imgsz 640 \
  --batch 8 \
  --project runs/train \
  --name baseline
```

## Track

```bash
python scripts/track.py \
  --source video.mp4 \
  --weights runs/train/baseline/weights/best.pt \
  --tracker configs/bytetrack.yaml \
  --imgsz 640 \
  --conf 0.25 \
  --iou 0.45 \
  --save
```

## Export

```bash
python scripts/export.py \
  --weights runs/train/baseline/weights/best.pt \
  --format onnx \
  --imgsz 640 \
  --dynamic
```

## Smoke test commands used for this baseline

```bash
python scripts/prepare_datasets.py --help
python scripts/train.py --help
python scripts/track.py --help
python scripts/export.py --help
bash scripts/download_datasets.sh
```

If `--help` works, scripts are wired correctly. For full pipeline execution, you still need valid YOLO-formatted labels and downloadable model weights.

### Validation notes from this bootstrap run

- `python scripts/* --help` commands executed successfully.
- `bash scripts/download_datasets.sh` executed successfully and skipped all datasets because URL env vars were not set.
- `pip install -r requirements.txt` may fail in restricted/proxied environments; rerun on a machine with normal PyPI access.

## Current limitations

- VisDrone DET conversion is implemented; AOD-4 and HIT-UAV conversion is not implemented yet.
- Training/tracking/export wrappers require `ultralytics` and valid data/weights at runtime.
- Tracker and hyperparameter configs are starter defaults and should be tuned per hardware/data.

## License

MIT