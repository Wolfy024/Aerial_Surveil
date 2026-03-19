#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_RAW_DIR="${DATA_RAW_DIR:-${ROOT_DIR}/data/raw}"

require_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "[error] Missing required command: ${cmd}" >&2
    exit 1
  fi
}

detect_ext() {
  local url="$1"
  case "${url}" in
    *.tar.gz|*.tgz) echo ".tar.gz" ;;
    *.tar.bz2|*.tbz2) echo ".tar.bz2" ;;
    *.zip) echo ".zip" ;;
    *)
      echo "[error] Could not detect archive type from URL: ${url}" >&2
      echo "Supported: .zip, .tar.gz/.tgz, .tar.bz2/.tbz2" >&2
      exit 1
      ;;
  esac
}

download_and_extract() {
  local dataset_name="$1"
  local url="$2"
  local out_dir="$3"

  if [[ -z "${url}" ]]; then
    echo "[skip] ${dataset_name}: URL not set."
    return 0
  fi

  local ext
  ext="$(detect_ext "${url}")"
  local archive="${out_dir}/${dataset_name}${ext}"

  mkdir -p "${out_dir}"
  echo "[download] ${dataset_name}: ${url}"
  curl --fail --location --retry 3 --output "${archive}" "${url}"

  echo "[extract] ${archive} -> ${out_dir}"
  case "${ext}" in
    .zip) unzip -q "${archive}" -d "${out_dir}" ;;
    .tar.gz) tar -xzf "${archive}" -C "${out_dir}" ;;
    .tar.bz2) tar -xjf "${archive}" -C "${out_dir}" ;;
    *)
      echo "[error] Unsupported archive extension: ${ext}" >&2
      exit 1
      ;;
  esac
}

main() {
  require_cmd curl
  require_cmd unzip
  require_cmd tar

  local visdrone_url="${VISDRONE_URL:-}"
  local hituav_url="${HITUAV_URL:-}"
  local thermal_vision_url="${THERMAL_VISION_URL:-}"
  local monet_url="${MONET_URL:-}"
  local aod4_url="${AOD4_URL:-}"

  download_and_extract "visdrone" "${visdrone_url}" "${DATA_RAW_DIR}/visdrone"
  download_and_extract "hit-uav" "${hituav_url}" "${DATA_RAW_DIR}/hit-uav"
  download_and_extract "thermal-vision" "${thermal_vision_url}" "${DATA_RAW_DIR}/thermal-vision"
  download_and_extract "monet" "${monet_url}" "${DATA_RAW_DIR}/monet"
  download_and_extract "aod4" "${aod4_url}" "${DATA_RAW_DIR}/aod-4"

  echo "[ok] Download step completed. Extracted data root: ${DATA_RAW_DIR}"
}

main "$@"

