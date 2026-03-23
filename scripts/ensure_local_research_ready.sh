#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "[pmtmax] uv is required but not installed." >&2
  exit 1
fi

MODEL_NAME="${PMTMAX_READY_MODEL_NAME:-gaussian_emos}"
DATASET_PATH="${PMTMAX_READY_DATASET_PATH:-data/parquet/gold/historical_training_set.parquet}"
MODEL_PATH="${PMTMAX_READY_MODEL_PATH:-artifacts/models/${MODEL_NAME}.pkl}"
MODEL_DIR="$(dirname "${MODEL_PATH}")"

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "[pmtmax] canonical dataset missing -> running bootstrap-lab"
  uv run pmtmax bootstrap-lab
fi

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "[pmtmax] dataset still missing after bootstrap: ${DATASET_PATH}" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  mkdir -p "${MODEL_DIR}"
  echo "[pmtmax] baseline model missing -> training ${MODEL_NAME}"
  uv run pmtmax train-baseline \
    --dataset-path "${DATASET_PATH}" \
    --model-name "${MODEL_NAME}" \
    --artifacts-dir "${MODEL_DIR}"
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "[pmtmax] model still missing after training: ${MODEL_PATH}" >&2
  exit 1
fi

printf '[pmtmax] ready dataset=%s model=%s model_name=%s\n' \
  "${DATASET_PATH}" "${MODEL_PATH}" "${MODEL_NAME}"
