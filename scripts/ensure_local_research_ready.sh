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
if [[ -n "${PMTMAX_READY_MODEL_PATH:-}" ]]; then
  MODEL_PATH="${PMTMAX_READY_MODEL_PATH}"
elif [[ "${MODEL_NAME}" == "champion" ]]; then
  MODEL_PATH="artifacts/public_models/champion.pkl"
else
  MODEL_PATH="artifacts/models/${MODEL_NAME}.pkl"
fi
MODEL_DIR="$(dirname "${MODEL_PATH}")"

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "[pmtmax] canonical dataset missing -> running bootstrap-lab"
  uv run pmtmax bootstrap-lab
fi

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "[pmtmax] dataset still missing after bootstrap: ${DATASET_PATH}" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}" && "${MODEL_NAME}" != "champion" ]]; then
  mkdir -p "${MODEL_DIR}"
  echo "[pmtmax] baseline model missing -> training ${MODEL_NAME}"
  uv run pmtmax train-baseline \
    --dataset-path "${DATASET_PATH}" \
    --model-name "${MODEL_NAME}" \
    --artifacts-dir "${MODEL_DIR}"
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  if [[ "${MODEL_NAME}" == "champion" ]]; then
    echo "[pmtmax] public champion alias missing: ${MODEL_PATH}" >&2
    echo "[pmtmax] publish or restore artifacts/public_models/champion.* before using the champion alias." >&2
  else
    echo "[pmtmax] model still missing after training: ${MODEL_PATH}" >&2
  fi
  exit 1
fi

printf '[pmtmax] ready dataset=%s model=%s model_name=%s\n' \
  "${DATASET_PATH}" "${MODEL_PATH}" "${MODEL_NAME}"
