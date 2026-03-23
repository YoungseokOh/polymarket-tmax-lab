#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_NAME="${PMTMAX_READY_MODEL_NAME:-gaussian_emos}"
MODEL_PATH="${PMTMAX_READY_MODEL_PATH:-artifacts/models/${MODEL_NAME}.pkl}"
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
  if [[ "${ARGS[$i]}" == "--model-name" ]] && (( i + 1 < ${#ARGS[@]} )); then
    MODEL_NAME="${ARGS[$((i + 1))]}"
  fi
  if [[ "${ARGS[$i]}" == "--model-path" ]] && (( i + 1 < ${#ARGS[@]} )); then
    MODEL_PATH="${ARGS[$((i + 1))]}"
  fi
done
export PMTMAX_READY_MODEL_NAME="${MODEL_NAME}"
export PMTMAX_READY_MODEL_PATH="${MODEL_PATH}"

"${SCRIPT_DIR}/ensure_local_research_ready.sh"

exec uv run pmtmax opportunity-shadow \
  --model-path "${MODEL_PATH}" \
  --model-name "${MODEL_NAME}" \
  "$@"
