#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_NAME="${PMTMAX_READY_MODEL_NAME:-gaussian_emos}"
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
  if [[ "${ARGS[$i]}" == "--model-name" ]] && (( i + 1 < ${#ARGS[@]} )); then
    MODEL_NAME="${ARGS[$((i + 1))]}"
  fi
done
export PMTMAX_READY_MODEL_NAME="${MODEL_NAME}"

"${SCRIPT_DIR}/ensure_local_research_ready.sh"

exec uv run python scripts/run_recent_core_benchmark.py "$@"
