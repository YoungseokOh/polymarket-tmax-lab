#!/usr/bin/env bash
# Run the paper trader with the current champion model.
# Writes results to artifacts/paper_trader/YYYY-MM-DD_HH.json
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

"${SCRIPT_DIR}/ensure_local_research_ready.sh"

OUTPUT_DIR="${REPO_ROOT}/artifacts/paper_trader"
mkdir -p "${OUTPUT_DIR}"
TIMESTAMP="$(date -u '+%Y-%m-%d_%H')"
OUTPUT_FILE="${OUTPUT_DIR}/${TIMESTAMP}.json"

uv run pmtmax paper-trader "$@" | tee "${OUTPUT_FILE}"
echo "[paper-trader] saved -> ${OUTPUT_FILE}" >&2
