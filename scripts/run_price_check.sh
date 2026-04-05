#!/usr/bin/env bash
# 2-hour price check loop: append Gamma prices and settle forward paper trades.
#
# Cron-safe wrapper:
#   - fixes PATH because cron skips shell init
#   - fixes working directory because the Python scripts use repo-relative paths
#   - ensures the log directory exists before redirect targets are opened
#   - prevents overlapping runs with flock
set -euo pipefail

export PATH="/home/seok436/.local/bin:$PATH"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs artifacts/signals/v2

LOCK_FILE="${REPO_ROOT}/logs/price_check.lock"
LOG_PREFIX="[price_check $(date -u '+%Y-%m-%dT%H:%M:%SZ')]"

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    echo "${LOG_PREFIX} skipped: another price check run is still active"
    exit 0
fi

echo "${LOG_PREFIX} starting"

echo "${LOG_PREFIX} logging gamma prices..."
uv run python scripts/log_gamma_prices.py

echo "${LOG_PREFIX} tracking paper trade outcomes..."
uv run python scripts/track_paper_trade_outcomes.py

echo "${LOG_PREFIX} done"
