#!/usr/bin/env bash
# Resume synthetic NWP single-run backfill (smart scheduler with 24h rate-limit guard).
#
# This script:
# 1. Checks if 24+ hours have passed since the last run
# 2. If yes, resumes backfill-forecasts with --missing-only
# 3. Runs daily but only collects if API rate limit has reset
#
# Schedule: Run daily at 01:00 UTC (10:00 KST) via cron:
#   0 1 * * * /home/seok436/projects/polymarket-tmax-lab/scripts/resume_single_run_backfill.sh >> /home/seok436/projects/polymarket-tmax-lab/artifacts/batch_logs/resume_single_run.log 2>&1

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PATH="/home/seok436/.local/bin:$PATH"

MARKETS_FILE="configs/market_inventory/synthetic_historical_snapshots.json"
CHECKPOINT_FILE="artifacts/batch_logs/.single_run_last_checkpoint"
LOG_FILE="artifacts/batch_logs/single_run_$(date -u '+%Y%m%dT%H%M%S').log"
HOURS_REQUIRED=24

mkdir -p artifacts/batch_logs

echo "[resume_single_run] $(date -u '+%Y-%m-%dT%H:%M:%SZ') checking..."

# Check if checkpoint file exists
if [[ ! -f "${CHECKPOINT_FILE}" ]]; then
    echo "[resume_single_run] First run, no checkpoint. Starting collection..."
    SHOULD_RUN=true
else
    LAST_RUN=$(cat "${CHECKPOINT_FILE}")
    NOW=$(date +%s)
    SECONDS_SINCE=$((NOW - LAST_RUN))
    HOURS_SINCE=$((SECONDS_SINCE / 3600))

    if [[ ${HOURS_SINCE} -ge ${HOURS_REQUIRED} ]]; then
        echo "[resume_single_run] ${HOURS_SINCE}h since last run (required: ${HOURS_REQUIRED}h). Resuming..."
        SHOULD_RUN=true
    else
        HOURS_REMAINING=$((HOURS_REQUIRED - HOURS_SINCE))
        echo "[resume_single_run] Only ${HOURS_SINCE}h since last run. Need ${HOURS_REMAINING}h more. Skipping."
        SHOULD_RUN=false
    fi
fi

if [[ "${SHOULD_RUN}" == "true" ]]; then
    echo "[resume_single_run] Running backfill-forecasts with --missing-only..."
    date +%s > "${CHECKPOINT_FILE}"

    nohup uv run pmtmax backfill-forecasts \
        --markets-path "${MARKETS_FILE}" \
        --model gfs_seamless \
        --model ecmwf_ifs025 \
        --model kma_gdps \
        --single-run-horizon morning_of \
        --single-run-horizon previous_evening \
        --single-run-horizon market_open \
        --missing-only \
        > "${LOG_FILE}" 2>&1 &

    PID=$!
    echo "[resume_single_run] Started PID ${PID}, logging to ${LOG_FILE}"
fi

echo "[resume_single_run] Done."
