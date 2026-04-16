#!/usr/bin/env bash
# Daily experiment loop: collect Gamma prices + scan-edge signals.
#
# What this does each run:
#   1. scan-markets  — refresh discovered_markets.json with Gamma prices
#   2. log_gamma_prices — append prices to gamma_price_log.jsonl (for backtest expansion)
#   3. scan-edge     — generate today's signals using Gamma mid-prices
#   4. Append scan-edge snapshot to scan_edge_history.jsonl
#
# Schedule: run once per day (e.g., 09:00 KST = 00:00 UTC)
set -euo pipefail

# Ensure uv and python are on PATH (cron doesn't load ~/.bashrc)
export PATH="/home/seok436/.local/bin:$PATH"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${PMTMAX_WORKSPACE_NAME:-}" != "ops_daily" ]]; then
    exec "${REPO_ROOT}/scripts/pmtmax-workspace" ops_daily "${REPO_ROOT}/scripts/daily_experiment.sh" "$@"
fi

ARTIFACTS_ROOT="${PMTMAX_ARTIFACTS_DIR:-${REPO_ROOT}/artifacts}"
MODEL="champion"
DISCOVERED_MARKETS_PATH="${ARTIFACTS_ROOT}/discovered_markets.json"
SCAN_EDGE_OUTPUT="${ARTIFACTS_ROOT}/signals/v2/scan_edge_latest.json"
SCAN_EDGE_HISTORY="${ARTIFACTS_ROOT}/signals/v2/scan_edge_history.jsonl"
PAPER_SIGNALS_DIR="${ARTIFACTS_ROOT}/signals/v2/paper_snapshots"
PAPER_SIGNALS_LATEST="${ARTIFACTS_ROOT}/signals/v2/paper_signals_latest.json"

# Ensure log directory exists (cron will abort if missing)
mkdir -p logs "${ARTIFACTS_ROOT}/signals/v2" "${PAPER_SIGNALS_DIR}"
LOG_PREFIX="[daily_experiment $(date -u '+%Y-%m-%dT%H:%M:%SZ')]"

LOCK_FILE="${REPO_ROOT}/logs/daily_experiment.lock"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    echo "${LOG_PREFIX} skipped: another daily experiment run is still active"
    exit 0
fi

echo "${LOG_PREFIX} starting"

# 1. Refresh market snapshots + Gamma prices
echo "${LOG_PREFIX} scan-markets..."
uv run pmtmax scan-markets --output "${DISCOVERED_MARKETS_PATH}"

# 1b. Backfill truth for all active cities (uses updated station_catalog → Wunderground)
echo "${LOG_PREFIX} backfill-truth (all cities)..."
uv run pmtmax backfill-truth --markets-path "${DISCOVERED_MARKETS_PATH}"

# 1c. Backfill forecasts for all active cities
echo "${LOG_PREFIX} backfill-forecasts (all cities)..."
uv run pmtmax backfill-forecasts --markets-path "${DISCOVERED_MARKETS_PATH}"

# 2. Log Gamma prices to timeseries
echo "${LOG_PREFIX} logging gamma prices..."
uv run python scripts/log_gamma_prices.py

# 3. Run scan-edge with Gamma prices
echo "${LOG_PREFIX} scan-edge (model=${MODEL})..."
uv run pmtmax scan-edge \
    --model-name "${MODEL}" \
    --min-edge 0.15 \
    --min-model-prob 0.05 \
    --max-model-prob 0.95 \
    --min-market-price 0.10 \
    --min-gamma 0.15 \
    --max-gamma 0.85 \
    --max-no-gamma 0.70 \
    --output "${SCAN_EDGE_OUTPUT}"

# 3b. Record new signals as forward paper trades
echo "${LOG_PREFIX} recording paper trades..."
uv run python scripts/record_paper_trades.py

# 3c. Record ALL signals (unfiltered) for direction accuracy analysis
echo "${LOG_PREFIX} recording all signals (unfiltered)..."
uv run python scripts/record_all_signals.py

# 4. Append snapshot to history
if [[ -f "${SCAN_EDGE_OUTPUT}" ]]; then
    TIMESTAMP=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
    python3 -c "
import json, sys
with open('${SCAN_EDGE_OUTPUT}') as f:
    signals = json.load(f)
record = {'snapshot_at': '${TIMESTAMP}', 'signals': signals}
with open('${SCAN_EDGE_HISTORY}', 'a') as f:
    f.write(json.dumps(record) + '\n')
print(f'Appended {len(signals)} signals to history.')
"
fi

# 5. Run paper-trader (Kelly-sized positions, gamma prices, min 10% market price)
PAPER_DATE=$(date -u '+%Y-%m-%d')
PAPER_SNAPSHOT="${PAPER_SIGNALS_DIR}/paper_signals_${PAPER_DATE}.json"
echo "${LOG_PREFIX} paper-trader (bankroll=100, min-market-price=0.10)..."
uv run pmtmax paper-trader \
    --bankroll 100 \
    --price-source gamma \
    --min-market-price 0.10 \
    --output "${PAPER_SNAPSHOT}"
# Also keep a "latest" pointer
cp "${PAPER_SNAPSHOT}" "${PAPER_SIGNALS_LATEST}"

# 6. Track outcomes for forward paper trades
echo "${LOG_PREFIX} tracking paper trade outcomes..."
uv run python scripts/track_paper_trade_outcomes.py

echo "${LOG_PREFIX} done. Signals: ${SCAN_EDGE_OUTPUT}"
