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

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MODEL="trading_champion"
SCAN_EDGE_OUTPUT="artifacts/signals/v2/scan_edge_latest.json"
SCAN_EDGE_HISTORY="artifacts/signals/v2/scan_edge_history.jsonl"
LOG_PREFIX="[daily_experiment $(date -u '+%Y-%m-%dT%H:%M:%SZ')]"

echo "${LOG_PREFIX} starting"

# 1. Refresh market snapshots + Gamma prices
echo "${LOG_PREFIX} scan-markets..."
uv run pmtmax scan-markets

# 1b. Backfill truth for all active cities (uses updated station_catalog → Wunderground)
echo "${LOG_PREFIX} backfill-truth (all cities)..."
uv run pmtmax backfill-truth --markets-path artifacts/discovered_markets.json

# 1c. Backfill forecasts for all active cities
echo "${LOG_PREFIX} backfill-forecasts (all cities)..."
uv run pmtmax backfill-forecasts --markets-path artifacts/discovered_markets.json

# 2. Log Gamma prices to timeseries
echo "${LOG_PREFIX} logging gamma prices..."
uv run python scripts/log_gamma_prices.py

# 3. Run scan-edge with Gamma prices
echo "${LOG_PREFIX} scan-edge (model=${MODEL})..."
uv run pmtmax scan-edge \
    --model-name "${MODEL}" \
    --min-edge 0.05 \
    --output "${SCAN_EDGE_OUTPUT}"

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

# 5. Track outcomes for forward paper trades
echo "${LOG_PREFIX} tracking paper trade outcomes..."
uv run python scripts/track_paper_trade_outcomes.py

echo "${LOG_PREFIX} done. Signals: ${SCAN_EDGE_OUTPUT}"
