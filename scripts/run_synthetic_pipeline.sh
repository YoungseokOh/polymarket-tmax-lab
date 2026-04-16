#!/usr/bin/env bash
# Full synthetic data pipeline: inject → build-dataset → merge → benchmark
# Logs to /tmp/synthetic_pipeline.log
set -euo pipefail

REPO=/home/seok436/projects/polymarket-tmax-lab
LOG=/tmp/synthetic_pipeline.log
cd "$REPO"

if [[ "${PMTMAX_WORKSPACE_NAME:-}" != "research_synth" ]]; then
  exec scripts/pmtmax-workspace research_synth bash "$0" "$@"
fi

source .venv/bin/activate

PARQUET_ROOT="${PMTMAX_PARQUET_DIR:-$REPO/data/parquet}"
GOLD_ROOT="${PARQUET_ROOT}/gold"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== Synthetic Pipeline Start ==="

# Step 1: Wait for inject to finish (PID file or poll log)
log "STEP 1: Waiting for inject_synthetic_data.py to finish..."
while pgrep -f "inject_synthetic_data.py" > /dev/null 2>&1; do
    sleep 30
done
log "STEP 1: inject done."

# Verify inject results
python3 -c "
import sys
sys.path.insert(0, 'src')
import duckdb
import os
db = duckdb.connect(os.environ.get('PMTMAX_DUCKDB_PATH', 'data/duckdb/warehouse.duckdb'))
f = db.execute(\"SELECT count(*) FROM silver_forecast_runs_hourly WHERE market_id LIKE 'synthetic_%'\").fetchone()[0]
t = db.execute(\"SELECT count(*) FROM silver_observations_daily WHERE market_id LIKE 'synthetic_%'\").fetchone()[0]
print(f'Synthetic in DuckDB: {f:,} forecast, {t:,} truth')
" | tee -a "$LOG"

# Step 2: build-dataset for synthetic data
log "STEP 2: build-dataset (synthetic)..."
uv run pmtmax build-dataset \
    --markets-path configs/market_inventory/synthetic_historical_snapshots.json \
    --output-name synthetic_historical_training_set \
    2>&1 | tee -a "$LOG"
log "STEP 2: build-dataset done."

# Verify output
ls -lh "${GOLD_ROOT}"/synthetic_historical_training_set*.parquet 2>/dev/null | tee -a "$LOG" || true

# Step 3: merge training sets
log "STEP 3: merge_training_sets.py..."
uv run python scripts/merge_training_sets.py 2>&1 | tee -a "$LOG"
log "STEP 3: merge done."

# Verify merged output
python3 -c "
import pandas as pd
from pathlib import Path
p = Path('${GOLD_ROOT}/expanded_training_set.parquet')
if p.exists():
    df = pd.read_parquet(p)
    print(f'expanded_training_set: {len(df):,} rows, {df[\"city\"].nunique()} cities')
    print(f'date range: {df[\"target_date\"].min()} ~ {df[\"target_date\"].max()}')
else:
    print('ERROR: expanded_training_set.parquet not found')
" | tee -a "$LOG"

# Step 4: benchmark-models
log "STEP 4: benchmark-models..."
uv run pmtmax benchmark-models \
    --dataset-path "${GOLD_ROOT}/expanded_training_set.parquet" \
    --retrain-stride 30 \
    2>&1 | tee -a "$LOG"
log "STEP 4: benchmark done."

log "=== Pipeline Complete ==="
