#!/usr/bin/env bash
# Phase 15 NWP diversity pipeline: build → train → autoresearch
# Logs: artifacts/batch_logs/phase15_*.log
# Status: artifacts/batch_logs/.phase15_status.json

set -euo pipefail

export PATH="/home/seok436/.local/bin:$PATH"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

LOG_DIR="artifacts/batch_logs"
STATUS_FILE="${LOG_DIR}/.phase15_status.json"
mkdir -p "${LOG_DIR}"

# Initialize status
cat > "${STATUS_FILE}" << 'EOF'
{
  "started_at": "",
  "stage": "initializing",
  "stage_progress": "0%",
  "overall_progress": "0%",
  "build_dataset": {"status": "pending", "duration": ""},
  "train_advanced": {"status": "pending", "duration": ""},
  "autoresearch_init": {"status": "pending", "duration": ""}
}
EOF

function update_status() {
    local stage=$1
    local status=$2
    local progress=$3
    python3 << PYEOF
import json, sys
from datetime import datetime
with open('${STATUS_FILE}', 'r') as f:
    data = json.load(f)
data['stage'] = '${stage}'
data['stage_progress'] = '${progress}'
data['updated_at'] = datetime.utcnow().isoformat()
if '${status}' != '': data['${stage}']['status'] = '${status}'
with open('${STATUS_FILE}', 'w') as f: json.dump(data, f, indent=2)
PYEOF
}

function log_step() {
    local msg=$1
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $msg"
}

# ============= STAGE 1: build-dataset =============
echo "====== STAGE 1: build-dataset (NWP diversity) ======"
update_status "build_dataset" "running" "0%"
START_TIME=$(date +%s)

log_step "Starting materialize-training-set (synthetic_ready 91k, existing DB only)..."
if uv run pmtmax materialize-training-set \
    --markets-path configs/market_inventory/synthetic_ready_snapshots.json \
    --decision-horizon market_open \
    --decision-horizon previous_evening \
    --decision-horizon morning_of \
    --allow-canonical-overwrite \
    2>&1 | tee "${LOG_DIR}/phase15_01_build_dataset.log"; then

    DURATION=$(($(date +%s) - START_TIME))
    update_status "build_dataset" "completed" "100%"
    log_step "✅ materialize-training-set completed in ${DURATION}s"
else
    log_step "❌ materialize-training-set FAILED"
    exit 1
fi

# ============= STAGE 2: train-advanced =============
echo ""
echo "====== STAGE 2: train-advanced (recency baseline) ======"
update_status "train_advanced" "running" "0%"
START_TIME=$(date +%s)

log_step "Starting train-advanced..."
if uv run pmtmax train-advanced \
    --model-name lgbm_emos \
    --variant recency_neighbor_oof \
    2>&1 | tee "${LOG_DIR}/phase15_02_train_advanced.log"; then

    DURATION=$(($(date +%s) - START_TIME))
    update_status "train_advanced" "completed" "100%"
    log_step "✅ train-advanced completed in ${DURATION}s"
else
    log_step "❌ train-advanced FAILED"
    exit 1
fi

# ============= STAGE 3: autoresearch-init =============
echo ""
echo "====== STAGE 3: autoresearch-init (Phase 15) ======"
update_status "autoresearch_init" "running" "0%"
START_TIME=$(date +%s)

log_step "Starting autoresearch-init..."
if uv run pmtmax autoresearch-init \
    2>&1 | tee "${LOG_DIR}/phase15_03_autoresearch_init.log"; then

    DURATION=$(($(date +%s) - START_TIME))
    update_status "autoresearch_init" "completed" "100%"
    log_step "✅ autoresearch-init completed in ${DURATION}s"
else
    log_step "❌ autoresearch-init FAILED"
    exit 1
fi

echo ""
echo "====== ALL DONE ======"
log_step "Phase 15 pipeline completed successfully!"
