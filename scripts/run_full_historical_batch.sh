#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "${PMTMAX_WORKSPACE_NAME:-}" != "historical_real" ]]; then
  exec "${ROOT_DIR}/scripts/pmtmax-workspace" historical_real "$0" "$@"
fi

MARKETS_PATH="configs/market_inventory/historical_temperature_snapshots.json"
LOG_DIR="${PMTMAX_ARTIFACTS_DIR:-artifacts/workspaces/historical_real}/batch_logs"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="$LOG_DIR/full_historical_batch_${TIMESTAMP}.log"
MAX_PAGES=""
SKIP_REFRESH=0
SKIP_WATCHLIST=0
SKIP_MODEL_SMOKE=0
CITY_ARGS=()

usage() {
  cat <<'EOF'
Usage: scripts/run_full_historical_batch.sh [options]

Long-running shell entrypoint for supported-city historical expansion.

Options:
  --city CITY             Limit refresh/watchlist discovery and backfill scope to one supported city. Repeatable.
  --max-pages N           Gamma event pages to scan during refresh/watchlist.
  --skip-refresh          Reuse the checked-in URL manifest without refreshing closed events first.
  --skip-watchlist        Skip active watchlist refresh at the end.
  --skip-model-smoke      Skip train-baseline and backtest smoke.
  --log-file PATH         Override the batch log path.
  --markets-path PATH     Override the curated snapshot inventory path.
  --help                  Show this message and exit.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --city)
      CITY_ARGS+=("--city" "$2")
      shift 2
      ;;
    --max-pages)
      MAX_PAGES="$2"
      shift 2
      ;;
    --skip-refresh)
      SKIP_REFRESH=1
      shift
      ;;
    --skip-watchlist)
      SKIP_WATCHLIST=1
      shift
      ;;
    --skip-model-smoke)
      SKIP_MODEL_SMOKE=1
      shift
      ;;
    --log-file)
      LOG_FILE="$2"
      shift 2
      ;;
    --markets-path)
      MARKETS_PATH="$2"
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

mkdir -p "$(dirname "$LOG_FILE")"

run_step() {
  local label="$1"
  shift
  printf '\n[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$label" | tee -a "$LOG_FILE"
  set +e
  "$@" 2>&1 | tee -a "$LOG_FILE"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ $status -ne 0 ]]; then
    printf '[%s] FAILED: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$label" | tee -a "$LOG_FILE"
    exit "$status"
  fi
}

append_max_pages() {
  local -n target_ref=$1
  if [[ -n "$MAX_PAGES" ]]; then
    target_ref+=("--max-pages" "$MAX_PAGES")
  fi
}

printf '[%s] Batch log: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$LOG_FILE" | tee -a "$LOG_FILE"

if [[ $SKIP_REFRESH -eq 0 ]]; then
  refresh_cmd=(scripts/run_historical_refresh_pipeline.sh --stage all)
  refresh_cmd+=("${CITY_ARGS[@]}")
  append_max_pages refresh_cmd
  run_step "Refresh closed historical event URL manifest" "${refresh_cmd[@]}"
fi

run_step "Build curated historical snapshot inventory" \
  uv run python scripts/build_historical_market_inventory.py \
    --output "$MARKETS_PATH"

run_step "Validate curated historical snapshot inventory" \
  uv run python scripts/validate_historical_market_inventory.py \
    --input "$MARKETS_PATH"

if [[ ! -f "$MARKETS_PATH" ]]; then
  printf '[%s] FAILED: missing curated inventory %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$MARKETS_PATH" | tee -a "$LOG_FILE"
  exit 1
fi

run_step "Initialize canonical warehouse layout" \
  uv run pmtmax init-warehouse

run_step "Backfill market metadata" \
  uv run pmtmax backfill-markets --markets-path "$MARKETS_PATH" "${CITY_ARGS[@]}"

run_step "Backfill archived forecasts" \
  uv run pmtmax backfill-forecasts \
    --markets-path "$MARKETS_PATH" \
    "${CITY_ARGS[@]}" \
    --strict-archive \
    --missing-only \
    --single-run-horizon market_open \
    --single-run-horizon previous_evening \
    --single-run-horizon morning_of

run_step "Backfill official truth" \
  uv run pmtmax backfill-truth --markets-path "$MARKETS_PATH" "${CITY_ARGS[@]}"

run_step "Materialize training dataset" \
  uv run pmtmax materialize-training-set \
    --markets-path "$MARKETS_PATH" \
    "${CITY_ARGS[@]}" \
    --allow-canonical-overwrite \
    --decision-horizon market_open \
    --decision-horizon previous_evening \
    --decision-horizon morning_of

run_step "Summarize forecast availability" \
  uv run pmtmax summarize-forecast-availability

run_step "Compact canonical warehouse" \
  uv run pmtmax compact-warehouse

if [[ $SKIP_MODEL_SMOKE -eq 0 ]]; then
  run_step "Train baseline model" \
    uv run pmtmax train-baseline --model-name gaussian_emos

  run_step "Run research backtest" \
    uv run pmtmax backtest --pricing-source real_history --model-name gaussian_emos
fi

if [[ $SKIP_WATCHLIST -eq 0 ]]; then
  watchlist_cmd=(uv run python scripts/build_active_weather_watchlist.py)
  watchlist_cmd+=("${CITY_ARGS[@]}")
  append_max_pages watchlist_cmd
  run_step "Refresh active supported-city watchlist" "${watchlist_cmd[@]}"
fi

printf '\n[%s] Historical batch complete.\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"
