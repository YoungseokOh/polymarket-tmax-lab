#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="artifacts/batch_logs"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="$LOG_DIR/historical_refresh_${TIMESTAMP}.log"
STAGE="all"
MAX_PAGES=""
MAX_EVENTS=""
FETCH_WORKERS="8"
TRUTH_WORKERS="4"
TRUTH_PER_SOURCE_LIMIT="2"
CHECKPOINT_EVERY="25"
NO_RESUME=0
NO_CACHE=0
FILL_GAPS_ONLY=0
CITY_ARGS=()
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: scripts/run_historical_refresh_pipeline.sh [options]

Resumable closed-event refresh wrapper for supported-city historical expansion.

Options:
  --stage STAGE           discover | fetch-pages | classify | publish | all (default: all)
  --city CITY             Limit processing to one supported city. Repeatable.
  --max-pages N           Gamma event pages to scan during discovery.
  --max-events N          Maximum events to process in the selected stage.
  --fetch-workers N       Bounded concurrency for Polymarket event page fetches.
  --truth-workers N       Bounded concurrency for exact-source truth probes.
  --truth-per-source-limit N
                        Maximum concurrent truth probes per official source family.
  --checkpoint-every N    Persist fetch/classify progress after each batch of N events.
  --status-filter STATUS  Re-run only entries with the given collection status. Repeatable.
  --fill-gaps-only        Skip discovery and only fill fetch/classify/publish gaps from existing manifests.
  --no-resume             Ignore existing candidate/fetch/status manifests for this run.
  --no-cache              Disable cache reads while fetching new pages or truth payloads.
  --log-file PATH         Override the refresh log path.
  --output PATH           Override the published collected URL manifest path.
  --candidates-path PATH  Override the candidate backlog manifest path.
  --fetch-report PATH     Override the page fetch report path.
  --report PATH           Override the collection status report path.
  --help                  Show this message and exit.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)
      STAGE="$2"
      shift 2
      ;;
    --city)
      CITY_ARGS+=("--city" "$2")
      shift 2
      ;;
    --max-pages)
      MAX_PAGES="$2"
      shift 2
      ;;
    --max-events)
      MAX_EVENTS="$2"
      shift 2
      ;;
    --fetch-workers)
      FETCH_WORKERS="$2"
      shift 2
      ;;
    --truth-workers)
      TRUTH_WORKERS="$2"
      shift 2
      ;;
    --truth-per-source-limit)
      TRUTH_PER_SOURCE_LIMIT="$2"
      shift 2
      ;;
    --checkpoint-every)
      CHECKPOINT_EVERY="$2"
      shift 2
      ;;
    --status-filter)
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --fill-gaps-only)
      FILL_GAPS_ONLY=1
      shift
      ;;
    --no-resume)
      NO_RESUME=1
      shift
      ;;
    --no-cache)
      NO_CACHE=1
      shift
      ;;
    --log-file)
      LOG_FILE="$2"
      shift 2
      ;;
    --output|--candidates-path|--fetch-report|--report)
      EXTRA_ARGS+=("$1" "$2")
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

printf '[%s] Refresh log: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$LOG_FILE" | tee -a "$LOG_FILE"

build_refresh_cmd() {
  local stage="$1"
  local -n target_ref=$2
  target_ref=(
    uv run python scripts/refresh_historical_event_urls.py
    --stage "$stage"
    --fetch-workers "$FETCH_WORKERS"
    --truth-workers "$TRUTH_WORKERS"
    --truth-per-source-limit "$TRUTH_PER_SOURCE_LIMIT"
    --checkpoint-every "$CHECKPOINT_EVERY"
  )
  target_ref+=("${CITY_ARGS[@]}")
  target_ref+=("${EXTRA_ARGS[@]}")
  if [[ -n "$MAX_PAGES" ]]; then
    target_ref+=("--max-pages" "$MAX_PAGES")
  fi
  if [[ -n "$MAX_EVENTS" ]]; then
    target_ref+=("--max-events" "$MAX_EVENTS")
  fi
  if [[ $NO_RESUME -eq 1 ]]; then
    target_ref+=("--no-resume")
  fi
  if [[ $NO_CACHE -eq 1 ]]; then
    target_ref+=("--no-cache")
  fi
}

run_refresh_cmd() {
  local label="$1"
  shift
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$label" | tee -a "$LOG_FILE"
  printf '[%s] Command: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG_FILE"
  set +e
  "$@" 2>&1 | tee -a "$LOG_FILE"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ $status -ne 0 ]]; then
    printf '[%s] FAILED historical refresh.\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"
    exit "$status"
  fi
}

if [[ $FILL_GAPS_ONLY -eq 1 ]]; then
  fetch_cmd=()
  classify_cmd=()
  publish_cmd=()
  build_refresh_cmd "fetch-pages" fetch_cmd
  build_refresh_cmd "classify" classify_cmd
  build_refresh_cmd "publish" publish_cmd
  run_refresh_cmd "Fill fetch gaps from existing manifests" "${fetch_cmd[@]}"
  run_refresh_cmd "Fill classify gaps from existing manifests" "${classify_cmd[@]}"
  run_refresh_cmd "Publish collected URLs from existing manifests" "${publish_cmd[@]}"
else
  refresh_cmd=()
  build_refresh_cmd "$STAGE" refresh_cmd
  run_refresh_cmd "Run historical refresh stage" "${refresh_cmd[@]}"
fi

printf '[%s] Historical refresh complete.\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"
