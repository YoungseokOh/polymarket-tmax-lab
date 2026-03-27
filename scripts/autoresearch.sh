#!/usr/bin/env bash
# AutoResearch loop for det2prob_nn architecture search.
# Usage: bash scripts/autoresearch.sh [--variant VARIANT_NAME]
#
# Inspired by karpathy/autoresearch: modify the target file, run a fast eval,
# keep if the metric improves, revert otherwise. Results go to results.tsv.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

VARIANT="${1:-legacy_gaussian}"
LAST_N=100
RESULTS_FILE="results.tsv"
PRICING="real_history"

# Ensure results file has header
if [[ ! -f "${RESULTS_FILE}" ]]; then
    printf 'commit\tvariant\tavg_crps\tmae\tpnl\thit_rate\tstatus\tdescription\n' > "${RESULTS_FILE}"
fi

# ---- run eval for a given variant, return avg_crps ----
run_eval() {
    local variant="$1"
    uv run pmtmax backtest \
        --model-name det2prob_nn \
        --pricing-source "${PRICING}" \
        --variant "${variant}" \
        --last-n "${LAST_N}" \
        2>/dev/null
}

parse_metric() {
    local json="$1" key="$2"
    python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(d.get('${key}', 'null'))" <<< "${json}"
}

log_result() {
    local commit="$1" variant="$2" json="$3" status="$4" desc="$5"
    local crps mae pnl hit
    crps=$(parse_metric "${json}" avg_crps)
    mae=$(parse_metric "${json}" mae)
    pnl=$(parse_metric "${json}" pnl)
    hit=$(parse_metric "${json}" hit_rate)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${commit}" "${variant}" "${crps}" "${mae}" "${pnl}" "${hit}" "${status}" "${desc}" \
        >> "${RESULTS_FILE}"
    echo "  crps=${crps}  mae=${mae}  pnl=${pnl}  hit=${hit}  [${status}]"
}

echo "=== autoresearch: fast-eval on last ${LAST_N} groups ==="
echo "=== target variant: ${VARIANT} ==="
echo ""

# Run baseline
echo ">> Running baseline (legacy_gaussian)..."
baseline_json=$(run_eval "legacy_gaussian")
baseline_crps=$(parse_metric "${baseline_json}" avg_crps)
commit=$(git rev-parse --short HEAD)
log_result "${commit}" "legacy_gaussian" "${baseline_json}" "baseline" "baseline"
echo "Baseline CRPS: ${baseline_crps}"
echo ""

# Run the requested variant
echo ">> Running variant: ${VARIANT}..."
new_json=$(run_eval "${VARIANT}")
new_crps=$(parse_metric "${new_json}" avg_crps)
commit=$(git rev-parse --short HEAD)
log_result "${commit}" "${VARIANT}" "${new_json}" "evaluated" ""
echo "Variant CRPS: ${new_crps}"
echo ""

# Compare
improved=$(python3 -c "print('yes' if float('${new_crps}') < float('${baseline_crps}') else 'no')")
delta=$(python3 -c "print(f'{float(\"${baseline_crps}\") - float(\"${new_crps}\"):.4f}')")

if [[ "${improved}" == "yes" ]]; then
    echo "✓ IMPROVED by ${delta} CRPS  (${baseline_crps} → ${new_crps})"
    echo "  Keeping current state."
    # Update result status
    sed -i "s/${VARIANT}\t[^\t]*\t[^\t]*\t[^\t]*\t[^\t]*\tevaluated/${VARIANT}\t$(parse_metric "${new_json}" avg_crps)\t$(parse_metric "${new_json}" mae)\t$(parse_metric "${new_json}" pnl)\t$(parse_metric "${new_json}" hit_rate)\tkept/" "${RESULTS_FILE}" 2>/dev/null || true
else
    echo "✗ No improvement (delta=${delta}). Baseline: ${baseline_crps}, This: ${new_crps}"
fi

echo ""
echo "=== Results so far ==="
cat "${RESULTS_FILE}"
