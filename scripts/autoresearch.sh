#!/usr/bin/env bash
# Thin wrapper for the PM Tmax autoresearch CLI workflow.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ $# -lt 1 ]]; then
    cat >&2 <<'EOF'
Usage:
  bash scripts/autoresearch.sh init [args...]
  bash scripts/autoresearch.sh step --spec-path <candidate.yaml> [args...]
  bash scripts/autoresearch.sh gate --spec-path <candidate.yaml> [args...]
  bash scripts/autoresearch.sh analyze-paper --spec-path <candidate.yaml> [args...]
  bash scripts/autoresearch.sh promote --spec-path <candidate.yaml> [args...]
EOF
    exit 1
fi

command="$1"
shift

case "${command}" in
    init)
        uv run pmtmax autoresearch-init "$@"
        ;;
    step)
        uv run pmtmax autoresearch-step "$@"
        ;;
    gate)
        uv run pmtmax autoresearch-gate "$@"
        ;;
    analyze-paper)
        uv run pmtmax autoresearch-analyze-paper "$@"
        ;;
    promote)
        uv run pmtmax autoresearch-promote "$@"
        ;;
    *)
        echo "Unknown autoresearch command: ${command}" >&2
        exit 1
        ;;
esac
