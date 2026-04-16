"""Record ALL scan-edge signals (unfiltered) to a persistent append-only log.

Unlike record_paper_trades.py (which applies gamma filters), this records every
signal for model direction accuracy analysis — including signals outside our
trading filter. Used to:
  - Compute direction accuracy by gamma range, edge size, horizon, city
  - Find optimal filter thresholds from data rather than gut feel
  - Measure model accuracy independently of trading decisions

Output root follows `PMTMAX_ARTIFACTS_DIR` (default: repo `artifacts/`):
        signals/v2/all_signals_log.jsonl   (append-only)
        signals/v2/all_signals_latest.json (latest snapshot)

Usage:
    uv run python scripts/record_all_signals.py
    uv run python scripts/record_all_signals.py --analyze
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_ROOT = Path(os.environ.get("PMTMAX_ARTIFACTS_DIR", str(REPO_ROOT / "artifacts")))
SCAN_EDGE_PATH = ARTIFACTS_ROOT / "signals" / "v2" / "scan_edge_latest.json"
LOG_PATH = ARTIFACTS_ROOT / "signals" / "v2" / "all_signals_log.jsonl"
LATEST_PATH = ARTIFACTS_ROOT / "signals" / "v2" / "all_signals_latest.json"


def _dedup_key(sig: dict) -> str:
    return f"{sig.get('city','')}__{sig.get('date','')}__{sig.get('bin','')}__{sig.get('direction','')}"


def record() -> None:
    if not SCAN_EDGE_PATH.exists():
        print(f"[error] scan_edge_latest.json not found: {SCAN_EDGE_PATH}", file=sys.stderr)
        sys.exit(1)

    signals = json.loads(SCAN_EDGE_PATH.read_text())

    # Load existing keys from log to dedup
    existing_keys: set[str] = set()
    if LOG_PATH.exists():
        for line in LOG_PATH.read_text().splitlines():
            if line.strip():
                try:
                    existing_keys.add(_dedup_key(json.loads(line)))
                except Exception:
                    pass

    now_str = datetime.now(tz=UTC).isoformat()
    new_entries = []
    for sig in signals:
        entry = {
            "recorded_at": now_str,
            "city": sig.get("city", ""),
            "date": sig.get("date", ""),
            "bin": sig.get("bin", ""),
            "direction": sig.get("direction", "yes"),
            "model_prob": sig.get("model_prob"),
            "gamma_price": sig.get("gamma_price"),
            "edge": sig.get("best_edge"),
            "horizon": sig.get("horizon"),
            # Settlement fields — filled in later by settle_all_signals.py
            "outcome": None,      # "won" / "lost" (YES-token-centric, same convention)
        }
        key = _dedup_key(entry)
        if key in existing_keys:
            continue
        new_entries.append(entry)
        existing_keys.add(key)

    if not new_entries:
        print(f"[record_all_signals] no new signals (log has {len(existing_keys)} entries)")
        return

    with LOG_PATH.open("a") as fh:
        for e in new_entries:
            fh.write(json.dumps(e) + "\n")

    # Write latest snapshot for quick inspection
    LATEST_PATH.write_text(json.dumps(new_entries, indent=2))

    print(f"[record_all_signals] appended {len(new_entries)} signals → {LOG_PATH}")


def analyze() -> None:
    if not LOG_PATH.exists():
        print("[analyze] no log found", file=sys.stderr)
        return

    entries = []
    for line in LOG_PATH.read_text().splitlines():
        if line.strip():
            try:
                entries.append(json.loads(line))
            except Exception:
                pass

    settled = [e for e in entries if e.get("outcome") in ("won", "lost")]
    print(f"\nTotal logged: {len(entries)} | Settled: {len(settled)}")
    if not settled:
        print("No settled signals yet.")
        return

    def is_correct(e: dict) -> bool:
        d = e.get("direction", "yes").lower()
        o = e.get("outcome", "")
        return (d == "yes" and o == "won") or (d == "no" and o == "lost")

    print(f"\n=== Overall Direction Accuracy ===")
    correct = [e for e in settled if is_correct(e)]
    print(f"  {len(correct)}/{len(settled)} = {len(correct)/len(settled):.1%}")

    # By gamma range
    print(f"\n=== By Gamma Range (gamma_price of YES token) ===")
    buckets = [(0.0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
               (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]
    for lo, hi in buckets:
        bucket = [e for e in settled if e.get("gamma_price") is not None
                  and lo <= e["gamma_price"] < hi]
        if not bucket:
            continue
        c = sum(1 for e in bucket if is_correct(e))
        print(f"  gamma [{lo:.1f},{hi:.1f}): {c}/{len(bucket)} = {c/len(bucket):.1%}")

    # By edge size
    print(f"\n=== By Edge Size ===")
    for lo, hi in [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 1.0)]:
        bucket = [e for e in settled if e.get("edge") is not None
                  and lo <= abs(e["edge"]) < hi]
        if not bucket:
            continue
        c = sum(1 for e in bucket if is_correct(e))
        print(f"  edge [{lo:.1f},{hi:.1f}): {c}/{len(bucket)} = {c/len(bucket):.1%}")

    # By horizon
    print(f"\n=== By Horizon ===")
    horizons = {e.get("horizon", "unknown") for e in settled}
    for h in sorted(horizons):
        bucket = [e for e in settled if e.get("horizon") == h]
        c = sum(1 for e in bucket if is_correct(e))
        print(f"  {h}: {c}/{len(bucket)} = {c/len(bucket):.1%}")

    # By city (top 10 by volume)
    print(f"\n=== By City (min 3 settled) ===")
    cities = {}
    for e in settled:
        city = e.get("city", "unknown")
        cities.setdefault(city, []).append(e)
    city_rows = []
    for city, ces in cities.items():
        if len(ces) < 3:
            continue
        c = sum(1 for e in ces if is_correct(e))
        city_rows.append((c / len(ces), c, len(ces), city))
    for acc, c, n, city in sorted(city_rows, reverse=True):
        print(f"  {city:<16} {c}/{n} = {acc:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze", action="store_true", help="Print accuracy breakdown")
    args = parser.parse_args()

    if args.analyze:
        analyze()
    else:
        record()


if __name__ == "__main__":
    main()
