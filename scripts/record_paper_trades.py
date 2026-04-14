"""Record new scan-edge signals as forward paper trades.

Reads scan_edge_latest.json and appends any signals not already present
in forward_paper_trades.json (deduped by city + target_date + outcome_label + direction).

Run:
    uv run python scripts/record_paper_trades.py

Called automatically by daily_experiment.sh after scan-edge.
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCAN_EDGE_PATH = REPO_ROOT / "artifacts/signals/v2/scan_edge_latest.json"
TRADES_PATH = REPO_ROOT / "artifacts/signals/v2/forward_paper_trades.json"

# Direction-specific gamma filters derived from paper-trade outcome analysis (2026-04-15):
#   YES bets: gamma [0.10,0.20) → 13% wr, [0.20,0.30) → 16% wr, [0.30,0.50) → 57% wr
#   NO bets (min): gamma < 0.50 risks losing (1 - entry) on plausible outcomes
#   NO bets (max): gamma > 0.75 → market highly confident in YES, loses 87.5% of the time
#     Analysis: NO wins avg gamma=0.576, NO losses avg gamma=0.892 (63 settled trades)
MIN_GAMMA_YES = 0.30
MAX_GAMMA_YES = 0.40  # YES bets: gamma [0.30,0.40) → 58.8% wr; gamma [0.40,0.60) → 0% wr (23 trades)
MIN_GAMMA_NO = 0.50
MAX_GAMMA_NO = 0.70  # NO bets: gamma ≤0.70 → 45.5% wr; gamma >0.70 → market always right (87.5% loss)
# Sweep analysis 2026-04-15 (63 settled trades)


def _load_signals() -> list[dict]:
    if not SCAN_EDGE_PATH.exists():
        print(f"[error] scan_edge_latest.json not found: {SCAN_EDGE_PATH}", file=sys.stderr)
        sys.exit(1)
    with SCAN_EDGE_PATH.open() as fh:
        return json.load(fh)


def _load_trades() -> tuple[dict, list[dict]]:
    if not TRADES_PATH.exists():
        return {"recorded_at": None, "trades": []}, []
    with TRADES_PATH.open() as fh:
        data = json.load(fh)
    return data, data.get("trades", [])


def _dedup_key(trade: dict) -> tuple:
    return (
        trade.get("city", ""),
        trade.get("target_date", ""),
        trade.get("outcome_label", ""),
        trade.get("direction", ""),
    )


def main() -> None:
    signals = _load_signals()
    data, existing_trades = _load_trades()

    existing_keys = {_dedup_key(t) for t in existing_trades}

    now_str = datetime.now(tz=UTC).isoformat()
    new_trades = []

    skipped_low_price = 0
    for sig in signals:
        gamma_price = sig.get("gamma_price")
        direction = sig.get("direction", "yes").lower()
        if gamma_price is not None:
            gp = float(gamma_price)
            min_threshold = MIN_GAMMA_YES if direction == "yes" else MIN_GAMMA_NO
            if gp < min_threshold:
                skipped_low_price += 1
                continue
            if direction == "yes" and gp > MAX_GAMMA_YES:
                skipped_low_price += 1
                continue
            if direction == "no" and gp > MAX_GAMMA_NO:
                skipped_low_price += 1
                continue
        trade = {
            "recorded_at": now_str,
            "city": sig.get("city", ""),
            "target_date": sig.get("date", ""),
            "outcome_label": sig.get("bin", ""),
            "direction": sig.get("direction", "yes"),
            "model_prob": sig.get("model_prob"),
            "gamma_price": sig.get("gamma_price"),
            "edge_after_fee": sig.get("best_edge"),
            "horizon": sig.get("horizon"),
            "outcome": None,
            "realized_pnl": None,
            "latest_price": None,
            "price_source": None,
        }
        key = _dedup_key(trade)
        if key in existing_keys:
            continue
        new_trades.append(trade)
        existing_keys.add(key)

    if skipped_low_price:
        print(f"[record_paper_trades] skipped {skipped_low_price} signals (YES<{MIN_GAMMA_YES:.0%} or NO<{MIN_GAMMA_NO:.0%})")

    if not new_trades:
        print(f"[record_paper_trades] no new signals to record (existing: {len(existing_trades)})")
        return

    all_trades = existing_trades + new_trades
    updated = {
        "recorded_at": now_str,
        "trades": all_trades,
    }
    with TRADES_PATH.open("w") as fh:
        json.dump(updated, fh, indent=2)

    print(
        f"[record_paper_trades] added {len(new_trades)} new trades "
        f"(total: {len(all_trades)}, existing: {len(existing_trades)})"
    )
    for t in new_trades:
        print(
            f"  + {t['city']:<16} {t['target_date']:<12} {t['outcome_label']:<20} "
            f"{t['direction']:<4} gamma={t['gamma_price']:.3f}  edge={t['edge_after_fee']:.4f}"
        )


if __name__ == "__main__":
    main()
