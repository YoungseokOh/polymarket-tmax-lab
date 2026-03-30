"""Track outcomes for forward paper trades recorded in forward_paper_trades.json.

For each trade, checks whether the market has settled by consulting:
  1. data/parquet/silver/silver_price_timeseries.parquet (historical settlement prices)
  2. artifacts/signals/v2/gamma_price_log.jsonl (daily Gamma price snapshots)

A market outcome is considered settled when the latest observed price for that
(city, target_date, outcome_label) tuple is:
  - >= WINNER_THRESHOLD (0.95)  → the outcome won
  - <= LOSER_THRESHOLD (0.02)   → the outcome lost

PnL per-contract (unit size = 1):
  direction=YES, won  : +( 1 - entry_price )  [bought YES cheap, pays 1]
  direction=YES, lost : -(   entry_price    )  [bought YES, pays 0]
  direction=NO,  won  : +(   entry_price    )  [bought NO cheap = sold YES high → pays 0 on YES side]
  direction=NO,  lost : -(1 - entry_price   )  [bought NO, pays 0 on NO side]

Run:
    uv run python scripts/track_paper_trade_outcomes.py
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

TRADES_PATH = REPO_ROOT / "artifacts/signals/v2/forward_paper_trades.json"
GAMMA_LOG_PATH = REPO_ROOT / "artifacts/signals/v2/gamma_price_log.jsonl"
SILVER_PARQUET_PATH = REPO_ROOT / "data/parquet/silver/silver_price_timeseries.parquet"

WINNER_THRESHOLD = 0.95
LOSER_THRESHOLD = 0.02


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_trades() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return (metadata_dict, list_of_trades)."""
    if not TRADES_PATH.exists():
        print(f"[error] Trades file not found: {TRADES_PATH}", file=sys.stderr)
        sys.exit(1)
    with TRADES_PATH.open() as fh:
        data = json.load(fh)
    trades = data.get("trades", [])
    return data, trades


def _load_gamma_latest_prices() -> dict[tuple[str, str, str], float]:
    """Return mapping (city, target_date, outcome_label) -> latest gamma_price."""
    if not GAMMA_LOG_PATH.exists():
        return {}

    # Keep only the latest observation per (city, target_date, outcome_label)
    latest: dict[tuple[str, str, str], tuple[str, float]] = {}
    with GAMMA_LOG_PATH.open() as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            key = (
                str(rec.get("city", "")),
                str(rec.get("target_date", "")),
                str(rec.get("outcome_label", "")),
            )
            observed_at = str(rec.get("observed_at", ""))
            price = rec.get("gamma_price")
            if price is None:
                continue
            try:
                price = float(price)
            except (TypeError, ValueError):
                continue
            prev = latest.get(key)
            if prev is None or observed_at >= prev[0]:
                latest[key] = (observed_at, price)

    return {k: v[1] for k, v in latest.items()}


def _load_silver_latest_prices() -> dict[tuple[str, str, str], float]:
    """Return mapping (city, target_date_str, outcome_label) -> latest price."""
    if not SILVER_PARQUET_PATH.exists():
        print(
            f"[warn] Silver parquet not found: {SILVER_PARQUET_PATH}",
            file=sys.stderr,
        )
        return {}

    try:
        import pandas as pd
        import pyarrow.parquet as pq
    except ImportError:
        print("[warn] pyarrow/pandas not available; skipping silver parquet.", file=sys.stderr)
        return {}

    try:
        df = pq.read_table(
            SILVER_PARQUET_PATH,
            columns=["city", "target_local_date", "outcome_label", "price", "timestamp"],
        ).to_pandas()
    except Exception as exc:
        print(f"[warn] Could not read silver parquet: {exc}", file=sys.stderr)
        return {}

    df["target_date_str"] = pd.to_datetime(df["target_local_date"]).dt.strftime("%Y-%m-%d")
    # Latest price per (city, target_date, outcome_label)
    df_sorted = df.sort_values("timestamp")
    latest = df_sorted.groupby(["city", "target_date_str", "outcome_label"])["price"].last()
    return {(city, date_str, label): price for (city, date_str, label), price in latest.items()}


def _classify_outcome(latest_price: float | None) -> str | None:
    """Return 'won', 'lost', or None if unsettled."""
    if latest_price is None:
        return None
    if latest_price >= WINNER_THRESHOLD:
        return "won"
    if latest_price <= LOSER_THRESHOLD:
        return "lost"
    return None


def _compute_pnl(direction: str, entry_price: float, outcome: str) -> float:
    """Compute realized PnL for a unit-size paper trade.

    direction=YES: we bought YES at entry_price.
      - won  → collect 1.00, paid entry_price  → PnL = 1 - entry_price
      - lost → collect 0.00, paid entry_price  → PnL = -entry_price

    direction=NO: we bought NO at (1 - entry_price)  [entry_price is Gamma mid for YES].
      - won  → market resolves YES, NO pays 0 → PnL = -(1 - entry_price)
      - lost → market resolves NO,  NO pays 1 → PnL = entry_price
    """
    d = direction.lower()
    if d == "yes":
        return round((1.0 - entry_price) if outcome == "won" else -entry_price, 6)
    else:  # "no"
        return round(entry_price if outcome == "lost" else -(1.0 - entry_price), 6)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _, trades = _load_trades()
    total = len(trades)

    print(f"[track_paper_trade_outcomes] loaded {total} trades from {TRADES_PATH}")

    # Build lookup tables
    print("  loading gamma price log...")
    gamma_prices = _load_gamma_latest_prices()
    print(f"  loaded {len(gamma_prices)} unique (city, date, label) gamma prices")

    print("  loading silver price timeseries...")
    silver_prices = _load_silver_latest_prices()
    print(f"  loaded {len(silver_prices)} unique (city, date, label) silver prices")

    # Evaluate each trade
    settled_count = 0
    pending_count = 0
    wins = 0
    losses = 0
    total_pnl = 0.0

    results: list[dict[str, Any]] = []

    for trade in trades:
        city = str(trade.get("city", ""))
        target_date = str(trade.get("target_date", ""))
        outcome_label = str(trade.get("outcome_label", ""))
        direction = str(trade.get("direction", "yes")).lower()
        entry_price = float(trade.get("gamma_price", 0.5))

        key = (city, target_date, outcome_label)

        # Prefer silver parquet (authoritative settlement), fall back to gamma log
        latest_price: float | None = silver_prices.get(key)
        price_source = "silver" if latest_price is not None else None
        if latest_price is None:
            latest_price = gamma_prices.get(key)
            if latest_price is not None:
                price_source = "gamma_log"

        outcome = _classify_outcome(latest_price)

        if outcome is not None:
            pnl = _compute_pnl(direction, entry_price, outcome)
            settled_count += 1
            total_pnl += pnl
            if (direction == "yes" and outcome == "won") or (direction == "no" and outcome == "lost"):
                wins += 1
            else:
                losses += 1

            result = {
                **trade,
                "outcome": outcome,
                "realized_pnl": pnl,
                "latest_price": latest_price,
                "price_source": price_source,
            }
        else:
            pending_count += 1
            result = {
                **trade,
                "outcome": None,
                "realized_pnl": None,
                "latest_price": latest_price,
                "price_source": price_source,
            }

        results.append(result)

    # Print summary
    print()
    print("=" * 60)
    print("  PAPER TRADE OUTCOME SUMMARY")
    print("=" * 60)
    print(f"  Total forward trades : {total}")
    print(f"  Settled              : {settled_count}")
    print(f"  Pending              : {pending_count}")
    if settled_count > 0:
        hit_rate = wins / settled_count
        avg_pnl = total_pnl / settled_count
        print(f"  Wins                 : {wins}")
        print(f"  Losses               : {losses}")
        print(f"  Hit rate             : {hit_rate:.1%}")
        print(f"  Total PnL (settled)  : {total_pnl:+.4f}")
        print(f"  Avg PnL per trade    : {avg_pnl:+.4f}")
    else:
        print("  (no settled trades yet)")
    print("=" * 60)

    # Print per-trade settled breakdown
    settled_results = [r for r in results if r["outcome"] is not None]
    if settled_results:
        print()
        print("  Settled trade details:")
        header = f"  {'City':<16} {'Date':<12} {'Label':<20} {'Dir':<4} {'Entry':>6} {'Latest':>7} {'Outcome':<6} {'PnL':>8}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for r in settled_results:
            print(
                f"  {r['city']:<16} {r['target_date']:<12} {r['outcome_label']:<20} "
                f"{r['direction']:<4} {r['gamma_price']:>6.3f} {r['latest_price']:>7.4f} "
                f"{r['outcome']:<6} {r['realized_pnl']:>+8.4f}"
            )

    # Write updated trades back to file
    now_str = datetime.now(tz=UTC).isoformat()
    updated_data = {
        "recorded_at": now_str,
        "trades": results,
    }
    with TRADES_PATH.open("w") as fh:
        json.dump(updated_data, fh, indent=2)
    print()
    print(f"  Updated trades written to {TRADES_PATH}")


if __name__ == "__main__":
    main()
