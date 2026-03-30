"""Extract Gamma mid-prices from discovered_markets.json and append to price log.

Run after scan-markets to build a daily timeseries of Gamma prices for all
open markets. Used to expand the backtest panel price coverage beyond the
CLOB-based silver_price_timeseries.
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    markets_path = Path("artifacts/discovered_markets.json")
    log_path = Path("artifacts/signals/v2/gamma_price_log.jsonl")

    if not markets_path.exists():
        print(f"[error] {markets_path} not found — run scan-markets first", file=sys.stderr)
        sys.exit(1)

    with markets_path.open() as fh:
        snapshots = json.load(fh)

    observed_at = datetime.now(tz=UTC).isoformat()
    records: list[dict[str, object]] = []

    for snap in snapshots:
        market = snap.get("market", {})
        spec = snap.get("spec")
        outcome_prices = snap.get("outcome_prices") or {}

        market_id = str(market.get("id") or market.get("market_id") or "")
        if not market_id:
            continue

        city = (spec or {}).get("city") or market.get("city") or ""
        question = (spec or {}).get("question") or market.get("question") or ""
        target_date = str((spec or {}).get("target_local_date") or "")

        for outcome_label, price in outcome_prices.items():
            if price is None:
                continue
            try:
                price_float = float(price)
            except (TypeError, ValueError):
                continue
            if price_float <= 0.0 or price_float >= 1.0:
                continue

            records.append({
                "observed_at": observed_at,
                "market_id": market_id,
                "city": city,
                "question": question,
                "target_date": target_date,
                "outcome_label": outcome_label,
                "gamma_price": round(price_float, 6),
            })

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    print(f"[log_gamma_prices] {len(records)} prices logged → {log_path}")


if __name__ == "__main__":
    main()
