from __future__ import annotations

import pandas as pd

from pmtmax.backtest.rolling_origin import rolling_origin_splits


def test_market_day_split_keeps_same_market_date_group_together() -> None:
    rows: list[dict[str, object]] = []
    for group_idx, target_date in enumerate(["2026-01-01", "2026-01-02", "2026-01-03"], start=1):
        for horizon_idx, horizon in enumerate(["previous_evening", "morning_of"]):
            rows.append(
                {
                    "market_id": f"m{group_idx}",
                    "target_date": pd.Timestamp(target_date),
                    "decision_time_utc": pd.Timestamp(f"{target_date} {6 + horizon_idx}:00:00", tz="UTC"),
                    "decision_horizon": horizon,
                    "value": float(group_idx),
                }
            )
    frame = pd.DataFrame(rows)

    splits = list(
        rolling_origin_splits(
            frame,
            min_train_size=1,
            test_size=1,
            split_policy="market_day",
        )
    )

    assert len(splits) == 2
    for train, test in splits:
        train_groups = set(train[["market_id", "target_date"]].astype(str).agg("|".join, axis=1))
        test_groups = set(test[["market_id", "target_date"]].astype(str).agg("|".join, axis=1))
        assert train_groups.isdisjoint(test_groups)
        assert len(test) == 2
