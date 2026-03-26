"""Rolling-origin validation splits."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

import pandas as pd


def rolling_origin_splits(
    frame: pd.DataFrame,
    *,
    min_train_size: int,
    test_size: int,
    split_policy: Literal["row", "market_day", "target_day"] = "row",
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield chronological train/test splits."""

    sort_columns = [column for column in ["target_date", "decision_time_utc", "market_id", "decision_horizon"] if column in frame.columns]
    ordered = frame.sort_values(sort_columns).reset_index(drop=True)
    if split_policy == "row":
        for start in range(min_train_size, len(ordered), test_size):
            train = ordered.iloc[:start]
            test = ordered.iloc[start : start + test_size]
            if test.empty:
                break
            yield train.copy(), test.copy()
        return

    if split_policy == "market_day":
        key_frame = ordered[["market_id", "target_date"]].astype({"market_id": str}).copy()
    else:
        key_frame = ordered[["target_date"]].copy()

    group_ids = key_frame.astype(str).agg("|".join, axis=1)
    ordered = ordered.assign(_split_group_id=group_ids)
    unique_groups = ordered["_split_group_id"].drop_duplicates().tolist()
    for start in range(min_train_size, len(unique_groups), test_size):
        train_groups = set(unique_groups[:start])
        test_groups = set(unique_groups[start : start + test_size])
        if not test_groups:
            break
        train = ordered.loc[ordered["_split_group_id"].isin(train_groups)].drop(columns="_split_group_id")
        test = ordered.loc[ordered["_split_group_id"].isin(test_groups)].drop(columns="_split_group_id")
        if test.empty:
            break
        yield train.reset_index(drop=True).copy(), test.reset_index(drop=True).copy()
