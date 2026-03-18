"""Rolling-origin validation splits."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd


def rolling_origin_splits(
    frame: pd.DataFrame,
    *,
    min_train_size: int,
    test_size: int,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield chronological train/test splits."""

    ordered = frame.sort_values("target_date").reset_index(drop=True)
    for start in range(min_train_size, len(ordered), test_size):
        train = ordered.iloc[:start]
        test = ordered.iloc[start : start + test_size]
        if test.empty:
            break
        yield train.copy(), test.copy()

