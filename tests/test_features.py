from __future__ import annotations

import numpy as np
import pandas as pd

from homecredit_service.features import filter_by_temporal_cutoff


def test_filter_by_temporal_cutoff_keeps_only_non_future_rows() -> None:
    df = pd.DataFrame(
        {
            "MONTHS_BALANCE": [-3, 0, 2, np.nan],
            "value": [1, 2, 3, 4],
        }
    )

    filtered = filter_by_temporal_cutoff(df, ("MONTHS_BALANCE",))

    assert filtered["value"].tolist() == [1, 2, 4]


def test_filter_by_temporal_cutoff_noop_when_columns_missing() -> None:
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

    filtered = filter_by_temporal_cutoff(df, ("DAYS_DECISION",))

    assert filtered.equals(df)
