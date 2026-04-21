"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils.journal import Journal


@pytest.fixture
def tmp_journal(tmp_path: Path) -> Iterator[Journal]:
    """Fresh SQLite journal in a temp dir."""
    db = tmp_path / "journal.db"
    yield Journal(db)


@pytest.fixture
def synthetic_m5_candles() -> pd.DataFrame:
    """Generate 500 M5 bars with mean-reverting noise around 1.0800.

    Bid/ask columns for spread-aware code paths.
    """
    rng = np.random.default_rng(42)
    n = 500
    start = datetime(2024, 1, 2, 0, 0, tzinfo=UTC)
    index = pd.date_range(start=start, periods=n, freq="5min")

    # AR(1) process around 1.0800 with small tenth-of-pip noise.
    mid = np.empty(n)
    mid[0] = 1.0800
    for i in range(1, n):
        mid[i] = 0.9998 * mid[i - 1] + 0.0002 * 1.0800 + rng.normal(0, 0.0005)

    half_spread = 0.00006  # 0.6 pip half-spread on EUR/USD
    bid = mid - half_spread
    ask = mid + half_spread

    df = pd.DataFrame(
        {
            "mid_open": mid,
            "mid_high": mid + 0.0002,
            "mid_low": mid - 0.0002,
            "mid_close": mid,
            "bid_open": bid,
            "bid_high": bid + 0.0002,
            "bid_low": bid - 0.0002,
            "bid_close": bid,
            "ask_open": ask,
            "ask_high": ask + 0.0002,
            "ask_low": ask - 0.0002,
            "ask_close": ask,
            "volume": rng.integers(50, 500, size=n),
            "complete": True,
        },
        index=index,
    )
    return df


@pytest.fixture
def minute_equity_returns() -> pd.Series:
    """500 minutes of synthetic returns with a modestly positive mean."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-02", periods=500, freq="1min", tz="UTC")
    return pd.Series(rng.normal(0.00001, 0.0005, size=500), index=idx)
