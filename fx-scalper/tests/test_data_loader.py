"""Tests for backtest data loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.data_loader import (
    load_symbol_bars,
    resample_ticks_to_m1,
    save_symbol_bars,
)


def _make_bars(start: str, periods: int) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="1min", tz="UTC")
    rng = np.random.default_rng(0)
    px = 1.08 + rng.normal(0, 0.0002, periods)
    return pd.DataFrame(
        {
            "bid_open": px - 6e-5,
            "bid_high": px - 6e-5 + 2e-4,
            "bid_low": px - 6e-5 - 2e-4,
            "bid_close": px - 6e-5,
            "ask_open": px + 6e-5,
            "ask_high": px + 6e-5 + 2e-4,
            "ask_low": px + 6e-5 - 2e-4,
            "ask_close": px + 6e-5,
            "volume": 100,
        },
        index=idx,
    )


def test_save_then_load_roundtrip(tmp_path: Path) -> None:
    bars = _make_bars("2024-01-01", 60 * 24 * 3)  # 3 days of minute bars
    paths = save_symbol_bars(bars, symbol="EUR_USD", root=tmp_path)
    assert paths, "save_symbol_bars wrote no partitions"

    loaded = load_symbol_bars("EUR_USD", root=tmp_path)
    pd.testing.assert_frame_equal(
        loaded.sort_index(), bars.sort_index(), check_freq=False
    )


def test_save_partitions_by_month(tmp_path: Path) -> None:
    """A January+February input must produce two partitions."""
    bars_jan = _make_bars("2024-01-15", 60 * 24)  # 1 day in Jan
    bars_feb = _make_bars("2024-02-01", 60 * 24)  # 1 day in Feb
    bars = pd.concat([bars_jan, bars_feb])
    paths = save_symbol_bars(bars, symbol="EUR_USD", root=tmp_path)
    assert len(paths) == 2

    months = {str(p).split("month=")[-1].split("/")[0] for p in paths}
    assert months == {"01", "02"}


def test_save_on_sliced_frame_partitions_correctly(tmp_path: Path) -> None:
    """Regression: previous _group_by_year_month used iloc on positional
    indices which broke on filtered slices. This exercises that path.
    """
    full = _make_bars("2024-01-15", 60 * 24 * 45)  # ~Jan 15 → Feb 28
    # Slice to a sub-range (not starting from row 0) — ensures the
    # bars.iloc[...] positional indexing wouldn't have worked.
    sub = full[full.index >= pd.Timestamp("2024-02-01", tz="UTC")]
    paths = save_symbol_bars(sub, symbol="EUR_USD", root=tmp_path)
    assert paths
    loaded = load_symbol_bars("EUR_USD", root=tmp_path)
    assert loaded.index.min() >= pd.Timestamp("2024-02-01", tz="UTC")
    pd.testing.assert_frame_equal(
        loaded.sort_index(), sub.sort_index(), check_freq=False
    )


def test_resample_ticks_to_m1() -> None:
    idx = pd.date_range("2024-01-01 00:00", periods=120, freq="30s", tz="UTC")
    ticks = pd.DataFrame(
        {
            "bid": 1.08 + np.linspace(-0.0001, 0.0001, 120),
            "ask": 1.0801 + np.linspace(-0.0001, 0.0001, 120),
            "volume": 1,
        },
        index=idx,
    )
    m1 = resample_ticks_to_m1(ticks)
    # 60 minutes of data at 30s = 60 bars.
    assert len(m1) == 60
    for col in ("bid_open", "bid_high", "bid_low", "bid_close",
                "ask_open", "ask_high", "ask_low", "ask_close"):
        assert col in m1.columns


def test_resample_rejects_naive_ticks() -> None:
    import pytest

    idx = pd.date_range("2024-01-01", periods=10, freq="30s")  # naive
    ticks = pd.DataFrame({"bid": 1.08, "ask": 1.0801}, index=idx)
    with pytest.raises(ValueError):
        resample_ticks_to_m1(ticks)


def test_load_symbol_bars_filters_by_date_range(tmp_path: Path) -> None:
    bars = _make_bars("2024-01-01", 60 * 24 * 10)  # 10 days
    save_symbol_bars(bars, symbol="EUR_USD", root=tmp_path)
    sliced = load_symbol_bars(
        "EUR_USD",
        start="2024-01-05",
        end="2024-01-07",
        root=tmp_path,
    )
    assert sliced.index.min() >= pd.Timestamp("2024-01-05", tz="UTC")
    assert sliced.index.max() <= pd.Timestamp("2024-01-07 23:59", tz="UTC")
