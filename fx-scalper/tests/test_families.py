"""Smoke tests for each signal family on synthetic data.

Verifies:
- Family instantiates with default params.
- generate() returns a FamilySignals with two boolean Series aligned to input.
- Every family exposes a non-empty param_grid().
- No stray NaN in the output (all False where conditions aren't met).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies.families import ALL_FAMILIES
from src.strategies.families.base_family import FamilySignals


@pytest.fixture
def synthetic_candles() -> pd.DataFrame:
    """500 M5 bars with varying price action: trend up, chop, trend down."""
    rng = np.random.default_rng(0)
    n = 500
    idx = pd.date_range("2024-01-02", periods=n, freq="5min", tz="UTC")
    base = 1.08
    # Three regimes: uptrend, chop, downtrend.
    first_third = np.linspace(0, 0.005, n // 3)
    mid_third = np.zeros(n // 3)
    last_third = np.linspace(0, -0.005, n - 2 * (n // 3))
    drift = np.concatenate([first_third, mid_third, last_third])
    noise = rng.normal(0, 0.0003, n)
    close = base + drift + noise.cumsum() * 0.1

    high = close + 0.0002
    low = close - 0.0002
    half_spread = 6e-5
    return pd.DataFrame(
        {
            "mid_open": close,
            "mid_high": high,
            "mid_low": low,
            "mid_close": close,
            "bid_close": close - half_spread,
            "ask_close": close + half_spread,
            "bid_high": high - half_spread,
            "bid_low": low - half_spread,
            "ask_high": high + half_spread,
            "ask_low": low + half_spread,
            "volume": rng.integers(50, 500, n),
            "complete": True,
        },
        index=idx,
    )


@pytest.mark.parametrize("family_cls", ALL_FAMILIES, ids=lambda c: c.name)
def test_family_generates_valid_signals(
    family_cls: type, synthetic_candles: pd.DataFrame
) -> None:
    family = family_cls()
    signals = family.generate(synthetic_candles)
    assert isinstance(signals, FamilySignals)
    # Both series must be boolean-dtype.
    assert signals.entries_long.dtype == bool
    assert signals.entries_short.dtype == bool
    # Aligned to input index.
    assert signals.entries_long.index.equals(synthetic_candles.index)
    assert signals.entries_short.index.equals(synthetic_candles.index)
    # No NaN — must be clean booleans.
    assert signals.entries_long.notna().all()
    assert signals.entries_short.notna().all()


@pytest.mark.parametrize("family_cls", ALL_FAMILIES, ids=lambda c: c.name)
def test_family_param_grid_nonempty(family_cls: type) -> None:
    family = family_cls()
    grid = family.param_grid()
    assert isinstance(grid, dict)
    assert len(grid) > 0
    for key, values in grid.items():
        assert isinstance(key, str)
        assert len(values) > 0


def test_families_can_produce_some_entries(
    synthetic_candles: pd.DataFrame,
) -> None:
    """At least one family should fire at least one entry on 500 synthetic bars.

    This isn't a correctness check — just a sanity that the families aren't
    all silently empty.
    """
    total_long = 0
    total_short = 0
    for family_cls in ALL_FAMILIES:
        family = family_cls()
        sig = family.generate(synthetic_candles)
        total_long += int(sig.entries_long.sum())
        total_short += int(sig.entries_short.sum())
    assert total_long + total_short > 0, (
        "All 6 families fired zero entries on 500 synthetic bars — "
        "suggests a systematic bug rather than a quiet market."
    )
