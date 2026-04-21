"""Tests for the common exit framework."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.exits import (
    ExitConfig,
    compute_initial_stops,
    compute_take_profits,
    config_to_vbt_params,
    enumerate_exit_configs,
)


def _simple_frame(n: int = 50) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=n, freq="5min", tz="UTC")
    close = np.full(n, 1.08) + np.linspace(-0.001, 0.001, n)
    atr = np.full(n, 0.0010)
    return pd.DataFrame(
        {"mid_close": close, "atr_14": atr},
        index=idx,
    )


def test_enumerate_exit_configs_returns_nonempty() -> None:
    configs = enumerate_exit_configs()
    assert len(configs) > 0
    # Sanity: default grid should include at least one "off" trail.
    assert any(c.trail_kind == "off" for c in configs)


def test_enumerate_exit_configs_skips_trail_mult_when_off() -> None:
    """When trail_kind="off", the trail_atr_mult axis shouldn't multiply combos."""
    with_off_only = enumerate_exit_configs(trail_kinds=("off",))
    # 4 sl × 5 tp × 1 trail = 20 combos; no trail_mult multiplication.
    assert len(with_off_only) == 4 * 5 * 1


def test_compute_initial_stops() -> None:
    frame = _simple_frame()
    entries_long = pd.Series(False, index=frame.index)
    entries_long.iloc[10] = True
    entries_short = pd.Series(False, index=frame.index)

    cfg = ExitConfig(sl_atr_mult=1.5, atr_length=14)
    result = compute_initial_stops(
        entries_long=entries_long,
        entries_short=entries_short,
        close=frame["mid_close"],
        atr=frame["atr_14"],
        config=cfg,
    )
    assert "sl_long" in result and "sl_short" in result
    # Long entry at bar 10: SL price = close - 1.5 × ATR.
    expected = frame["mid_close"].iloc[10] - 1.5 * frame["atr_14"].iloc[10]
    assert abs(result["sl_long"].iloc[10] - expected) < 1e-12
    # Non-entry bars should be NaN.
    assert pd.isna(result["sl_long"].iloc[0])


def test_compute_take_profits_with_no_tp() -> None:
    frame = _simple_frame()
    entries_long = pd.Series(False, index=frame.index)
    entries_long.iloc[5] = True
    entries_short = pd.Series(False, index=frame.index)

    cfg = ExitConfig(tp_r_mult=None)
    result = compute_take_profits(
        entries_long=entries_long,
        entries_short=entries_short,
        close=frame["mid_close"],
        atr=frame["atr_14"],
        config=cfg,
    )
    assert result["tp_long"].isna().all()


def test_compute_take_profits_with_tp_multiplier() -> None:
    frame = _simple_frame()
    entries_long = pd.Series(False, index=frame.index)
    entries_long.iloc[5] = True
    entries_short = pd.Series(False, index=frame.index)

    cfg = ExitConfig(sl_atr_mult=1.0, tp_r_mult=2.0)
    result = compute_take_profits(
        entries_long=entries_long,
        entries_short=entries_short,
        close=frame["mid_close"],
        atr=frame["atr_14"],
        config=cfg,
    )
    close_at_entry = frame["mid_close"].iloc[5]
    atr_at_entry = frame["atr_14"].iloc[5]
    expected_tp = close_at_entry + 2.0 * 1.0 * atr_at_entry
    assert abs(result["tp_long"].iloc[5] - expected_tp) < 1e-12


def test_config_to_vbt_params_emits_fractions() -> None:
    frame = _simple_frame()
    entries_long = pd.Series(False, index=frame.index)
    entries_long.iloc[10] = True
    entries_short = pd.Series(False, index=frame.index)

    cfg = ExitConfig(sl_atr_mult=1.0, tp_r_mult=1.0, trail_kind="atr_trail",
                     trail_atr_mult=2.0)
    vp = config_to_vbt_params(
        entries_long=entries_long,
        entries_short=entries_short,
        close=frame["mid_close"],
        atr=frame["atr_14"],
        config=cfg,
    )
    # SL fraction at entry bar = ATR / close ≈ 0.0010 / 1.08 ≈ 9.26e-4
    assert vp.sl_stop.iloc[10] > 0
    assert vp.tp_stop.iloc[10] > 0
    assert vp.sl_trail is True
    assert vp.trail_distance_pct is not None
    # Off-entry bars must be NaN.
    assert pd.isna(vp.sl_stop.iloc[0])


def test_config_to_vbt_params_trail_off() -> None:
    frame = _simple_frame()
    entries_long = pd.Series(False, index=frame.index)
    entries_long.iloc[5] = True
    entries_short = pd.Series(False, index=frame.index)

    cfg = ExitConfig(trail_kind="off")
    vp = config_to_vbt_params(
        entries_long=entries_long,
        entries_short=entries_short,
        close=frame["mid_close"],
        atr=frame["atr_14"],
        config=cfg,
    )
    assert vp.sl_trail is False
    assert vp.trail_distance_pct is None
