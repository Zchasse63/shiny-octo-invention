"""Tests for the explorer orchestrator utilities (no vbt calls)."""

from __future__ import annotations

import pandas as pd

from src.backtest.explorer import (
    _iter_param_combos,
    _subsample,
    _walk_forward_slices,
)


def test_iter_param_combos_cartesian() -> None:
    grid = {"a": [1, 2], "b": ["x", "y"]}
    combos = list(_iter_param_combos(grid))
    assert len(combos) == 4
    assert {"a": 1, "b": "x"} in combos


def test_iter_param_combos_empty_grid() -> None:
    combos = list(_iter_param_combos({}))
    assert combos == [{}]


def test_subsample_caps_when_exceeded() -> None:
    items = list(range(100))
    out = _subsample(items, 10)
    assert len(out) == 10
    # Elements preserved, just fewer.
    assert all(i in items for i in out)


def test_subsample_passthrough_when_under_cap() -> None:
    items = list(range(5))
    out = _subsample(items, 10)
    assert out == items


def test_subsample_none_returns_all() -> None:
    items = list(range(5))
    assert _subsample(items, None) is items


def test_walk_forward_slices_single_split() -> None:
    idx = pd.date_range("2024-01-01", periods=1000, freq="1min", tz="UTC")
    close = pd.Series(range(1000), index=idx)
    splits = _walk_forward_slices(close, n_windows=0, train_frac=0.5)
    assert len(splits) == 1
    label, is_slice, oos_slice = splits[0]
    assert label == "full"
    assert is_slice.start == 0
    assert is_slice.stop == 500
    assert oos_slice.start == 500
    assert oos_slice.stop == 1000


def test_walk_forward_slices_multiple_windows() -> None:
    idx = pd.date_range("2024-01-01", periods=1200, freq="1min", tz="UTC")
    close = pd.Series(range(1200), index=idx)
    splits = _walk_forward_slices(close, n_windows=4, train_frac=0.5)
    assert len(splits) == 4
    # Each window is 300 bars, train is 150, test is 150.
    for _label, is_slice, oos_slice in splits:
        assert is_slice.stop - is_slice.start == 150
        assert oos_slice.stop - oos_slice.start == 150
