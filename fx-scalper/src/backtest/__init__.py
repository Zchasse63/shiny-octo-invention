"""Backtest harness (vectorbt Pro) + metrics + Dukascopy client."""

from __future__ import annotations

from src.backtest.data_loader import load_symbol_bars, resample_ticks_to_m1
from src.backtest.dukascopy_client import fetch_day_ticks, fetch_range
from src.backtest.metrics import compute_metrics

__all__ = [
    "compute_metrics",
    "fetch_day_ticks",
    "fetch_range",
    "load_symbol_bars",
    "resample_ticks_to_m1",
]
