"""Resample M1 bid/ask bars to higher timeframes for multi-TF exploration.

Round 3 of the research loop: test every family on M5, M15, M30, H1 bars
instead of M1. Spread cost per trade becomes ~1/N as timeframe grows by N,
which dramatically changes the profitability profile for signal families
that were borderline at M1.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

Timeframe = Literal["1min", "5min", "15min", "30min", "1H", "4H"]


def resample_bars(bars: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
    """Resample M1 bid/ask OHLCV bars to a higher timeframe.

    Args:
        bars: M1 frame with ``bid_*`` / ``ask_*`` / ``volume`` columns,
            indexed by tz-aware UTC timestamp.
        timeframe: Pandas offset alias: ``"5min"``, ``"15min"``, ``"30min"``,
            ``"1H"``, ``"4H"``. ``"1min"`` returns the input unchanged.

    Returns:
        Resampled frame with the same column layout as the input.
    """
    if timeframe == "1min":
        return bars.copy()
    agg_spec: dict[str, str] = {}
    for prefix in ("bid_", "ask_", "mid_"):
        if f"{prefix}open" in bars.columns:
            agg_spec[f"{prefix}open"] = "first"
            agg_spec[f"{prefix}high"] = "max"
            agg_spec[f"{prefix}low"] = "min"
            agg_spec[f"{prefix}close"] = "last"
    if "volume" in bars.columns:
        agg_spec["volume"] = "sum"
    resampled = bars.resample(timeframe, label="left", closed="left").agg(agg_spec)
    resampled = resampled.dropna(how="any")
    # Compute mid_* if not present (needed by many families).
    if (
        "mid_close" not in resampled.columns
        and "bid_close" in resampled.columns
        and "ask_close" in resampled.columns
    ):
        resampled["mid_open"] = (resampled["bid_open"] + resampled["ask_open"]) / 2
        resampled["mid_high"] = (resampled["bid_high"] + resampled["ask_high"]) / 2
        resampled["mid_low"] = (resampled["bid_low"] + resampled["ask_low"]) / 2
        resampled["mid_close"] = (resampled["bid_close"] + resampled["ask_close"]) / 2
    return resampled
