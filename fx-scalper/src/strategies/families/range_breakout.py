"""ATR-contraction range breakout.

Enter on break of N-bar high/low after an ATR contraction (quiet-before-
storm pattern). Classic momentum capture — small base-hit targets plus
trailing when volatility expansion has real momentum behind it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.indicators.engine import add_atr
from src.strategies.families.base_family import FamilySignals, SignalFamily


@dataclass(frozen=True, slots=True)
class RangeBreakoutParams:
    """Params for :class:`RangeBreakoutFamily`."""

    lookback: int = 20
    """How many bars to look back for range high/low."""
    atr_ratio_threshold: float = 0.75
    """Current ATR / ATR-of-last-N ≤ this → volatility has contracted."""
    atr_ratio_lookback: int = 50
    atr_length: int = 14


class RangeBreakoutFamily(SignalFamily):
    """Break of recent range after volatility contraction."""

    name = "range_breakout"
    params_cls = RangeBreakoutParams

    def __init__(self, params: RangeBreakoutParams | None = None) -> None:
        self._p = params or RangeBreakoutParams()

    def generate(self, candles: pd.DataFrame) -> FamilySignals:
        df = add_atr(candles, length=self._p.atr_length)
        close = _close(df)
        high = _series(df, "high")
        low = _series(df, "low")
        atr = df[f"atr_{self._p.atr_length}"]

        # Lookback range (prior bars only — exclude current to avoid leak).
        range_high = high.shift(1).rolling(self._p.lookback, min_periods=self._p.lookback).max()
        range_low = low.shift(1).rolling(self._p.lookback, min_periods=self._p.lookback).min()

        # Volatility contraction: current ATR < threshold × mean ATR over
        # the longer lookback.
        atr_mean = atr.rolling(
            self._p.atr_ratio_lookback, min_periods=self._p.atr_ratio_lookback
        ).mean()
        contracted = (atr / atr_mean) <= self._p.atr_ratio_threshold
        # Apply contraction flag from the bar BEFORE breakout.
        contracted_prev = contracted.shift(1).fillna(False).astype(bool)

        entries_long = (close > range_high) & contracted_prev
        entries_short = (close < range_low) & contracted_prev

        return FamilySignals(
            entries_long=entries_long.fillna(False),
            entries_short=entries_short.fillna(False),
        )

    def param_grid(self) -> dict[str, list[Any]]:
        return {
            "lookback": [10, 20, 40],
            "atr_ratio_threshold": [0.6, 0.75, 0.9],
            "atr_ratio_lookback": [30, 50, 100],
        }


def _close(df: pd.DataFrame) -> pd.Series:
    for col in ("mid_close", "bid_close", "close"):
        if col in df.columns:
            return df[col]
    raise KeyError("No close column")


def _series(df: pd.DataFrame, which: str) -> pd.Series:
    for col in (f"mid_{which}", f"bid_{which}", which):
        if col in df.columns:
            return df[col]
    raise KeyError(f"No {which} column")
