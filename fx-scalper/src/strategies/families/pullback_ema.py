"""Pullback-to-EMA family.

Enter LONG when price pulls back to fast EMA while slow EMA is sloping up.
Enter SHORT on rallies to fast EMA while slow EMA is sloping down.

Captures trend-continuation base hits — price dips, we buy; price rallies
in downtrend, we sell. Trail handles the "occasional homerun" when the
continuation has legs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.indicators.engine import add_atr, add_ema
from src.strategies.families.base_family import FamilySignals, SignalFamily


@dataclass(frozen=True, slots=True)
class PullbackEMAParams:
    """Params for :class:`PullbackEMAFamily`."""

    fast_ema: int = 20
    slow_ema: int = 50
    slope_lookback: int = 10
    """Bars used to measure slow EMA slope direction."""
    touch_atr_frac: float = 0.25
    """How close to fast EMA counts as a pullback, in ATR fractions."""
    atr_length: int = 14


class PullbackEMAFamily(SignalFamily):
    """Pullback to fast EMA in the direction of slow EMA slope."""

    name = "pullback_ema"

    def __init__(self, params: PullbackEMAParams | None = None) -> None:
        self._p = params or PullbackEMAParams()

    def generate(self, candles: pd.DataFrame) -> FamilySignals:
        df = add_ema(candles, length=self._p.fast_ema)
        df = add_ema(df, length=self._p.slow_ema)
        df = add_atr(df, length=self._p.atr_length)

        fast = df[f"ema_{self._p.fast_ema}"]
        slow = df[f"ema_{self._p.slow_ema}"]
        atr = df[f"atr_{self._p.atr_length}"]
        close = _close(df)

        # Slope of slow EMA: positive if now > N bars ago.
        slow_slope_up = slow > slow.shift(self._p.slope_lookback)
        slow_slope_down = slow < slow.shift(self._p.slope_lookback)

        # Pullback = price close to fast EMA, in ATR units.
        touch_dist = self._p.touch_atr_frac * atr
        near_fast = (close - fast).abs() <= touch_dist

        entries_long = near_fast & slow_slope_up & (close > slow)
        entries_short = near_fast & slow_slope_down & (close < slow)

        return FamilySignals(
            entries_long=entries_long.fillna(False),
            entries_short=entries_short.fillna(False),
        )

    def param_grid(self) -> dict[str, list[Any]]:
        return {
            "fast_ema": [10, 20, 34],
            "slow_ema": [50, 100, 200],
            "slope_lookback": [5, 10, 20],
            "touch_atr_frac": [0.15, 0.25, 0.5],
        }


def _close(df: pd.DataFrame) -> pd.Series:
    for col in ("mid_close", "bid_close", "close"):
        if col in df.columns:
            return df[col]
    raise KeyError("No close column on DataFrame")
