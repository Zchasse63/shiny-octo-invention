"""Bollinger + RSI mean reversion (baseline carryover).

Kept as one of the exploratory families — NOT the primary strategy.
Long: close < lower BB and RSI < threshold.
Short: close > upper BB and RSI > threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.indicators.engine import add_bbands, add_rsi
from src.strategies.families.base_family import FamilySignals, SignalFamily


@dataclass(frozen=True, slots=True)
class BBRSIMRParams:
    """Params for :class:`BBRSIMRFamily`."""

    bb_length: int = 20
    bb_std: float = 2.0
    rsi_length: int = 14
    rsi_long_threshold: float = 30.0
    rsi_short_threshold: float = 70.0


class BBRSIMRFamily(SignalFamily):
    """Bollinger band + RSI extreme mean reversion."""

    name = "bb_rsi_mr"

    def __init__(self, params: BBRSIMRParams | None = None) -> None:
        self._p = params or BBRSIMRParams()

    def generate(self, candles: pd.DataFrame) -> FamilySignals:
        df = add_bbands(candles, length=self._p.bb_length, std=self._p.bb_std)
        df = add_rsi(df, length=self._p.rsi_length)

        close = _close(df)
        bb_lo = df[f"bb_lower_{self._p.bb_length}_{self._p.bb_std}"]
        bb_hi = df[f"bb_upper_{self._p.bb_length}_{self._p.bb_std}"]
        rsi = df[f"rsi_{self._p.rsi_length}"]

        entries_long = (close < bb_lo) & (rsi < self._p.rsi_long_threshold)
        entries_short = (close > bb_hi) & (rsi > self._p.rsi_short_threshold)

        return FamilySignals(
            entries_long=entries_long.fillna(False),
            entries_short=entries_short.fillna(False),
        )

    def param_grid(self) -> dict[str, list[Any]]:
        return {
            "bb_length": [10, 20, 30, 50],
            "bb_std": [1.5, 2.0, 2.5],
            "rsi_length": [7, 14, 21],
            "rsi_long_threshold": [20, 25, 30, 35],
            "rsi_short_threshold": [65, 70, 75, 80],
        }


def _close(df: pd.DataFrame) -> pd.Series:
    for col in ("mid_close", "bid_close", "close"):
        if col in df.columns:
            return df[col]
    raise KeyError("No close column")
