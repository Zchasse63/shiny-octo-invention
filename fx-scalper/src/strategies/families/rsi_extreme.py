"""RSI extreme oscillator.

Enter on RSI crossing out of oversold/overbought territory — a simple
oscillator family, different from BB-RSI MR in that it doesn't require
price to be outside Bollinger bands; just RSI's own signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.indicators.engine import add_rsi
from src.strategies.families.base_family import FamilySignals, SignalFamily


@dataclass(frozen=True, slots=True)
class RSIExtremeParams:
    """Params for :class:`RSIExtremeFamily`."""

    rsi_length: int = 14
    oversold: float = 30.0
    overbought: float = 70.0


class RSIExtremeFamily(SignalFamily):
    """RSI oversold-cross-up (long) / overbought-cross-down (short)."""

    name = "rsi_extreme"
    params_cls = RSIExtremeParams

    def __init__(self, params: RSIExtremeParams | None = None) -> None:
        self._p = params or RSIExtremeParams()

    def generate(self, candles: pd.DataFrame) -> FamilySignals:
        df = add_rsi(candles, length=self._p.rsi_length)
        rsi = df[f"rsi_{self._p.rsi_length}"]
        rsi_prev = rsi.shift(1)

        # Long: RSI was oversold, now rising above oversold threshold.
        entries_long = (rsi_prev < self._p.oversold) & (rsi >= self._p.oversold)
        # Short: RSI was overbought, now dropping below overbought threshold.
        entries_short = (rsi_prev > self._p.overbought) & (rsi <= self._p.overbought)

        return FamilySignals(
            entries_long=entries_long.fillna(False),
            entries_short=entries_short.fillna(False),
        )

    def param_grid(self) -> dict[str, list[Any]]:
        return {
            "rsi_length": [7, 10, 14, 21],
            "oversold": [20, 25, 30, 35],
            "overbought": [65, 70, 75, 80],
        }
