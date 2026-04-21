"""Strategy 2: Trend-filtered M15 momentum — scaffold for Day 5.

Per CLAUDE.md §Starter Strategies #2:

* Pair: GBP/USD primary
* Filter: price vs EMA200 on H1
* Signal: RSI(14) crosses 50 in trend direction, ADX(14) > 25
* SL: 1.5 × ATR, TP: 2.5 × ATR
* Trail: Chandelier exit (highest_since_entry − 3 × ATR for longs)

This module is implemented Day 5 — kept as a stub now so the package layout
is complete and imports resolve.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.strategies.base import Signal, Strategy


@dataclass(frozen=True, slots=True)
class TrendMomentumParams:
    """Placeholder; filled in Day 5."""

    ema_length: int = 200
    rsi_length: int = 14
    adx_length: int = 14
    adx_threshold: float = 25.0
    sl_atr_multiplier: float = 1.5
    tp_atr_multiplier: float = 2.5
    chandelier_atr_multiplier: float = 3.0


class TrendMomentum(Strategy):
    """Day 5 — not yet implemented. Raises :class:`NotImplementedError`."""

    NAME = "trend_momentum"

    def __init__(self, params: TrendMomentumParams | None = None) -> None:
        self._p = params or TrendMomentumParams()

    @property
    def name(self) -> str:
        return self.NAME

    def generate_signal(
        self,
        *,
        instrument: str,
        candles: pd.DataFrame,
    ) -> Signal | None:
        raise NotImplementedError("Strategy 2 implemented in Day 5.")
