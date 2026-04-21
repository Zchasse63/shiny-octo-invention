"""Fast/slow EMA crossover.

Classic momentum: when fast EMA crosses above slow EMA go LONG; reverse
for shorts. Simple baseline — useful as a reality check for
more-complex families.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.indicators.engine import add_ema
from src.strategies.families.base_family import FamilySignals, SignalFamily


@dataclass(frozen=True, slots=True)
class EMACrossParams:
    """Params for :class:`EMACrossFamily`."""

    fast_ema: int = 9
    slow_ema: int = 21


class EMACrossFamily(SignalFamily):
    """Momentum crossover of two EMAs."""

    name = "ema_cross"
    params_cls = EMACrossParams

    def __init__(self, params: EMACrossParams | None = None) -> None:
        self._p = params or EMACrossParams()

    def generate(self, candles: pd.DataFrame) -> FamilySignals:
        df = add_ema(candles, length=self._p.fast_ema)
        df = add_ema(df, length=self._p.slow_ema)

        fast = df[f"ema_{self._p.fast_ema}"]
        slow = df[f"ema_{self._p.slow_ema}"]

        # Crossover: fast > slow this bar AND fast <= slow previous bar.
        above = (fast > slow).astype(bool)
        above_prev = above.shift(1).fillna(False).astype(bool)
        crossed_up = above & (~above_prev)
        crossed_down = (~above) & above_prev

        return FamilySignals(
            entries_long=crossed_up.fillna(False),
            entries_short=crossed_down.fillna(False),
        )

    def param_grid(self) -> dict[str, list[Any]]:
        return {
            "fast_ema": [5, 9, 13, 21],
            "slow_ema": [21, 34, 55, 89],
        }

    def param_filter(self, params: dict[str, Any]) -> bool:
        """Require fast_ema strictly less than slow_ema."""
        return params["fast_ema"] < params["slow_ema"]
