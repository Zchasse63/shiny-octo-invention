"""Session-anchored VWAP deviation reversion.

Compute session VWAP over rolling N-bar window (a proxy for session-anchored
VWAP since forex doesn't have a single daily session). Enter COUNTER to
deviation: price Nσ above VWAP → short, Nσ below → long.

Matches classic intraday mean-reversion — price tends to pull back to
VWAP in ranging conditions. Trail catches the occasional overshoot reverse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.strategies.families.base_family import FamilySignals, SignalFamily


@dataclass(frozen=True, slots=True)
class VWAPDeviationParams:
    """Params for :class:`VWAPDeviationFamily`."""

    vwap_window: int = 60
    """Bars used for rolling VWAP (on M5 this is 5 hours)."""
    sigma_window: int = 60
    """Bars used for rolling stddev of (close - vwap)."""
    entry_sigma: float = 2.0
    """Deviation threshold in rolling-stddev units."""


class VWAPDeviationFamily(SignalFamily):
    """Rolling VWAP deviation fade."""

    name = "vwap_deviation"

    def __init__(self, params: VWAPDeviationParams | None = None) -> None:
        self._p = params or VWAPDeviationParams()

    def generate(self, candles: pd.DataFrame) -> FamilySignals:
        close = _close(candles)
        volume = candles["volume"] if "volume" in candles.columns else pd.Series(
            1.0, index=candles.index
        )

        # Rolling VWAP = Σ(price × vol) / Σ(vol) over window.
        pv = close * volume
        rolling_pv = pv.rolling(self._p.vwap_window, min_periods=self._p.vwap_window).sum()
        rolling_vol = volume.rolling(self._p.vwap_window, min_periods=self._p.vwap_window).sum()
        vwap = rolling_pv / rolling_vol

        deviation = close - vwap
        rolling_sigma = deviation.rolling(
            self._p.sigma_window, min_periods=self._p.sigma_window
        ).std()
        z_score = deviation / rolling_sigma

        entries_long = z_score <= -self._p.entry_sigma
        entries_short = z_score >= self._p.entry_sigma

        return FamilySignals(
            entries_long=entries_long.fillna(False),
            entries_short=entries_short.fillna(False),
        )

    def param_grid(self) -> dict[str, list[Any]]:
        return {
            "vwap_window": [30, 60, 120],
            "sigma_window": [30, 60, 120],
            "entry_sigma": [1.5, 2.0, 2.5, 3.0],
        }


def _close(df: pd.DataFrame) -> pd.Series:
    for col in ("mid_close", "bid_close", "close"):
        if col in df.columns:
            return df[col]
    raise KeyError("No close column")
