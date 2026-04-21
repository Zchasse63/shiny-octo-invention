"""Round-2 families: mean-reversion with regime / session filters.

Wraps the base ``bb_rsi_mr`` and ``rsi_extreme`` families with composable
filters from :mod:`src.strategies.filters`. The filter parameters are
part of the family param grid so the sweep can search over them directly.

Per the vbt.chat iteration analysis (docs/research/ai_queries/
20260421T210830-iter1_eurusd_what_went_wrong.md), regime filtering was
the #1 missing piece from round 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.strategies.families.base_family import FamilySignals, SignalFamily
from src.strategies.families.bb_rsi_mr import BBRSIMRFamily, BBRSIMRParams
from src.strategies.families.rsi_extreme import RSIExtremeFamily, RSIExtremeParams
from src.strategies.filters import (
    ADXFilterParams,
    SessionFilterParams,
    SpreadFilterParams,
    apply_filter_stack,
)

# Pre-defined session buckets covering the main FX sessions.
_SESSION_PRESETS: dict[str, tuple[int, ...]] = {
    "all": tuple(range(24)),
    "asian": (23, 0, 1, 2, 3, 4, 5, 6),
    "pre_london": (6, 7),
    "london": (8, 9, 10, 11),
    "london_ny_overlap": (12, 13, 14, 15),
    "ny": (16, 17, 18, 19, 20),
    "active": (7, 8, 9, 10, 11, 12, 13, 14, 15, 16),  # London+NY combined
    "asian_plus_london": (23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
}


@dataclass(frozen=True, slots=True)
class FilteredBBRSIMRParams:
    """Filtered BB-RSI-MR params — includes regime + session + spread filters.

    Attributes:
        bb_length: Bollinger Band length.
        bb_std: Bollinger Band std multiplier.
        rsi_length: RSI period.
        rsi_long_threshold: Long when RSI < this value.
        rsi_short_threshold: Short when RSI > this value.
        max_adx: ADX must be <= this (ranging regime). None = no ADX filter.
        session: One of ``_SESSION_PRESETS`` keys.
        max_spread_atr_frac: Skip entries when spread > this × ATR.
    """

    # Signal params — narrowed to round-1's basin
    bb_length: int = 30
    bb_std: float = 2.0
    rsi_length: int = 21
    rsi_long_threshold: float = 25.0
    rsi_short_threshold: float = 75.0
    # Filters
    max_adx: float | None = 25.0
    session: str = "all"
    max_spread_atr_frac: float = 0.25


class FilteredBBRSIMRFamily(SignalFamily):
    """BB-RSI mean reversion with ADX + session + spread filters.

    Round 2 focus: wrap round-1's most-promising family with the regime
    and session filtering vbt.chat identified as the #1 gap.
    """

    name = "bb_rsi_mr_filtered"
    params_cls = FilteredBBRSIMRParams

    def __init__(self, params: FilteredBBRSIMRParams | None = None) -> None:
        self._p = params or FilteredBBRSIMRParams()

    def generate(self, candles: pd.DataFrame) -> FamilySignals:
        # Run the base bb_rsi_mr logic first.
        base = BBRSIMRFamily(
            BBRSIMRParams(
                bb_length=self._p.bb_length,
                bb_std=self._p.bb_std,
                rsi_length=self._p.rsi_length,
                rsi_long_threshold=self._p.rsi_long_threshold,
                rsi_short_threshold=self._p.rsi_short_threshold,
            )
        ).generate(candles)

        # Compose filters.
        session_hours = _SESSION_PRESETS.get(self._p.session, _SESSION_PRESETS["all"])
        filtered_long, filtered_short = apply_filter_stack(
            entries_long=base.entries_long,
            entries_short=base.entries_short,
            candles=candles,
            adx=(
                ADXFilterParams(max_adx=self._p.max_adx)
                if self._p.max_adx is not None
                else None
            ),
            session=SessionFilterParams(allowed_hours_utc=session_hours),
            spread=(
                SpreadFilterParams(max_spread_atr_frac=self._p.max_spread_atr_frac)
                if self._p.max_spread_atr_frac < 1.0
                else None
            ),
        )
        return FamilySignals(
            entries_long=filtered_long.fillna(False).astype(bool),
            entries_short=filtered_short.fillna(False).astype(bool),
        )

    def param_grid(self) -> dict[str, list[Any]]:
        return {
            # Round-1 basin, widened slightly
            "bb_length": [20, 30, 40],
            "bb_std": [2.0, 2.25, 2.5],
            "rsi_length": [14, 21, 30],
            "rsi_long_threshold": [20, 25, 30],
            "rsi_short_threshold": [70, 75, 80],
            # New filter dimensions
            "max_adx": [None, 20.0, 25.0, 30.0],
            "session": ["all", "asian", "active", "london_ny_overlap"],
            "max_spread_atr_frac": [0.15, 0.25, 0.5],
        }


@dataclass(frozen=True, slots=True)
class FilteredRSIExtremeParams:
    """Filtered RSI-extreme params."""

    rsi_length: int = 21
    oversold: float = 25.0
    overbought: float = 75.0
    max_adx: float | None = 25.0
    session: str = "all"


class FilteredRSIExtremeFamily(SignalFamily):
    """RSI-extreme family with ADX + session filters."""

    name = "rsi_extreme_filtered"
    params_cls = FilteredRSIExtremeParams

    def __init__(self, params: FilteredRSIExtremeParams | None = None) -> None:
        self._p = params or FilteredRSIExtremeParams()

    def generate(self, candles: pd.DataFrame) -> FamilySignals:
        base = RSIExtremeFamily(
            RSIExtremeParams(
                rsi_length=self._p.rsi_length,
                oversold=self._p.oversold,
                overbought=self._p.overbought,
            )
        ).generate(candles)

        session_hours = _SESSION_PRESETS.get(self._p.session, _SESSION_PRESETS["all"])
        filtered_long, filtered_short = apply_filter_stack(
            entries_long=base.entries_long,
            entries_short=base.entries_short,
            candles=candles,
            adx=(
                ADXFilterParams(max_adx=self._p.max_adx)
                if self._p.max_adx is not None
                else None
            ),
            session=SessionFilterParams(allowed_hours_utc=session_hours),
        )
        return FamilySignals(
            entries_long=filtered_long.fillna(False).astype(bool),
            entries_short=filtered_short.fillna(False).astype(bool),
        )

    def param_grid(self) -> dict[str, list[Any]]:
        return {
            "rsi_length": [14, 21, 30],
            "oversold": [20, 25, 30],
            "overbought": [70, 75, 80],
            "max_adx": [None, 20.0, 25.0, 30.0],
            "session": ["all", "asian", "active", "london_ny_overlap"],
        }
