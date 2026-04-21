"""Signal families for the exploratory phase.

Each family emits ``entries_long`` / ``entries_short`` boolean series
given a price DataFrame. Exits are handled uniformly by
:mod:`src.strategies.exits`. See ADR 0003 for the framework rationale.
"""

from __future__ import annotations

from src.strategies.families.base_family import FamilySignals, SignalFamily
from src.strategies.families.bb_rsi_mr import BBRSIMRFamily, BBRSIMRParams
from src.strategies.families.ema_cross import EMACrossFamily, EMACrossParams
from src.strategies.families.pullback_ema import PullbackEMAFamily, PullbackEMAParams
from src.strategies.families.range_breakout import (
    RangeBreakoutFamily,
    RangeBreakoutParams,
)
from src.strategies.families.rsi_extreme import RSIExtremeFamily, RSIExtremeParams
from src.strategies.families.vwap_deviation import (
    VWAPDeviationFamily,
    VWAPDeviationParams,
)

__all__ = [
    "BBRSIMRFamily",
    "BBRSIMRParams",
    "EMACrossFamily",
    "EMACrossParams",
    "FamilySignals",
    "PullbackEMAFamily",
    "PullbackEMAParams",
    "RangeBreakoutFamily",
    "RangeBreakoutParams",
    "RSIExtremeFamily",
    "RSIExtremeParams",
    "SignalFamily",
    "VWAPDeviationFamily",
    "VWAPDeviationParams",
    "ALL_FAMILIES",
]

ALL_FAMILIES: list[type[SignalFamily]] = [
    PullbackEMAFamily,
    RangeBreakoutFamily,
    VWAPDeviationFamily,
    EMACrossFamily,
    BBRSIMRFamily,
    RSIExtremeFamily,
]
"""Every family included in the exploratory sweep."""
