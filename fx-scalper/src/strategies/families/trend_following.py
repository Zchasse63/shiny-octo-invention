"""Round 10+ trend-following families for H4/D1 timeframes.

Per the evidence review done 2026-04-22:
- Trend following on H4/D1 across diversified FX basket is the most
  strongly evidenced retail-viable strategy class (SG Trend Index +27%
  in 2022, dual-regime survival through COVID + rate-hike cycle).
- Intraday M15 MR on majors is spread-dominated and produces marginal
  edge at best after realistic retail costs.

This module implements three canonical trend-following primitives:

  1. ``DonchianBreakoutFamily``  — classic Turtle-style N-day high/low
     breakout. Parameters: entry_lookback, exit_lookback (chandelier-
     style trailing exit).
  2. ``MACrossoverFamily`` — moving-average crossover (fast/slow). The
     archetypal Rob Carver trend primitive.
  3. ``MomentumRankFamily`` — N-day return ranking. Not yet implemented
     (requires basket context; see Round 13).

All three are trend FOLLOWERS, not predictors — they react to persistent
moves rather than forecast reversals. Per Carver, the honest realistic
Sharpe on any one of these is 0.3-0.8 post-cost; the edge comes from
running many together with low correlation (Round 13 work).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.strategies.families.base_family import FamilySignals, SignalFamily


@dataclass(frozen=True, slots=True)
class DonchianBreakoutParams:
    """Parameters for :class:`DonchianBreakoutFamily`.

    Attributes:
        entry_lookback: Bars to look back for the breakout level. Classic
            Turtle was 20 for entry. Longer = fewer but higher-quality
            signals; shorter = more noise.
        exit_lookback: Bars for the trailing-stop high/low. Classic
            Turtle was 10 for exits (chandelier-style). ``None`` means
            "use the same value as entry_lookback".
        use_both_sides: If True, trade both long and short breakouts.
            If False, long-only (equities-style; rare for FX).
    """

    entry_lookback: int = 20
    exit_lookback: int | None = 10
    use_both_sides: bool = True


class DonchianBreakoutFamily(SignalFamily):
    """N-day high/low breakout with N-day trailing exit (Turtle-style).

    Enters long when the close breaks above the ``entry_lookback``-bar
    high; enters short when it breaks below the ``entry_lookback``-bar
    low. Exits long when the close drops below the ``exit_lookback``-bar
    low (and mirror for shorts). Stop distance is handled externally
    via the common :mod:`src.strategies.exits` framework.
    """

    name = "donchian_breakout"
    params_cls = DonchianBreakoutParams

    def __init__(self, params: DonchianBreakoutParams | None = None) -> None:
        self._p = params or DonchianBreakoutParams()

    def generate(self, candles: pd.DataFrame) -> FamilySignals:
        close_col = "mid_close" if "mid_close" in candles.columns else "bid_close"
        high_col = "mid_high" if "mid_high" in candles.columns else "bid_high"
        low_col = "mid_low" if "mid_low" in candles.columns else "bid_low"
        close = candles[close_col]
        highs = candles[high_col]
        lows = candles[low_col]

        n = self._p.entry_lookback
        # Shift by 1 to avoid look-ahead: the "N-day high" for bar t is the
        # rolling max over bars (t-N, ..., t-1), NOT including bar t itself.
        rolling_high = highs.rolling(n, min_periods=n).max().shift(1)
        rolling_low = lows.rolling(n, min_periods=n).min().shift(1)

        entries_long = close > rolling_high
        entries_short = (
            (close < rolling_low) if self._p.use_both_sides
            else pd.Series(False, index=close.index)
        )

        return FamilySignals(
            entries_long=entries_long.fillna(False).astype(bool),
            entries_short=entries_short.fillna(False).astype(bool),
        )

    def param_grid(self) -> dict[str, list[Any]]:
        # Carver cites 20-80 day as the evidence-supported range for FX
        # trend-following breakouts on D1. We include shorter values for
        # H4 testing (where 20-H4 bars ≈ 3 trading days).
        return {
            "entry_lookback": [20, 40, 60, 80, 100, 120],
            "exit_lookback": [10, 20, 40],
            "use_both_sides": [True],
        }


@dataclass(frozen=True, slots=True)
class MACrossoverParams:
    """Parameters for :class:`MACrossoverFamily`.

    Attributes:
        fast_length: Fast MA window (bars).
        slow_length: Slow MA window (bars). Must be > fast.
        ma_type: ``'ema'`` or ``'sma'``.
    """

    fast_length: int = 20
    slow_length: int = 80
    ma_type: str = "ema"


class MACrossoverFamily(SignalFamily):
    """Moving-average crossover trend-follower.

    Enters long when ``fast_ma`` crosses above ``slow_ma``; mirror for
    short. This is Rob Carver's canonical trend primitive — in his
    systematic-trading work he runs multiple (fast, slow) pairs
    simultaneously as diversified speed-buckets. That setup is built
    in Round 13; this class implements one pair.
    """

    name = "ma_crossover"
    params_cls = MACrossoverParams

    def __init__(self, params: MACrossoverParams | None = None) -> None:
        self._p = params or MACrossoverParams()

    def generate(self, candles: pd.DataFrame) -> FamilySignals:
        close_col = "mid_close" if "mid_close" in candles.columns else "bid_close"
        close = candles[close_col]

        if self._p.ma_type == "ema":
            fast = close.ewm(span=self._p.fast_length, adjust=False).mean()
            slow = close.ewm(span=self._p.slow_length, adjust=False).mean()
        else:
            fast = close.rolling(self._p.fast_length).mean()
            slow = close.rolling(self._p.slow_length).mean()

        above = (fast > slow).fillna(False).astype(bool)
        above_prev = above.shift(1).fillna(False).astype(bool)

        cross_up = above & ~above_prev
        cross_down = ~above & above_prev
        return FamilySignals(
            entries_long=cross_up,
            entries_short=cross_down,
        )

    def param_filter(self, params: dict[str, Any]) -> bool:
        # Enforce fast < slow to avoid degenerate zero-trade configs.
        return bool(params.get("fast_length", 0) < params.get("slow_length", 0))

    def param_grid(self) -> dict[str, list[Any]]:
        # Carver's speed buckets: (8,32), (16,64), (32,128), (64,256) for
        # daily data. We include smaller values for H4 testing.
        return {
            "fast_length": [8, 16, 32, 64],
            "slow_length": [32, 64, 128, 200, 256],
            "ma_type": ["ema", "sma"],
        }
