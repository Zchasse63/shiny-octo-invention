"""Signal filters — composable gates that mask entries based on regime / session / state.

A filter takes a DataFrame of candles + the family's raw entries and returns
masked entries (same shape, some bars turned False). Multiple filters compose:

    filtered = session_filter(adx_filter(raw_entries, candles), candles)

Round 1 exploration showed that every naive family loses money OOS at M1.
The vbt.chat iteration analysis identified regime-unfiltered trading as the
#1 gap — "the same signal likely trades dead hours, transition hours, trend
hours, mean-reverting hours; that destroys a weak intraday edge."

This module implements the filters needed for round 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.indicators.engine import add_adx, add_atr

# ---------------------------------------------------------------------------
# ADX regime filter — gate entries by trending/ranging regime
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ADXFilterParams:
    """Parameters for :func:`adx_filter`.

    Attributes:
        adx_length: ADX indicator window.
        max_adx: Entries allowed only when ADX <= this (ranging regime).
            Mean-reversion strategies want low ADX.
        min_adx: Entries allowed only when ADX >= this (trending regime).
            Momentum strategies want high ADX. Default None = no lower bound.
    """

    adx_length: int = 14
    max_adx: float | None = 25.0
    min_adx: float | None = None


def adx_filter(
    entries: pd.Series,
    candles: pd.DataFrame,
    params: ADXFilterParams | None = None,
) -> pd.Series:
    """Mask entries where ADX regime doesn't match.

    Args:
        entries: Bool series from a family's signal logic.
        candles: OHLCV frame the entries were computed against.
        params: :class:`ADXFilterParams`.

    Returns:
        Masked entries: True only where original AND in the ADX regime window.
    """
    p = params or ADXFilterParams()
    df = add_adx(candles, length=p.adx_length)
    adx = df[f"adx_{p.adx_length}"]
    mask = pd.Series(True, index=candles.index)
    if p.max_adx is not None:
        mask &= adx <= p.max_adx
    if p.min_adx is not None:
        mask &= adx >= p.min_adx
    mask = mask.fillna(False).astype(bool)
    return entries & mask


# ---------------------------------------------------------------------------
# Session filter — gate by time-of-day (UTC)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SessionFilterParams:
    """Parameters for :func:`session_filter`.

    Attributes:
        allowed_hours_utc: Tuple of UTC hours (0-23) where entries are allowed.
            Examples:
            - (23, 0, 1, 2, 3, 4, 5, 6) = Asian session (23:00-07:00 UTC)
            - (7, 8, 9, 10, 11) = pre-London + London open
            - (12, 13, 14, 15) = London-NY overlap
            - (16, 17, 18, 19, 20) = NY afternoon
    """

    allowed_hours_utc: tuple[int, ...] = tuple(range(24))  # all hours by default


def session_filter(
    entries: pd.Series,
    candles: pd.DataFrame,
    params: SessionFilterParams | None = None,
) -> pd.Series:
    """Mask entries outside the allowed UTC hour window.

    Args:
        entries: Bool entries.
        candles: Tz-aware UTC-indexed frame.
        params: :class:`SessionFilterParams`.

    Returns:
        Masked entries: True only where original AND in an allowed UTC hour.
    """
    p = params or SessionFilterParams()
    if not p.allowed_hours_utc:
        return entries
    hours = candles.index.hour
    mask = pd.Series(hours, index=candles.index).isin(p.allowed_hours_utc)
    return entries & mask


# ---------------------------------------------------------------------------
# Volatility regime filter — gate by ATR percentile
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class VolRegimeFilterParams:
    """Parameters for :func:`vol_regime_filter`.

    Attributes:
        atr_length: ATR window.
        lookback: Rolling window to compute ATR percentile rank.
        min_percentile: Allow entries only where current ATR percentile >= this.
            Default 0 = no lower bound.
        max_percentile: Allow entries only where current ATR percentile <= this.
            Default 1 = no upper bound.
    """

    atr_length: int = 14
    lookback: int = 1440  # ~1 day of M1 bars
    min_percentile: float = 0.0
    max_percentile: float = 1.0


def vol_regime_filter(
    entries: pd.Series,
    candles: pd.DataFrame,
    params: VolRegimeFilterParams | None = None,
) -> pd.Series:
    """Mask entries outside the desired volatility percentile window.

    Args:
        entries: Bool entries.
        candles: OHLCV frame.
        params: :class:`VolRegimeFilterParams`.

    Returns:
        Masked entries.
    """
    p = params or VolRegimeFilterParams()
    df = add_atr(candles, length=p.atr_length)
    atr = df[f"atr_{p.atr_length}"]
    pct = atr.rolling(p.lookback, min_periods=p.lookback).rank(pct=True)
    mask = (pct >= p.min_percentile) & (pct <= p.max_percentile)
    mask = mask.fillna(False).astype(bool)
    return entries & mask


# ---------------------------------------------------------------------------
# Spread filter — skip entries when cost-to-range ratio is bad
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SpreadFilterParams:
    """Parameters for :func:`spread_filter`.

    Attributes:
        max_spread_atr_frac: Skip entries when (ask-bid) > N × ATR. Default 0.2
            means spread must be ≤20% of ATR at entry bar. Wide-spread bars
            (overnight lull, news releases) are often unprofitable to trade.
        atr_length: ATR window.
    """

    max_spread_atr_frac: float = 0.2
    atr_length: int = 14


def spread_filter(
    entries: pd.Series,
    candles: pd.DataFrame,
    params: SpreadFilterParams | None = None,
) -> pd.Series:
    """Mask entries where spread is too wide relative to ATR."""
    p = params or SpreadFilterParams()
    if "bid_close" not in candles.columns or "ask_close" not in candles.columns:
        return entries
    df = add_atr(candles, length=p.atr_length)
    atr = df[f"atr_{p.atr_length}"]
    spread = candles["ask_close"] - candles["bid_close"]
    mask = (spread / atr) <= p.max_spread_atr_frac
    mask = mask.fillna(False).astype(bool)
    return entries & mask


# ---------------------------------------------------------------------------
# Weekday filter — gate by day of week
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class WeekdayFilterParams:
    """Parameters for :func:`weekday_filter`.

    Attributes:
        allowed_weekdays: Tuple of Python weekdays (Mon=0 … Fri=4) where
            entries are allowed. Saturday=5 and Sunday=6 are market-closed.
            Common variants:
            - (0,1,2,3,4) = all weekdays (default, no filter)
            - (1,2,3) = Tue-Thu only (avoid Monday gap + Friday chop)
            - (0,1,2,3) = Mon-Thu (avoid Friday profit-taking)
            - (1,2,3,4) = Tue-Fri (avoid Monday gap)
    """

    allowed_weekdays: tuple[int, ...] = (0, 1, 2, 3, 4)


def weekday_filter(
    entries: pd.Series,
    candles: pd.DataFrame,
    params: WeekdayFilterParams | None = None,
) -> pd.Series:
    """Mask entries outside allowed weekdays.

    vbt.chat iteration (round-3.5 artifact) identified weekday / intraday
    filters as the cheapest untested "non-indicator" dimension.
    """
    p = params or WeekdayFilterParams()
    if len(p.allowed_weekdays) >= 5:
        return entries
    weekdays = candles.index.weekday
    mask = pd.Series(weekdays, index=candles.index).isin(p.allowed_weekdays)
    return entries & mask


# ---------------------------------------------------------------------------
# Compose-all — default preset for "sensible" filtering
# ---------------------------------------------------------------------------

def apply_filter_stack(
    entries_long: pd.Series,
    entries_short: pd.Series,
    candles: pd.DataFrame,
    *,
    adx: ADXFilterParams | None = None,
    session: SessionFilterParams | None = None,
    vol: VolRegimeFilterParams | None = None,
    weekday: WeekdayFilterParams | None = None,
    spread: SpreadFilterParams | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Apply a stack of filters to long + short entries.

    Each ``None`` means skip that filter. Filters compose via AND.

    Returns:
        (filtered_long, filtered_short).
    """
    def _apply_one(entries: pd.Series) -> pd.Series:
        out = entries
        if adx is not None:
            out = adx_filter(out, candles, adx)
        if session is not None:
            out = session_filter(out, candles, session)
        if vol is not None:
            out = vol_regime_filter(out, candles, vol)
        if weekday is not None:
            out = weekday_filter(out, candles, weekday)
        if spread is not None:
            out = spread_filter(out, candles, spread)
        return out

    return _apply_one(entries_long), _apply_one(entries_short)
