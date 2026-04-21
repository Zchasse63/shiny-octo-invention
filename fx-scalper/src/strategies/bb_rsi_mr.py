"""Strategy 1: Bollinger Band + RSI mean reversion.

Per CLAUDE.md §Starter Strategies:

* Pairs: EUR/USD, USD/JPY
* Timeframe: M5 or M15
* Session: Asian (23:00–07:00 UTC) preferred
* Signal:
    Long:  close < lower BB(20, 2.0) AND RSI(14) < 30 AND ADX(14) < 20
    Short: close > upper BB(20, 2.0) AND RSI(14) > 70 AND ADX(14) < 20
* SL: 1.5 × ATR(14)
* TP: opposite BB band (set by strategy; trail takes over if profitable)

Sizing is applied by the caller via :mod:`src.live.sizing`. This module
emits signals only.

Per CLAUDE.md §Code Standards: no magic numbers — parameters are exposed via
:class:`BBRSIParams` so the Day-4 parameter sweep can grid over them.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config.settings import (
    ASIAN_SESSION_END_UTC_HOUR,
    ASIAN_SESSION_START_UTC_HOUR,
    CASH_PER_TRADE_USD,
    MAX_LEVERAGE,
)
from src.indicators.engine import add_adx, add_atr, add_bbands, add_rsi
from src.live.sizing import compute_position_units
from src.strategies.base import Side, Signal, Strategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class BBRSIParams:
    """Tunable parameters — grid-searchable on Day 4.

    Attributes:
        bb_length: Bollinger Band window (CLAUDE.md sweep grid: 15/20/30).
        bb_std: Bollinger Band std multiplier (1.8/2.0/2.2).
        rsi_length: RSI window (10/14/21).
        rsi_long_threshold: Long when RSI below this (25/30/35).
        rsi_short_threshold: Short when RSI above this (65/70/75).
        adx_threshold: Only trade when ADX below this (no-trend filter).
        adx_length: ADX window.
        atr_length: ATR window for SL distance.
        sl_atr_multiplier: SL = entry ∓ k × ATR.
        tp_band: ``"opposite"`` (exit at opposite BB) or ``"midline"``.
        asian_session_only: Restrict entries to 23:00–07:00 UTC window.
    """

    bb_length: int = 20
    bb_std: float = 2.0
    rsi_length: int = 14
    rsi_long_threshold: float = 30.0
    rsi_short_threshold: float = 70.0
    adx_threshold: float = 20.0
    adx_length: int = 14
    atr_length: int = 14
    sl_atr_multiplier: float = 1.5
    tp_band: str = "opposite"
    asian_session_only: bool = True


class BBRSIMeanReversion(Strategy):
    """Bollinger + RSI mean reversion with ADX no-trend filter.

    Args:
        params: Strategy parameters.
        cash_per_trade_usd: Cash committed per trade (default: settings value).
        leverage: Leverage multiplier (default: settings value).
    """

    NAME = "bb_rsi_mr"

    def __init__(
        self,
        params: BBRSIParams | None = None,
        *,
        cash_per_trade_usd: float = CASH_PER_TRADE_USD,
        leverage: int = MAX_LEVERAGE,
    ) -> None:
        self._p = params or BBRSIParams()
        self._cash_per_trade_usd = cash_per_trade_usd
        self._leverage = leverage

    @property
    def name(self) -> str:
        return self.NAME

    def generate_signal(
        self,
        *,
        instrument: str,
        candles: pd.DataFrame,
    ) -> Signal | None:
        """Evaluate entry on the last CLOSED bar of ``candles``.

        CLAUDE.md §Code Standards: use ``iloc[-1]`` here because the caller
        has already filtered to closed bars only (see bot._scan_instrument).
        """
        min_warmup = max(
            self._p.bb_length,
            self._p.rsi_length,
            self._p.adx_length,
            self._p.atr_length,
        ) + 2
        if len(candles) < min_warmup:
            return None

        df = add_bbands(candles, length=self._p.bb_length, std=self._p.bb_std)
        df = add_rsi(df, length=self._p.rsi_length)
        df = add_adx(df, length=self._p.adx_length)
        df = add_atr(df, length=self._p.atr_length)

        row = df.iloc[-1]
        ts = df.index[-1]

        if self._p.asian_session_only and not _in_asian_session(ts):
            return None

        mid_close = _close_price(row)
        bb_lo = row[f"bb_lower_{self._p.bb_length}_{self._p.bb_std}"]
        bb_hi = row[f"bb_upper_{self._p.bb_length}_{self._p.bb_std}"]
        bb_mid = row[f"bb_middle_{self._p.bb_length}_{self._p.bb_std}"]
        rsi = row[f"rsi_{self._p.rsi_length}"]
        adx = row[f"adx_{self._p.adx_length}"]
        atr = row[f"atr_{self._p.atr_length}"]

        # Any NaN means insufficient warmup; skip.
        if any(pd.isna(x) for x in (mid_close, bb_lo, bb_hi, bb_mid, rsi, adx, atr)):
            return None

        # Only trade in low-trend regime (ADX below threshold).
        if adx >= self._p.adx_threshold:
            return None

        side: Side | None = None
        # Signal trigger uses mid — equal treatment for long and short entries.
        if mid_close < bb_lo and rsi < self._p.rsi_long_threshold:
            side = "LONG"
        elif mid_close > bb_hi and rsi > self._p.rsi_short_threshold:
            side = "SHORT"

        if side is None:
            return None

        # SL/TP must be anchored to the EXECUTION price the broker will fill at,
        # not the mid price used for indicators. Per CLAUDE.md §"Honest backtests":
        # model the spread. A LONG fills at ask, a SHORT at bid.
        fill_anchor = _fill_price_anchor(row, side=side, fallback=mid_close)

        if side == "LONG":
            sl_price = fill_anchor - self._p.sl_atr_multiplier * atr
            tp_price = bb_hi if self._p.tp_band == "opposite" else bb_mid
        else:
            sl_price = fill_anchor + self._p.sl_atr_multiplier * atr
            tp_price = bb_lo if self._p.tp_band == "opposite" else bb_mid

        units = compute_position_units(
            cash_committed_usd=self._cash_per_trade_usd,
            leverage=self._leverage,
            current_price=fill_anchor,
            instrument=instrument,
        )

        return Signal(
            strategy=self.NAME,
            instrument=instrument,
            side=side,
            units=units,
            sl_price=float(sl_price),
            tp_price=float(tp_price),
            trailing_stop_distance=None,  # Managed client-side initially.
        )


def _close_price(row: pd.Series) -> float:
    """Pull close from mid_close / bid_close / close depending on availability."""
    for col in ("mid_close", "bid_close", "close"):
        if col in row.index and not pd.isna(row[col]):
            return float(row[col])
    raise KeyError("No close column found on row")


def _fill_price_anchor(row: pd.Series, *, side: Side, fallback: float) -> float:
    """Return the price the broker will fill a market order at.

    LONG fills at ask, SHORT fills at bid. If bid/ask columns aren't present
    on the row (e.g. a test fixture with only mid), fall back to ``fallback``.
    """
    col = "ask_close" if side == "LONG" else "bid_close"
    if col in row.index and not pd.isna(row[col]):
        return float(row[col])
    return fallback


def _in_asian_session(ts: pd.Timestamp) -> bool:
    """True if ``ts`` falls within 23:00–07:00 UTC (Asian session)."""
    if ts.tz is None:
        raise ValueError("ts must be tz-aware UTC")
    hour = ts.tz_convert("UTC").hour
    start = ASIAN_SESSION_START_UTC_HOUR
    end = ASIAN_SESSION_END_UTC_HOUR
    # Wraps midnight.
    return hour >= start or hour < end
