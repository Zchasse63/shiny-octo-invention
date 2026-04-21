"""vectorbt Pro backtest harness — scaffold (Day 3).

Per CLAUDE.md §Day 3: build a harness on top of ``vectorbt Pro`` that:

* Takes a close-price DataFrame + a bid/ask spread DataFrame.
* Calls ``Portfolio.from_signals`` with ``sl_stop``, ``tp_stop``, ``sl_trail``.
* Sets ``leverage=50``, ``freq='1min'``.
* Applies time-varying slippage from spread.
* Excludes Fri 17:00 ET → Sun 17:00 ET session.

vectorbt Pro is a paid private package. Import is deferred so this module
imports cleanly before the user installs it; calling ``run_backtest`` raises
a helpful error if the package is missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from config.settings import (
    FRIDAY_FLAT_BY_UTC_HOUR,
    MAX_LEVERAGE,
    SUNDAY_OPEN_UTC_HOUR,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class BacktestInputs:
    """Inputs passed to the vectorbt Pro portfolio builder.

    Attributes:
        close: Close price time series (mid close preferred), indexed UTC.
        entries: Boolean long entries aligned to ``close``.
        exits: Boolean long exits aligned to ``close``.
        short_entries: Boolean short entries.
        short_exits: Boolean short exits.
        spread: Half-spread in price units (same shape as ``close``), used for
            per-bar slippage modelling.
        sl_stop: Per-entry stop-loss distance as a fraction of price (e.g. 0.005).
        tp_stop: Per-entry take-profit distance as a fraction of price.
        sl_trail: True if SL should trail.
        leverage: Leverage multiplier (default: settings).
        initial_cash: Starting equity.
    """

    close: pd.Series
    entries: pd.Series
    exits: pd.Series
    short_entries: pd.Series
    short_exits: pd.Series
    spread: pd.Series
    sl_stop: float
    tp_stop: float
    sl_trail: bool = True
    leverage: int = MAX_LEVERAGE
    initial_cash: float = 500.0


def run_backtest(inputs: BacktestInputs) -> Any:
    """Run a vectorbt Pro backtest.

    Returns the ``Portfolio`` object so caller can pull stats / trades / drawdown.

    Raises:
        ImportError: If ``vectorbtpro`` is not installed.
    """
    try:
        import vectorbtpro as vbt
    except ImportError as e:
        raise ImportError(
            "vectorbtpro is required for backtesting. Install with:\n"
            "  pip install -U 'vectorbtpro[base] @ git+"
            "https://github.com/polakowo/vectorbt.pro.git'\n"
            "After confirming the user's GitHub has been granted access by Oleg."
        ) from e

    close = inputs.close
    _validate_aligned(
        close,
        inputs.entries,
        inputs.exits,
        inputs.short_entries,
        inputs.short_exits,
        inputs.spread,
    )

    weekend_mask = _weekend_mask(close.index)
    # Suppress all signals during the flat-by-Friday → Sunday-open window.
    entries = inputs.entries & ~weekend_mask
    short_entries = inputs.short_entries & ~weekend_mask
    # Force-exit at Friday cutoff if weekend_mask just turned True.
    cutoff_exit = weekend_mask & (~weekend_mask.shift(1, fill_value=False))
    exits = inputs.exits | cutoff_exit
    short_exits = inputs.short_exits | cutoff_exit

    # pandas 2.x removed fillna(method=); use explicit ffill().
    # vectorbt Pro expects a numpy array for time-varying slippage, not a Series.
    slippage_arr = (
        inputs.spread.reindex(close.index).ffill().fillna(0.0).to_numpy() / 2.0
    )

    portfolio = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        short_entries=short_entries,
        short_exits=short_exits,
        sl_stop=inputs.sl_stop,
        tp_stop=inputs.tp_stop,
        sl_trail=inputs.sl_trail,
        leverage=inputs.leverage,
        init_cash=inputs.initial_cash,
        freq="1min",
        slippage=slippage_arr,
    )
    logger.info(
        f"Backtest ran: trades={portfolio.trades.count()} "
        f"final_value={portfolio.final_value():.2f}"
    )
    return portfolio


def _validate_aligned(*series: pd.Series) -> None:
    """All series must share the same index."""
    ref = series[0].index
    for s in series[1:]:
        if not s.index.equals(ref):
            raise ValueError("All input series must share the same index")


def _weekend_mask(index: pd.DatetimeIndex) -> pd.Series:
    """True on bars inside the Fri 20:00 UTC → Sun 22:00 UTC no-trade window.

    Uses :data:`FRIDAY_FLAT_BY_UTC_HOUR` (20:00 UTC, roughly 16:00 ET) as the
    flat-by time and :data:`SUNDAY_OPEN_UTC_HOUR` (22:00 UTC) as re-open.
    """
    if index.tz is None:
        raise ValueError("index must be tz-aware UTC")
    idx = index.tz_convert("UTC")
    wd = idx.weekday
    hr = idx.hour

    fri_after = (wd == 4) & (hr >= FRIDAY_FLAT_BY_UTC_HOUR)
    saturday = wd == 5
    sunday_before = (wd == 6) & (hr < SUNDAY_OPEN_UTC_HOUR)

    mask = fri_after | saturday | sunday_before
    return pd.Series(np.asarray(mask), index=index)
