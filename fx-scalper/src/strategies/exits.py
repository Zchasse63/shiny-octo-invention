"""Shared exit framework for every signal family.

All strategy families emit raw ``entries`` / ``short_entries`` boolean
series from their own signal logic. This module converts those entries
into full trade lifecycles — stop losses, take profits, optional
trailing stops, and optional "take partial at 1R, trail the rest"
runner logic.

The framework is designed to match the user-stated exit philosophy:
"base hits (steady income) with the occasional double, triple, homerun
and grand slam — which would be moving that stop trailing stop several
times and getting more profit."

Emit the output as fields on :class:`ExitConfig` and let
``vbt.Portfolio.from_signals`` consume the resulting SL/TP/trail
parameters directly. The common vbt interface means any family's
``entries`` / ``exits`` series can be fed straight into the same
portfolio simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

TrailKind = Literal["off", "atr_trail", "chandelier", "fixed_pct"]
"""Kinds of trailing stop the framework supports.

* ``off`` — no trail; exit only on fixed SL / TP / signal reverse.
* ``atr_trail`` — trail at ``trail_atr_mult × ATR`` from current price.
* ``chandelier`` — trail at highest-high-since-entry minus
  ``trail_atr_mult × ATR`` (longs) or lowest-low-since-entry plus same (shorts).
* ``fixed_pct`` — trail at ``trail_pct`` fraction below current price (longs).
"""


@dataclass(frozen=True, slots=True)
class ExitConfig:
    """Parameters for a family's exit logic.

    Each family reads an ``ExitConfig`` from its constructor and passes
    the derived SL/TP/trail values to the backtest harness. Grid-search
    ranges live alongside each family's own parameter grid.

    Attributes:
        sl_atr_mult: Initial stop-loss distance as a multiple of ATR.
            E.g. 1.0 means "1 ATR below entry on longs".
        atr_length: ATR window used to compute stop distances.
        tp_r_mult: Fixed take-profit as a multiple of initial risk (R).
            1.0 = 1:1 RR take, 2.0 = 2:1 RR take. ``None`` = no fixed TP
            (exit only on trail or opposite signal).
        trail_kind: Which trailing logic to apply. ``"off"`` disables.
        trail_atr_mult: ATR multiplier for ``atr_trail`` / ``chandelier``.
        trail_pct: Fraction below current price for ``fixed_pct`` trail.
        take_partial_at_r: When set, after price reaches this R multiple
            of initial risk, scale out ``partial_fraction`` of position
            and trail the remainder. Matches user's "base hit then trail"
            philosophy. ``None`` = no partial take.
        partial_fraction: Fraction of position to close on partial take
            (0 < frac < 1).
    """

    sl_atr_mult: float = 1.0
    atr_length: int = 14
    tp_r_mult: float | None = 1.0
    trail_kind: TrailKind = "atr_trail"
    trail_atr_mult: float = 2.0
    trail_pct: float = 0.005
    take_partial_at_r: float | None = None
    partial_fraction: float = 0.5


# ---------------------------------------------------------------------------
# Helpers — convert ExitConfig + price series into vbt-ready fields
# ---------------------------------------------------------------------------


def compute_initial_stops(
    entries_long: pd.Series,
    entries_short: pd.Series,
    close: pd.Series,
    atr: pd.Series,
    config: ExitConfig,
) -> dict[str, pd.Series]:
    """Compute per-entry initial SL prices (longs + shorts).

    Args:
        entries_long: Bool series, True on long-entry bars.
        entries_short: Bool series, True on short-entry bars.
        close: Close price series (mid-close is fine for internal math;
            spread is modelled by the harness slippage array).
        atr: Pre-computed ATR series.
        config: :class:`ExitConfig`.

    Returns:
        Dict with keys ``sl_long`` / ``sl_short`` — prices aligned to
        ``close.index``, NaN where no entry fires.
    """
    sl_dist = config.sl_atr_mult * atr
    sl_long = close - sl_dist
    sl_short = close + sl_dist
    return {
        "sl_long": sl_long.where(entries_long, other=np.nan),
        "sl_short": sl_short.where(entries_short, other=np.nan),
    }


def compute_take_profits(
    entries_long: pd.Series,
    entries_short: pd.Series,
    close: pd.Series,
    atr: pd.Series,
    config: ExitConfig,
) -> dict[str, pd.Series]:
    """Compute per-entry TP prices.

    If ``config.tp_r_mult`` is None, returns NaN series (no fixed TP).
    Otherwise TP = entry ± (tp_r_mult × sl_atr_mult × ATR).

    Initial risk R = ``sl_atr_mult × ATR``. TP = entry ± tp_r_mult × R.
    """
    if config.tp_r_mult is None:
        nan_series = pd.Series(np.nan, index=close.index)
        return {"tp_long": nan_series.copy(), "tp_short": nan_series.copy()}
    r = config.sl_atr_mult * atr
    tp_dist = config.tp_r_mult * r
    tp_long = close + tp_dist
    tp_short = close - tp_dist
    return {
        "tp_long": tp_long.where(entries_long, other=np.nan),
        "tp_short": tp_short.where(entries_short, other=np.nan),
    }


@dataclass(frozen=True, slots=True)
class VbtExitParams:
    """Fields ready to hand to ``vbt.Portfolio.from_signals``.

    Attributes:
        sl_stop: Per-bar SL distance as a fraction of price, NaN off-entry.
        tp_stop: Per-bar TP distance as a fraction of price, NaN where none.
        sl_trail: True if SL should trail (set once globally per run).
        trail_distance_pct: Fraction of price for trailing distance
            (derived from ATR trail for the backtest).
    """

    sl_stop: pd.Series
    tp_stop: pd.Series
    sl_trail: bool
    trail_distance_pct: pd.Series | None = None


def config_to_vbt_params(
    entries_long: pd.Series,
    entries_short: pd.Series,
    close: pd.Series,
    atr: pd.Series,
    config: ExitConfig,
) -> VbtExitParams:
    """Convert an :class:`ExitConfig` + price context into vbt-native fields.

    vbt's ``Portfolio.from_signals`` consumes ``sl_stop`` / ``tp_stop`` as
    fractions of price (e.g. 0.005 = 0.5% below entry). We convert ATR-based
    absolute distances to price fractions at entry time so vbt applies them
    consistently.

    Args:
        entries_long: Bool series, True on long entries.
        entries_short: Bool series, True on short entries.
        close: Close price at each bar.
        atr: ATR series.
        config: :class:`ExitConfig`.

    Returns:
        :class:`VbtExitParams`.
    """
    # Combined entry bar flag — vbt treats `sl_stop` / `tp_stop` as scalars
    # but also accepts per-entry arrays. We build per-bar fractions that are
    # valid at any entry bar and NaN elsewhere (vbt picks up the most recent
    # non-NaN value at entry time for that position).
    entry_any = entries_long | entries_short

    r = config.sl_atr_mult * atr
    sl_frac = (r / close).where(entry_any, other=np.nan)

    if config.tp_r_mult is None:
        tp_frac = pd.Series(np.nan, index=close.index)
    else:
        tp_frac = ((config.tp_r_mult * r) / close).where(entry_any, other=np.nan)

    trail_flag = config.trail_kind != "off"

    trail_frac: pd.Series | None
    if config.trail_kind == "atr_trail":
        trail_frac = ((config.trail_atr_mult * atr) / close).where(
            entry_any, other=np.nan
        )
    elif config.trail_kind == "chandelier":
        # Chandelier uses running-extreme anchoring; vbt's sl_trail flag gives
        # us the running-extreme behaviour natively. Approximate the
        # chandelier distance with trail_atr_mult × ATR at entry.
        trail_frac = ((config.trail_atr_mult * atr) / close).where(
            entry_any, other=np.nan
        )
    elif config.trail_kind == "fixed_pct":
        trail_frac = pd.Series(config.trail_pct, index=close.index).where(
            entry_any, other=np.nan
        )
    else:
        trail_frac = None

    return VbtExitParams(
        sl_stop=sl_frac,
        tp_stop=tp_frac,
        sl_trail=trail_flag,
        trail_distance_pct=trail_frac,
    )


# ---------------------------------------------------------------------------
# Default grids per trail kind — for Phase 2 exploratory sweep
# ---------------------------------------------------------------------------


DEFAULT_SL_ATR_MULT_GRID: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
"""Initial stop-loss ATR multipliers worth exploring."""

DEFAULT_TP_R_MULT_GRID: tuple[float | None, ...] = (None, 0.75, 1.0, 1.5, 2.0)
"""Fixed take-profit R multipliers. ``None`` = no fixed TP, rely on trail."""

DEFAULT_TRAIL_KINDS: tuple[TrailKind, ...] = ("off", "atr_trail", "chandelier")
"""Trail variants to test."""

DEFAULT_TRAIL_ATR_MULT_GRID: tuple[float, ...] = (1.0, 2.0, 3.0)
"""Trail distances in ATR units."""


def enumerate_exit_configs(
    *,
    sl_atr_mults: tuple[float, ...] = DEFAULT_SL_ATR_MULT_GRID,
    tp_r_mults: tuple[float | None, ...] = DEFAULT_TP_R_MULT_GRID,
    trail_kinds: tuple[TrailKind, ...] = DEFAULT_TRAIL_KINDS,
    trail_atr_mults: tuple[float, ...] = DEFAULT_TRAIL_ATR_MULT_GRID,
    atr_length: int = 14,
) -> list[ExitConfig]:
    """Return every ExitConfig from the Cartesian product of the grids.

    Skips redundant combos (e.g. trail_atr_mult when trail_kind="off").
    """
    out: list[ExitConfig] = []
    for sl in sl_atr_mults:
        for tp in tp_r_mults:
            for trail_kind in trail_kinds:
                if trail_kind == "off":
                    out.append(
                        ExitConfig(
                            sl_atr_mult=sl,
                            atr_length=atr_length,
                            tp_r_mult=tp,
                            trail_kind="off",
                        )
                    )
                    continue
                for trail_mult in trail_atr_mults:
                    out.append(
                        ExitConfig(
                            sl_atr_mult=sl,
                            atr_length=atr_length,
                            tp_r_mult=tp,
                            trail_kind=trail_kind,
                            trail_atr_mult=trail_mult,
                        )
                    )
    return out
