"""Dynamic trailing-stop logic.

Per CLAUDE.md §Mission: "Trades are NOT closed by time — only by price action
against the trail or by hitting stop loss."

Per Strategy 1: "Trail: 2× ATR, tighten to 0.5× ATR if RSI reverses through 50."

This module is strategy-agnostic: it computes a *new* trailing distance given
an open trade's context, and returns the distance (caller invokes
``OrderClient.modify_trailing_stop``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrailingContext:
    """Inputs for a trailing-stop recompute.

    Attributes:
        side: ``"LONG"`` or ``"SHORT"``.
        current_price: Latest market price.
        current_atr: Current ATR(14) value in price units.
        entry_price: Fill price when the trade opened.
        initial_atr_multiplier: Starting trail distance = k × ATR (e.g. 2.0).
        tight_atr_multiplier: Tightened trail = k_tight × ATR (e.g. 0.5).
        should_tighten: Strategy-specific flag (e.g. RSI reversed through 50).
    """

    side: str
    current_price: float
    current_atr: float
    entry_price: float
    initial_atr_multiplier: float = 2.0
    tight_atr_multiplier: float = 0.5
    should_tighten: bool = False


def compute_trailing_distance(ctx: TrailingContext) -> float:
    """Return the target trailing-stop distance in price units.

    Args:
        ctx: Trailing recompute context.

    Returns:
        Positive float — distance in price units (e.g. 0.0020 on EUR/USD = 20 pips).

    Raises:
        ValueError: If ``current_atr`` is non-positive.
    """
    if ctx.current_atr <= 0:
        raise ValueError(f"current_atr must be > 0, got {ctx.current_atr}")
    k = ctx.tight_atr_multiplier if ctx.should_tighten else ctx.initial_atr_multiplier
    return k * ctx.current_atr


def chandelier_stop_price(
    *,
    side: str,
    extreme_since_entry: float,
    current_atr: float,
    atr_multiplier: float = 3.0,
) -> float:
    """Chandelier exit price (Strategy 2 trailing).

    For longs: ``highest_since_entry − k × ATR``.
    For shorts: ``lowest_since_entry + k × ATR``.

    Args:
        side: ``"LONG"`` or ``"SHORT"``.
        extreme_since_entry: Highest high (LONG) or lowest low (SHORT) observed
            since entry, in price units.
        current_atr: ATR(14) value in price units.
        atr_multiplier: k (default 3).

    Returns:
        Absolute price for the stop.
    """
    if current_atr <= 0:
        raise ValueError(f"current_atr must be > 0, got {current_atr}")
    if side == "LONG":
        return extreme_since_entry - atr_multiplier * current_atr
    if side == "SHORT":
        return extreme_since_entry + atr_multiplier * current_atr
    raise ValueError(f"side must be 'LONG' or 'SHORT', got {side!r}")
