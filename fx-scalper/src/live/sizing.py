"""Position sizing: cash-committed × leverage → integer units.

Implements the exact formula in CLAUDE.md §Position Sizing Logic. Do not
alter without updating the architecture doc. Sizing is *not* derived from
risk-percentage — it's derived from the cash the user commits per trade.
"""

from __future__ import annotations

from config.settings import ACCOUNT_CURRENCY, CASH_PER_TRADE_USD, MAX_LEVERAGE


def compute_position_units(
    cash_committed_usd: float = CASH_PER_TRADE_USD,
    leverage: int = MAX_LEVERAGE,
    *,
    current_price: float,
    instrument: str,
    account_currency: str = ACCOUNT_CURRENCY,
) -> int:
    """Return integer units for a trade sized from cash committed × leverage.

    Args:
        cash_committed_usd: Cash margin committed to this trade (default $100).
        leverage: Leverage multiplier (default 50 — OANDA US max).
        current_price: Current mid price of the instrument (e.g. 1.0800 on EUR/USD).
        instrument: OANDA underscore-form instrument name (e.g. ``EUR_USD``).
        account_currency: Account denomination (default ``USD``).

    Returns:
        Integer units, positive.

    Raises:
        ValueError: If inputs are non-positive.

    Examples:
        >>> compute_position_units(
        ...     cash_committed_usd=100, leverage=50,
        ...     current_price=1.0800, instrument="EUR_USD")
        4629
        >>> compute_position_units(
        ...     cash_committed_usd=100, leverage=50,
        ...     current_price=155.20, instrument="USD_JPY")
        5000
    """
    if cash_committed_usd <= 0:
        raise ValueError(f"cash_committed_usd must be > 0, got {cash_committed_usd}")
    if leverage <= 0:
        raise ValueError(f"leverage must be > 0, got {leverage}")
    if current_price <= 0:
        raise ValueError(f"current_price must be > 0, got {current_price}")
    if "_" not in instrument:
        raise ValueError(f"instrument must be underscore-form, got {instrument!r}")

    notional_usd = cash_committed_usd * leverage
    base_ccy, quote_ccy = instrument.split("_")

    if base_ccy == account_currency:
        # e.g. USD_JPY at price 155.20 with $5,000 notional → 5,000 units of USD.
        units = notional_usd
    elif quote_ccy == account_currency:
        # e.g. EUR_USD at price 1.0800 with $5,000 notional →
        # 5,000 / 1.0800 = 4,629 units of EUR (the base).
        units = notional_usd / current_price
    else:
        # Non-USD cross. Fallback: treat current_price as base→USD conversion.
        # Precise handling would require a separate USD-anchor rate lookup.
        units = notional_usd / current_price

    return int(units)
