"""Position sizing unit tests — verify the CLAUDE.md formula exactly."""

from __future__ import annotations

import pytest

from src.live.sizing import compute_position_units


class TestComputePositionUnits:
    """§Position Sizing Logic — cash × leverage → units."""

    def test_eur_usd_default(self) -> None:
        """At 1.0800, $100 × 50 = $5,000 notional ≈ 4,629 units of EUR."""
        units = compute_position_units(
            cash_committed_usd=100,
            leverage=50,
            current_price=1.0800,
            instrument="EUR_USD",
        )
        # Expected: int(5000 / 1.08) = int(4629.62...) = 4629
        assert units == 4629

    def test_usd_jpy_base_is_usd(self) -> None:
        """USD_JPY: USD is base — units equal notional in USD."""
        units = compute_position_units(
            cash_committed_usd=100,
            leverage=50,
            current_price=155.20,
            instrument="USD_JPY",
        )
        assert units == 5000

    def test_gbp_usd(self) -> None:
        """GBP_USD at 1.25 → int(5000 / 1.25) = 4000 units of GBP."""
        units = compute_position_units(
            cash_committed_usd=100,
            leverage=50,
            current_price=1.2500,
            instrument="GBP_USD",
        )
        assert units == 4000

    def test_rejects_non_positive_cash(self) -> None:
        with pytest.raises(ValueError):
            compute_position_units(
                cash_committed_usd=0,
                leverage=50,
                current_price=1.0800,
                instrument="EUR_USD",
            )

    def test_rejects_non_positive_leverage(self) -> None:
        with pytest.raises(ValueError):
            compute_position_units(
                cash_committed_usd=100,
                leverage=0,
                current_price=1.0800,
                instrument="EUR_USD",
            )

    def test_rejects_non_positive_price(self) -> None:
        with pytest.raises(ValueError):
            compute_position_units(
                cash_committed_usd=100,
                leverage=50,
                current_price=0,
                instrument="EUR_USD",
            )

    def test_rejects_bad_instrument_format(self) -> None:
        with pytest.raises(ValueError):
            compute_position_units(
                cash_committed_usd=100,
                leverage=50,
                current_price=1.08,
                instrument="EURUSD",
            )

    def test_scales_linearly_with_cash(self) -> None:
        """Double the cash → double the units (approximately)."""
        u1 = compute_position_units(
            cash_committed_usd=100,
            leverage=50,
            current_price=1.0800,
            instrument="EUR_USD",
        )
        u2 = compute_position_units(
            cash_committed_usd=200,
            leverage=50,
            current_price=1.0800,
            instrument="EUR_USD",
        )
        assert 1.99 <= u2 / u1 <= 2.01

    def test_scales_with_leverage(self) -> None:
        u1 = compute_position_units(
            cash_committed_usd=100,
            leverage=25,
            current_price=1.0800,
            instrument="EUR_USD",
        )
        u2 = compute_position_units(
            cash_committed_usd=100,
            leverage=50,
            current_price=1.0800,
            instrument="EUR_USD",
        )
        assert 1.99 <= u2 / u1 <= 2.01
