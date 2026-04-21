"""Trailing-stop math tests."""

from __future__ import annotations

import pytest

from src.live.trailing import TrailingContext, chandelier_stop_price, compute_trailing_distance


class TestComputeTrailingDistance:
    def test_initial_multiplier(self) -> None:
        ctx = TrailingContext(
            side="LONG",
            current_price=1.08,
            current_atr=0.0015,
            entry_price=1.078,
            initial_atr_multiplier=2.0,
            tight_atr_multiplier=0.5,
            should_tighten=False,
        )
        assert abs(compute_trailing_distance(ctx) - 0.003) < 1e-9

    def test_tightens(self) -> None:
        ctx = TrailingContext(
            side="LONG",
            current_price=1.08,
            current_atr=0.0015,
            entry_price=1.078,
            initial_atr_multiplier=2.0,
            tight_atr_multiplier=0.5,
            should_tighten=True,
        )
        assert abs(compute_trailing_distance(ctx) - 0.00075) < 1e-9

    def test_rejects_non_positive_atr(self) -> None:
        with pytest.raises(ValueError):
            compute_trailing_distance(
                TrailingContext(
                    side="LONG",
                    current_price=1.08,
                    current_atr=0.0,
                    entry_price=1.078,
                )
            )


class TestChandelierStopPrice:
    def test_long(self) -> None:
        price = chandelier_stop_price(
            side="LONG",
            extreme_since_entry=1.0900,
            current_atr=0.0020,
            atr_multiplier=3.0,
        )
        assert abs(price - (1.09 - 0.006)) < 1e-9

    def test_short(self) -> None:
        price = chandelier_stop_price(
            side="SHORT",
            extreme_since_entry=1.0800,
            current_atr=0.0020,
            atr_multiplier=3.0,
        )
        assert abs(price - (1.08 + 0.006)) < 1e-9

    def test_rejects_bad_side(self) -> None:
        with pytest.raises(ValueError):
            chandelier_stop_price(
                side="NEUTRAL",
                extreme_since_entry=1.08,
                current_atr=0.002,
            )
