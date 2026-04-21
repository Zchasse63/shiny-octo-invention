"""Circuit breaker tests — every breaker must trip on its condition."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from config.settings import (
    ACCOUNT_FLOOR_USD,
    DAILY_LOSS_LIMIT_USD,
    MAX_CONSECUTIVE_LOSSES,
    OANDA_CONSECUTIVE_FAILURE_LIMIT,
    SINGLE_TRADE_MAX_LOSS_USD,
)
from src.live.risk import RiskContext, RiskGuard, RiskState


def _mid_week_utc() -> datetime:
    """A Tuesday at 10:00 UTC — outside all weekend/session gates."""
    # 2024-03-05 is a Tuesday.
    return datetime(2024, 3, 5, 10, 0, tzinfo=UTC)


class TestAccountFloor:
    def test_trips_emergency_on_floor_breach(self, tmp_journal) -> None:
        guard = RiskGuard(tmp_journal)
        ctx = RiskContext(
            now_utc=_mid_week_utc(),
            nav=ACCOUNT_FLOOR_USD - 0.01,
            unrealized_pl=0.0,
            worst_open_trade_pl=0.0,
            open_position_count=0,
        )
        decision = guard.check(ctx)
        assert decision.state == RiskState.EMERGENCY_SHUTDOWN
        assert "account_floor" in decision.reason

    def test_ok_when_above_floor(self, tmp_journal) -> None:
        guard = RiskGuard(tmp_journal)
        ctx = RiskContext(
            now_utc=_mid_week_utc(),
            nav=ACCOUNT_FLOOR_USD + 50.0,
            unrealized_pl=0.0,
            worst_open_trade_pl=0.0,
            open_position_count=0,
        )
        assert guard.check(ctx).state == RiskState.OK


class TestSingleTradeBlowout:
    def test_trips_halt_all_on_blowout(self, tmp_journal) -> None:
        guard = RiskGuard(tmp_journal)
        ctx = RiskContext(
            now_utc=_mid_week_utc(),
            nav=500.0,
            unrealized_pl=-SINGLE_TRADE_MAX_LOSS_USD - 1.0,
            worst_open_trade_pl=-(SINGLE_TRADE_MAX_LOSS_USD + 1.0),
            open_position_count=1,
        )
        decision = guard.check(ctx)
        # Dollar-loss is only $31, well under the $50 daily limit, so blowout
        # is the dominant trip — HALT_ALL.
        assert decision.state == RiskState.HALT_ALL
        assert "single_trade_blowout" in decision.reason or any(
            "single_trade_blowout" in b for b in decision.tripped_breakers
        )


class TestDailyLossLimit:
    def test_trips_halt_all_when_unrealized_crosses(self, tmp_journal) -> None:
        guard = RiskGuard(tmp_journal)
        ctx = RiskContext(
            now_utc=_mid_week_utc(),
            nav=450.0,
            unrealized_pl=-DAILY_LOSS_LIMIT_USD - 1.0,
            worst_open_trade_pl=-10.0,  # Not enough for blowout
            open_position_count=1,
        )
        decision = guard.check(ctx)
        assert decision.state == RiskState.HALT_ALL

    def test_persists_halt_after_initial_trip(self, tmp_journal) -> None:
        guard = RiskGuard(tmp_journal)
        t0 = _mid_week_utc()
        # First check trips and sets internal state.
        guard.check(
            RiskContext(
                now_utc=t0,
                nav=450.0,
                unrealized_pl=-DAILY_LOSS_LIMIT_USD - 1.0,
                worst_open_trade_pl=-10.0,
                open_position_count=1,
            )
        )
        # Second check, an hour later, still halted even if unrealized recovered.
        later = t0 + timedelta(hours=1)
        d2 = guard.check(
            RiskContext(
                now_utc=later,
                nav=500.0,
                unrealized_pl=0.0,
                worst_open_trade_pl=0.0,
                open_position_count=0,
            )
        )
        assert d2.state == RiskState.HALT_ALL


class TestConsecutiveLosses:
    def _seed_losses(self, journal, n: int) -> None:
        for i in range(n):
            uuid = f"uuid-loss-{i}"
            journal.record_trade_open(
                trade_uuid=uuid,
                strategy="bb_rsi_mr",
                instrument="EUR_USD",
                side="LONG",
                units=4629,
                entry_price=1.08,
                sl_price=1.075,
                tp_price=1.085,
            )
            journal.record_trade_close(
                trade_uuid=uuid,
                exit_price=1.075,
                pl_realized=-10.0,
                commission_total=0.0,
                financing_total=0.0,
                close_reason="SL",
            )

    def test_trips_halt_new_entries(self, tmp_journal) -> None:
        self._seed_losses(tmp_journal, MAX_CONSECUTIVE_LOSSES)
        guard = RiskGuard(tmp_journal)
        decision = guard.check(
            RiskContext(
                now_utc=_mid_week_utc(),
                nav=470.0,
                unrealized_pl=0.0,
                worst_open_trade_pl=0.0,
                open_position_count=0,
            )
        )
        assert decision.state == RiskState.HALT_NEW_ENTRIES

    def test_pause_expires_after_window(self, tmp_journal) -> None:
        """Regression: the timer-reset bug would keep refreshing forever.

        The pause must expire after CONSECUTIVE_LOSS_PAUSE_MINUTES even if
        the journal still shows the same loss streak (no winning trade yet).
        """
        self._seed_losses(tmp_journal, MAX_CONSECUTIVE_LOSSES)
        guard = RiskGuard(tmp_journal)
        t0 = _mid_week_utc()
        # First check arms the timer.
        d0 = guard.check(
            RiskContext(
                now_utc=t0,
                nav=470.0,
                unrealized_pl=0.0,
                worst_open_trade_pl=0.0,
                open_position_count=0,
            )
        )
        assert d0.state == RiskState.HALT_NEW_ENTRIES
        # A second check 30 minutes later still halts (timer active).
        d1 = guard.check(
            RiskContext(
                now_utc=t0 + timedelta(minutes=30),
                nav=470.0,
                unrealized_pl=0.0,
                worst_open_trade_pl=0.0,
                open_position_count=0,
            )
        )
        assert d1.state == RiskState.HALT_NEW_ENTRIES
        # After >60min the pause clears even if losses still on record.
        d2 = guard.check(
            RiskContext(
                now_utc=t0 + timedelta(minutes=61),
                nav=470.0,
                unrealized_pl=0.0,
                worst_open_trade_pl=0.0,
                open_position_count=0,
            )
        )
        assert d2.state == RiskState.OK


class TestOandaDisconnect:
    def test_trips_halt_new_entries(self, tmp_journal) -> None:
        guard = RiskGuard(tmp_journal)
        decision = guard.check(
            RiskContext(
                now_utc=_mid_week_utc(),
                nav=500.0,
                unrealized_pl=0.0,
                worst_open_trade_pl=0.0,
                open_position_count=0,
                consecutive_api_failures=OANDA_CONSECUTIVE_FAILURE_LIMIT,
            )
        )
        assert decision.state == RiskState.HALT_NEW_ENTRIES


class TestWeekendBoundary:
    def test_friday_late_halts_all(self, tmp_journal) -> None:
        guard = RiskGuard(tmp_journal)
        # 2024-03-08 is a Friday; 21:00 UTC is after the flat-by cutoff.
        friday_late = datetime(2024, 3, 8, 21, 0, tzinfo=UTC)
        decision = guard.check(
            RiskContext(
                now_utc=friday_late,
                nav=500.0,
                unrealized_pl=0.0,
                worst_open_trade_pl=0.0,
                open_position_count=0,
            )
        )
        assert decision.state == RiskState.HALT_ALL

    def test_saturday_halts_all(self, tmp_journal) -> None:
        guard = RiskGuard(tmp_journal)
        saturday = datetime(2024, 3, 9, 12, 0, tzinfo=UTC)
        decision = guard.check(
            RiskContext(
                now_utc=saturday,
                nav=500.0,
                unrealized_pl=0.0,
                worst_open_trade_pl=0.0,
                open_position_count=0,
            )
        )
        assert decision.state == RiskState.HALT_ALL

    def test_sunday_before_open_halts_new_entries(self, tmp_journal) -> None:
        guard = RiskGuard(tmp_journal)
        # Sunday 2024-03-10 at 18:00 UTC — still before 22:00 open.
        sunday_pre = datetime(2024, 3, 10, 18, 0, tzinfo=UTC)
        decision = guard.check(
            RiskContext(
                now_utc=sunday_pre,
                nav=500.0,
                unrealized_pl=0.0,
                worst_open_trade_pl=0.0,
                open_position_count=0,
            )
        )
        assert decision.state == RiskState.HALT_NEW_ENTRIES

    def test_sunday_after_open_ok(self, tmp_journal) -> None:
        guard = RiskGuard(tmp_journal)
        # Sunday 22:30 UTC — after open.
        sunday_post = datetime(2024, 3, 10, 22, 30, tzinfo=UTC)
        decision = guard.check(
            RiskContext(
                now_utc=sunday_post,
                nav=500.0,
                unrealized_pl=0.0,
                worst_open_trade_pl=0.0,
                open_position_count=0,
            )
        )
        assert decision.state == RiskState.OK


class TestMaxPositions:
    def test_flag_at_cap(self, tmp_journal) -> None:
        guard = RiskGuard(tmp_journal)
        assert guard.max_positions_reached(2) is True

    def test_flag_under_cap(self, tmp_journal) -> None:
        guard = RiskGuard(tmp_journal)
        assert guard.max_positions_reached(1) is False


def test_requires_tz_aware(tmp_journal) -> None:
    guard = RiskGuard(tmp_journal)
    naive = datetime(2024, 3, 5, 10, 0)  # No tzinfo
    with pytest.raises(ValueError):
        guard.check(
            RiskContext(
                now_utc=naive,
                nav=500.0,
                unrealized_pl=0.0,
                worst_open_trade_pl=0.0,
                open_position_count=0,
            )
        )
