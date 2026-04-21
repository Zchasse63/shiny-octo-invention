"""Circuit breakers — non-negotiable overrides to strategy logic.

Per CLAUDE.md §Circuit Breakers, these six rules MUST be checked on every
poll cycle, not just at entry:

1. Account floor — NAV < $400 → ``EMERGENCY_SHUTDOWN``.
2. Daily loss limit — realized+unrealized PnL today ≤ -$50 → ``HALT_ALL``
   until 00:00 UTC tomorrow.
3. Consecutive losses — last 3 closed trades lost → ``HALT_NEW_ENTRIES``
   for 1 hour. Trailing on existing positions continues.
4. Single-trade blowout — any open trade unrealized < -$30 → ``HALT_ALL``,
   alert, do NOT auto-close (investigate).
5. OANDA disconnect — N consecutive API failures → ``HALT_NEW_ENTRIES``,
   keep trying to manage existing.
6. Sunday/Friday boundary — no new entries Fri 17:00 ET → Sun 17:00 ET;
   flat by Fri 16:55 ET.

The main loop calls :meth:`RiskGuard.check` before every entry and every
poll cycle. The returned state is authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum

from config.settings import (
    ACCOUNT_FLOOR_USD,
    CONSECUTIVE_LOSS_PAUSE_MINUTES,
    DAILY_LOSS_LIMIT_USD,
    FRIDAY_CLOSE_UTC_HOUR,
    FRIDAY_FLAT_BY_UTC_HOUR,
    MAX_CONCURRENT_POSITIONS,
    MAX_CONSECUTIVE_LOSSES,
    OANDA_CONSECUTIVE_FAILURE_LIMIT,
    SINGLE_TRADE_MAX_LOSS_USD,
    SUNDAY_OPEN_UTC_HOUR,
)
from src.utils.journal import Journal
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RiskState(Enum):
    """Authoritative risk state returned by :meth:`RiskGuard.check`."""

    OK = "OK"
    """Strategy may enter new trades and manage existing ones normally."""

    HALT_NEW_ENTRIES = "HALT_NEW_ENTRIES"
    """No new positions. Existing positions continue to be managed (trail/SL)."""

    HALT_ALL = "HALT_ALL"
    """No new entries AND no management actions beyond exits. Monitor only."""

    EMERGENCY_SHUTDOWN = "EMERGENCY_SHUTDOWN"
    """Close everything, exit the bot, alert the user."""


@dataclass
class RiskContext:
    """Inputs needed to evaluate circuit breakers.

    Attributes:
        now_utc: Current tz-aware UTC timestamp.
        nav: Account net asset value.
        unrealized_pl: Sum of unrealized PnL across open positions.
        worst_open_trade_pl: Lowest (most negative) unrealized PnL on any
            single open trade.
        open_position_count: Number of currently open positions.
        consecutive_api_failures: Count of back-to-back OANDA failures.
    """

    now_utc: datetime
    nav: float
    unrealized_pl: float
    worst_open_trade_pl: float
    open_position_count: int
    consecutive_api_failures: int = 0


@dataclass
class RiskDecision:
    """The check result.

    Attributes:
        state: The enum.
        reason: Short human-readable why.
        tripped_breakers: List of breaker names that triggered.
    """

    state: RiskState
    reason: str
    tripped_breakers: list[str] = field(default_factory=list)


class RiskGuard:
    """Evaluates circuit breakers against live state.

    Args:
        journal: Shared SQLite journal — queried for consecutive losses and
            written to on every trip.
    """

    def __init__(self, journal: Journal) -> None:
        self._journal = journal
        self._paused_until_utc: datetime | None = None
        self._halted_until_utc: datetime | None = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def check(self, ctx: RiskContext) -> RiskDecision:
        """Evaluate all breakers and return the most restrictive state.

        Priority: EMERGENCY_SHUTDOWN > HALT_ALL > HALT_NEW_ENTRIES > OK.

        Raises:
            ValueError: If ``ctx.now_utc`` is not tz-aware. All timestamps in
                this module must be tz-aware UTC per CLAUDE.md §Code Standards.
        """
        if ctx.now_utc.tzinfo is None:
            raise ValueError("ctx.now_utc must be tz-aware UTC")

        tripped: list[tuple[RiskState, str]] = []

        # 1) Account floor
        if ctx.nav < ACCOUNT_FLOOR_USD:
            tripped.append(
                (
                    RiskState.EMERGENCY_SHUTDOWN,
                    f"account_floor: nav ${ctx.nav:.2f} < ${ACCOUNT_FLOOR_USD:.2f}",
                )
            )

        # 4) Single-trade blowout
        if ctx.worst_open_trade_pl < -SINGLE_TRADE_MAX_LOSS_USD:
            tripped.append(
                (
                    RiskState.HALT_ALL,
                    (
                        f"single_trade_blowout: worst open trade "
                        f"${ctx.worst_open_trade_pl:.2f} < "
                        f"${-SINGLE_TRADE_MAX_LOSS_USD:.2f}"
                    ),
                )
            )

        # 2) Daily loss limit
        day_start = ctx.now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        realized_today = self._journal.realized_pl_since(day_start.isoformat())
        total_day_pl = realized_today + ctx.unrealized_pl
        if total_day_pl <= -DAILY_LOSS_LIMIT_USD:
            # Halt until next UTC midnight.
            self._halted_until_utc = day_start + timedelta(days=1)
            tripped.append(
                (
                    RiskState.HALT_ALL,
                    (
                        f"daily_loss_limit: realized={realized_today:.2f} + "
                        f"unrealized={ctx.unrealized_pl:.2f} = {total_day_pl:.2f} "
                        f"≤ ${-DAILY_LOSS_LIMIT_USD:.2f}; halted until "
                        f"{self._halted_until_utc.isoformat()}"
                    ),
                )
            )
        elif self._halted_until_utc is not None and ctx.now_utc < self._halted_until_utc:
            tripped.append(
                (
                    RiskState.HALT_ALL,
                    f"daily_loss_limit carryover until {self._halted_until_utc.isoformat()}",
                )
            )
        elif self._halted_until_utc is not None:
            # Expire the halt.
            logger.info(f"Daily loss halt cleared at {ctx.now_utc.isoformat()}")
            self._halted_until_utc = None

        # 3) Consecutive losses → pause new entries.
        # Arm the timer only on the edge (when we first cross the threshold),
        # otherwise the timer would reset on every poll and never expire.
        consec = self._journal.consecutive_losses()
        if consec >= MAX_CONSECUTIVE_LOSSES and self._paused_until_utc is None:
            self._paused_until_utc = ctx.now_utc + timedelta(
                minutes=CONSECUTIVE_LOSS_PAUSE_MINUTES
            )
            tripped.append(
                (
                    RiskState.HALT_NEW_ENTRIES,
                    (
                        f"consecutive_losses: {consec} in a row; paused until "
                        f"{self._paused_until_utc.isoformat()}"
                    ),
                )
            )
        elif (
            self._paused_until_utc is not None
            and ctx.now_utc < self._paused_until_utc
        ):
            tripped.append(
                (
                    RiskState.HALT_NEW_ENTRIES,
                    f"consecutive_losses carryover until {self._paused_until_utc.isoformat()}",
                )
            )
        elif self._paused_until_utc is not None:
            # Timer expired. Clear it — a fresh loss-streak will re-arm.
            logger.info(f"Consecutive-loss pause cleared at {ctx.now_utc.isoformat()}")
            self._paused_until_utc = None

        # 5) OANDA disconnect
        if ctx.consecutive_api_failures >= OANDA_CONSECUTIVE_FAILURE_LIMIT:
            tripped.append(
                (
                    RiskState.HALT_NEW_ENTRIES,
                    f"oanda_disconnect: {ctx.consecutive_api_failures} consecutive failures",
                )
            )

        # 6) Friday/Sunday boundary
        boundary = self._weekend_boundary_state(ctx.now_utc)
        if boundary is not None:
            tripped.append(boundary)

        # Max concurrent positions is a soft gate (not a halt) — skip new
        # entries when at cap, but keep state OK otherwise. Strategies should
        # consult this via :meth:`max_positions_reached`.

        if not tripped:
            return RiskDecision(state=RiskState.OK, reason="all_clear")

        # Pick the most restrictive.
        order = {
            RiskState.EMERGENCY_SHUTDOWN: 4,
            RiskState.HALT_ALL: 3,
            RiskState.HALT_NEW_ENTRIES: 2,
            RiskState.OK: 1,
        }
        tripped.sort(key=lambda t: order[t[0]], reverse=True)
        state, reason = tripped[0]
        names = [r.split(":", 1)[0] for _, r in tripped]

        # Journal every trip (dedup by reason string within one minute could be
        # added later if log volume gets noisy).
        self._journal.record_risk_event(
            breaker=names[0],
            state=state.value,
            detail=reason,
            context={
                "nav": ctx.nav,
                "unrealized_pl": ctx.unrealized_pl,
                "worst_open_trade_pl": ctx.worst_open_trade_pl,
                "open_position_count": ctx.open_position_count,
                "consecutive_api_failures": ctx.consecutive_api_failures,
                "tripped": names,
            },
        )
        return RiskDecision(state=state, reason=reason, tripped_breakers=names)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def max_positions_reached(self, open_position_count: int) -> bool:
        """True if we're at or above the concurrent-positions cap."""
        return open_position_count >= MAX_CONCURRENT_POSITIONS

    @staticmethod
    def _weekend_boundary_state(now_utc: datetime) -> tuple[RiskState, str] | None:
        """Return halt state during Fri 16:55 ET → Sun 17:00 ET window.

        Args:
            now_utc: Current tz-aware UTC time.

        Returns:
            None if outside the window, else (RiskState, reason).
        """
        if now_utc.tzinfo is None:
            raise ValueError("now_utc must be tz-aware")
        # weekday(): Mon=0 ... Sun=6
        wd = now_utc.weekday()
        hr = now_utc.hour

        # Friday: after FRIDAY_FLAT_BY_UTC_HOUR → HALT_ALL (close + no new).
        if wd == 4 and hr >= FRIDAY_FLAT_BY_UTC_HOUR:
            return (
                RiskState.HALT_ALL,
                f"weekend_boundary: Fri {hr:02d}:00 UTC >= "
                f"{FRIDAY_FLAT_BY_UTC_HOUR:02d}:00 UTC flat-by cutoff",
            )
        # Saturday: always HALT_ALL.
        if wd == 5:
            return (
                RiskState.HALT_ALL,
                "weekend_boundary: Saturday — market closed",
            )
        # Sunday: until SUNDAY_OPEN_UTC_HOUR.
        if wd == 6 and hr < SUNDAY_OPEN_UTC_HOUR:
            return (
                RiskState.HALT_NEW_ENTRIES,
                f"weekend_boundary: Sun {hr:02d}:00 UTC < "
                f"{SUNDAY_OPEN_UTC_HOUR:02d}:00 UTC open",
            )
        # Keep FRIDAY_CLOSE_UTC_HOUR as a backstop (overlap with FRIDAY_FLAT_BY).
        if wd == 4 and hr >= FRIDAY_CLOSE_UTC_HOUR:
            return (
                RiskState.HALT_ALL,
                f"weekend_boundary: Fri {hr:02d}:00 UTC >= "
                f"{FRIDAY_CLOSE_UTC_HOUR:02d}:00 UTC close",
            )
        return None


def now_utc() -> datetime:
    """Return current tz-aware UTC datetime (helper for callers)."""
    return datetime.now(UTC)
