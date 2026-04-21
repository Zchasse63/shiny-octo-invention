"""Main polling bot — scaffold only.

Wires together OANDA clients, risk guard, strategy, and journal. The
core polling loop is kept simple and synchronous: every tick it reads
state, asks the risk guard, runs the strategy on the last CLOSED bar,
places/manages orders.

This is a scaffold — actual strategy execution wiring lands in Days 4–7.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime

from config.settings import INSTRUMENTS
from src.live.risk import RiskContext, RiskGuard, RiskState
from src.oanda.account import AccountClient
from src.oanda.client import OandaClient
from src.oanda.data import DataClient, Granularity
from src.oanda.instruments import InstrumentRegistry
from src.oanda.orders import OrderClient
from src.strategies.base import Strategy
from src.utils.journal import Journal
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BotConfig:
    """Bot runtime configuration.

    Attributes:
        poll_interval_seconds: Seconds between polls.
        granularity: Candle granularity for signals (e.g. ``"M5"``).
        candle_count: How many candles to pull each poll for indicator warmup.
        instruments: Tuple of instruments to scan. Defaults to settings.INSTRUMENTS.
    """

    poll_interval_seconds: float = 5.0
    granularity: Granularity = "M5"
    candle_count: int = 300
    instruments: tuple[str, ...] = INSTRUMENTS


class Bot:
    """Scaffold polling bot. Strategies plug in via :class:`Strategy`.

    Args:
        client: Authenticated :class:`OandaClient`.
        account: :class:`AccountClient`.
        data: :class:`DataClient`.
        orders: :class:`OrderClient`.
        instruments: Populated :class:`InstrumentRegistry`.
        journal: Shared :class:`Journal`.
        risk: :class:`RiskGuard`.
        strategy: The active strategy.
        config: :class:`BotConfig`.
    """

    def __init__(
        self,
        *,
        client: OandaClient,
        account: AccountClient,
        data: DataClient,
        orders: OrderClient,
        instruments: InstrumentRegistry,
        journal: Journal,
        risk: RiskGuard,
        strategy: Strategy,
        config: BotConfig | None = None,
    ) -> None:
        self._client = client
        self._account = account
        self._data = data
        self._orders = orders
        self._instruments = instruments
        self._journal = journal
        self._risk = risk
        self._strategy = strategy
        self._config = config or BotConfig()
        self._consecutive_api_failures = 0

    def run_forever(self) -> None:
        """Entry point: poll → check risk → run strategy → sleep."""
        logger.info(
            f"Bot starting: strategy={self._strategy.name} "
            f"instruments={self._config.instruments} "
            f"granularity={self._config.granularity}"
        )
        while True:
            try:
                self._tick()
                self._consecutive_api_failures = 0
            except Exception as e:
                self._consecutive_api_failures += 1
                logger.error(
                    f"Tick failed ({self._consecutive_api_failures} consecutive): {e}"
                )
            time.sleep(self._config.poll_interval_seconds)

    def _tick(self) -> None:
        """One poll cycle."""
        snapshot = self._account.snapshot()
        self._journal.record_equity_snapshot(
            balance=snapshot.balance,
            nav=snapshot.nav,
            margin_used=snapshot.margin_used,
            margin_available=snapshot.margin_available,
            open_position_count=snapshot.open_trade_count,
            unrealized_pl=snapshot.unrealized_pl,
        )

        open_trades = self._account.get_open_trades()
        worst_pl = min(
            (float(t.get("unrealizedPL", 0.0)) for t in open_trades),
            default=0.0,
        )
        risk_ctx = RiskContext(
            now_utc=datetime.now(UTC),
            nav=snapshot.nav,
            unrealized_pl=snapshot.unrealized_pl,
            worst_open_trade_pl=worst_pl,
            open_position_count=snapshot.open_trade_count,
            consecutive_api_failures=self._consecutive_api_failures,
        )
        decision = self._risk.check(risk_ctx)

        if decision.state == RiskState.EMERGENCY_SHUTDOWN:
            logger.critical(f"EMERGENCY SHUTDOWN: {decision.reason}")
            # Real implementation: close all, exit process, alert user.
            # Scaffold: raise to surface the condition.
            raise SystemExit(f"Emergency shutdown: {decision.reason}")

        if decision.state == RiskState.HALT_ALL:
            # Per CLAUDE.md §Circuit Breakers: HALT_ALL = no new entries AND
            # no discretionary management — but trailing stops on existing
            # positions must continue (server-side OANDA trails), and we must
            # monitor state so the operator sees the condition persist.
            logger.critical(f"HALT_ALL: {decision.reason}")
            self._monitor_existing_positions()
            return

        allow_new_entries = decision.state == RiskState.OK and not self._risk.max_positions_reached(
            snapshot.open_trade_count
        )

        # HALT_NEW_ENTRIES still allows scanning for trailing-stop management.
        # The strategy won't place new orders (allow_new_entries=False), but
        # existing positions continue under OANDA's server-side trail.
        for instrument in self._config.instruments:
            try:
                self._scan_instrument(instrument, allow_new_entries)
            except Exception as e:
                logger.error(f"scan_instrument({instrument}) failed: {e}")
                raise

    def _monitor_existing_positions(self) -> None:
        """Log existing positions during HALT_ALL so operator has visibility.

        Scaffold: full trailing-stop management wiring lands in Day 4+.
        """
        trades = self._account.get_open_trades()
        if not trades:
            return
        for t in trades:
            logger.warning(
                f"HALT_ALL — open trade: id={t.get('id')} "
                f"instr={t.get('instrument')} units={t.get('currentUnits')} "
                f"unrealized_pl={t.get('unrealizedPL')}"
            )

    def _scan_instrument(self, instrument: str, allow_new_entries: bool) -> None:
        """Run the strategy on one instrument."""
        df = self._data.get_candles(
            instrument=instrument,
            granularity=self._config.granularity,
            count=self._config.candle_count,
            price="BA",
            include_incomplete=False,  # CLAUDE.md: only closed bars.
        )
        if df.empty:
            logger.debug(f"No candles returned for {instrument}")
            return

        # Strategy works on closed bars only; drop anything incomplete that
        # slipped through (defensive).
        df_closed = df[df["complete"]].copy()
        signal = self._strategy.generate_signal(instrument=instrument, candles=df_closed)
        if signal is None:
            return
        if not allow_new_entries:
            logger.debug(f"Signal on {instrument} suppressed: new entries halted")
            return
        self._orders.place_market_order(signal.to_order_request())
