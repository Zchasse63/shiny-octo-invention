"""Account-level queries: balance, NAV, margin, positions, trades."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from oandapyV20.endpoints.accounts import AccountDetails, AccountSummary
from oandapyV20.endpoints.positions import OpenPositions
from oandapyV20.endpoints.trades import OpenTrades

from src.oanda.client import OandaClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    """Point-in-time account state.

    Attributes:
        balance: Cash balance in account currency.
        nav: Net asset value (balance + unrealized PnL).
        margin_used: Margin currently tied up in open positions.
        margin_available: Margin free to open new trades.
        unrealized_pl: Unrealized PnL across all open positions.
        open_trade_count: Number of distinct open trades.
        currency: Account currency (e.g. "USD").
    """

    balance: float
    nav: float
    margin_used: float
    margin_available: float
    unrealized_pl: float
    open_trade_count: int
    currency: str


class AccountClient:
    """Account queries wrapping OANDA endpoints.

    Args:
        client: Authenticated :class:`OandaClient`.
    """

    def __init__(self, client: OandaClient) -> None:
        self._client = client

    # ------------------------------------------------------------------
    # Summary/snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> AccountSnapshot:
        """Return a fresh :class:`AccountSnapshot` from OANDA."""
        resp = self._client.request(AccountSummary(accountID=self._client.account_id))
        acct = resp["account"]
        return AccountSnapshot(
            balance=float(acct["balance"]),
            nav=float(acct["NAV"]),
            margin_used=float(acct.get("marginUsed", 0.0)),
            margin_available=float(acct.get("marginAvailable", 0.0)),
            unrealized_pl=float(acct.get("unrealizedPL", 0.0)),
            open_trade_count=int(acct.get("openTradeCount", 0)),
            currency=str(acct.get("currency", "USD")),
        )

    def details(self) -> dict[str, Any]:
        """Full AccountDetails response (trades + positions + orders)."""
        return self._client.request(AccountDetails(accountID=self._client.account_id))

    # ------------------------------------------------------------------
    # Convenience scalars
    # ------------------------------------------------------------------

    def get_balance(self) -> float:
        """Cash balance in account currency."""
        return self.snapshot().balance

    def get_nav(self) -> float:
        """Net asset value (balance + unrealized PnL)."""
        return self.snapshot().nav

    def get_margin_used(self) -> float:
        """Margin currently committed to open positions."""
        return self.snapshot().margin_used

    def get_margin_available(self) -> float:
        """Margin available for new positions."""
        return self.snapshot().margin_available

    # ------------------------------------------------------------------
    # Positions / trades
    # ------------------------------------------------------------------

    def get_open_positions(self) -> list[dict[str, Any]]:
        """List open positions (one per instrument with non-zero net units)."""
        resp = self._client.request(OpenPositions(accountID=self._client.account_id))
        return list(resp.get("positions", []))

    def get_open_trades(self) -> list[dict[str, Any]]:
        """List individual open trades (tickets)."""
        resp = self._client.request(OpenTrades(accountID=self._client.account_id))
        return list(resp.get("trades", []))
