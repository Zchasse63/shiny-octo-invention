"""Order placement, modification, and close.

Per CLAUDE.md §Code Standards: every order carries a magic number, strategy
name, and trade UUID in ``clientExtensions.id``. All requests + responses
are journaled.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Literal

from oandapyV20.endpoints.orders import OrderCreate
from oandapyV20.endpoints.trades import TradeClose, TradeCRCDO

from config.settings import MAGIC_NUMBER
from src.oanda.client import OandaClient
from src.oanda.instruments import InstrumentRegistry, InstrumentSpec
from src.utils.logger import get_logger

logger = get_logger(__name__)


Side = Literal["LONG", "SHORT"]


@dataclass(frozen=True, slots=True)
class OrderRequest:
    """Intent to open a market order.

    Attributes:
        strategy: Strategy name (e.g. ``"bb_rsi_mr"``).
        instrument: OANDA instrument (e.g. ``"EUR_USD"``).
        side: ``"LONG"`` or ``"SHORT"``.
        units: Integer unit count — caller is responsible for sign handling.
            Positive for long, negative for short. Absolute value must be >= min.
        sl_price: Optional stop-loss price (absolute, not distance).
        tp_price: Optional take-profit price (absolute, not distance).
        trailing_stop_distance: Optional server-side trailing distance in price
            units (e.g. 0.0020 = 20 pips on EUR/USD).
    """

    strategy: str
    instrument: str
    side: Side
    units: int
    sl_price: float | None = None
    tp_price: float | None = None
    trailing_stop_distance: float | None = None


@dataclass(frozen=True, slots=True)
class OrderResult:
    """Outcome of an order placement.

    Attributes:
        trade_uuid: Our UUID (used as magic-number suffix).
        oanda_order_id: OANDA's order-create transaction id, if the order was
            accepted.
        oanda_trade_id: OANDA's trade id if the order filled immediately.
        fill_price: Fill price if filled.
        status: One of ``"FILLED"``, ``"PENDING"``, ``"REJECTED"``, ``"CANCELED"``.
        raw_response: Full raw OANDA response for debugging.
    """

    trade_uuid: str
    oanda_order_id: str | None
    oanda_trade_id: str | None
    fill_price: float | None
    status: str
    raw_response: dict[str, Any]


class OrderClient:
    """Places and manages OANDA orders.

    Args:
        client: Authenticated :class:`OandaClient`.
        instruments: Populated :class:`InstrumentRegistry` for precision handling.
    """

    def __init__(self, client: OandaClient, instruments: InstrumentRegistry) -> None:
        self._client = client
        self._instruments = instruments

    # ------------------------------------------------------------------
    # Market order
    # ------------------------------------------------------------------

    def place_market_order(self, req: OrderRequest) -> OrderResult:
        """Submit a market order, journal it, and return a structured result.

        Args:
            req: Validated :class:`OrderRequest`.

        Returns:
            :class:`OrderResult` describing the outcome.
        """
        spec = self._instruments.get(req.instrument)
        signed_units = self._signed_units(req, spec)
        trade_uuid = str(uuid.uuid4())
        magic_id = f"{MAGIC_NUMBER}:{req.strategy}:{trade_uuid}"

        payload = self._build_order_payload(
            req=req,
            spec=spec,
            signed_units=signed_units,
            magic_id=magic_id,
        )

        self._client.journal.record_order(
            strategy=req.strategy,
            magic_id=magic_id,
            trade_uuid=trade_uuid,
            instrument=req.instrument,
            side=req.side,
            units=signed_units,
            entry_price_req=None,  # Market order; fill price comes from response.
            sl_price=req.sl_price,
            tp_price=req.tp_price,
            trailing_distance=req.trailing_stop_distance,
            request=payload,
            status="SUBMITTED",
        )

        try:
            response = self._client.request(
                OrderCreate(accountID=self._client.account_id, data=payload)
            )
        except Exception as e:
            logger.error(f"Order submission failed: {e}")
            self._client.journal.record_order(
                strategy=req.strategy,
                magic_id=magic_id,
                trade_uuid=trade_uuid,
                instrument=req.instrument,
                side=req.side,
                units=signed_units,
                entry_price_req=None,
                sl_price=req.sl_price,
                tp_price=req.tp_price,
                trailing_distance=req.trailing_stop_distance,
                request=payload,
                response={"error": str(e)},
                status="REJECTED",
            )
            raise

        result = self._parse_order_response(trade_uuid, response)

        self._client.journal.record_order(
            strategy=req.strategy,
            magic_id=magic_id,
            trade_uuid=trade_uuid,
            instrument=req.instrument,
            side=req.side,
            units=signed_units,
            entry_price_req=None,
            sl_price=req.sl_price,
            tp_price=req.tp_price,
            trailing_distance=req.trailing_stop_distance,
            request=payload,
            response=response,
            oanda_order_id=result.oanda_order_id,
            oanda_trade_id=result.oanda_trade_id,
            entry_price_fill=result.fill_price,
            status=result.status,
        )

        return result

    # ------------------------------------------------------------------
    # Modify / close
    # ------------------------------------------------------------------

    def modify_trailing_stop(
        self,
        trade_id: str,
        new_trail_distance: float,
    ) -> dict[str, Any]:
        """Update the server-side trailing stop distance on an open trade.

        Args:
            trade_id: OANDA trade id (string).
            new_trail_distance: New distance in price units.

        Returns:
            Raw OANDA response.
        """
        payload = {
            "trailingStopLoss": {
                "distance": str(new_trail_distance),
                "timeInForce": "GTC",
            }
        }
        return self._client.request(
            TradeCRCDO(
                accountID=self._client.account_id,
                tradeID=trade_id,
                data=payload,
            )
        )

    def close_trade(
        self,
        trade_id: str,
        units: int | Literal["ALL"] = "ALL",
    ) -> dict[str, Any]:
        """Close all or part of an open trade (FIFO constraints apply in US).

        Args:
            trade_id: OANDA trade id.
            units: ``"ALL"`` or integer count to partially close. Per CLAUDE.md
                §OANDA Gotcha #9, prefer full close + re-enter in US accounts.

        Returns:
            Raw OANDA response.
        """
        data: dict[str, Any] = {"units": units if units == "ALL" else str(units)}
        return self._client.request(
            TradeClose(
                accountID=self._client.account_id,
                tradeID=trade_id,
                data=data,
            )
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _signed_units(req: OrderRequest, spec: InstrumentSpec) -> int:
        """Convert (side, units) into OANDA's signed units convention.

        Accepts either signed or unsigned input ``req.units`` and returns the
        properly-signed integer per ``req.side``.
        """
        abs_units = abs(req.units)
        if abs_units < spec.minimum_trade_size:
            raise ValueError(
                f"Requested units {abs_units} below minimum trade size "
                f"{spec.minimum_trade_size} for {req.instrument}"
            )
        return abs_units if req.side == "LONG" else -abs_units

    def _build_order_payload(
        self,
        *,
        req: OrderRequest,
        spec: InstrumentSpec,
        signed_units: int,
        magic_id: str,
    ) -> dict[str, Any]:
        """Build the JSON body for a market order with SL/TP/trail + magic-id."""
        order: dict[str, Any] = {
            "type": "MARKET",
            "instrument": req.instrument,
            "units": str(signed_units),
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
            "clientExtensions": {
                "id": magic_id,
                "tag": req.strategy,
                "comment": MAGIC_NUMBER,
            },
        }
        if req.sl_price is not None:
            order["stopLossOnFill"] = {
                "price": f"{spec.round_price(req.sl_price):.{spec.display_precision}f}",
                "timeInForce": "GTC",
            }
        if req.tp_price is not None:
            order["takeProfitOnFill"] = {
                "price": f"{spec.round_price(req.tp_price):.{spec.display_precision}f}",
                "timeInForce": "GTC",
            }
        if req.trailing_stop_distance is not None:
            order["trailingStopLossOnFill"] = {
                "distance": f"{req.trailing_stop_distance:.{spec.display_precision}f}",
                "timeInForce": "GTC",
            }
        return {"order": order}

    @staticmethod
    def _parse_order_response(
        trade_uuid: str, response: dict[str, Any]
    ) -> OrderResult:
        """Extract status + ids from an OrderCreate response."""
        create_tx = response.get("orderCreateTransaction") or {}
        fill_tx = response.get("orderFillTransaction")
        cancel_tx = response.get("orderCancelTransaction")

        if fill_tx is not None:
            return OrderResult(
                trade_uuid=trade_uuid,
                oanda_order_id=create_tx.get("id"),
                oanda_trade_id=fill_tx.get("tradeOpened", {}).get("tradeID"),
                fill_price=float(fill_tx["price"]) if "price" in fill_tx else None,
                status="FILLED",
                raw_response=response,
            )
        if cancel_tx is not None:
            return OrderResult(
                trade_uuid=trade_uuid,
                oanda_order_id=create_tx.get("id"),
                oanda_trade_id=None,
                fill_price=None,
                status="CANCELED",
                raw_response=response,
            )
        return OrderResult(
            trade_uuid=trade_uuid,
            oanda_order_id=create_tx.get("id"),
            oanda_trade_id=None,
            fill_price=None,
            status="PENDING",
            raw_response=response,
        )
