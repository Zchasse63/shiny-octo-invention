"""Strategy ABC.

All strategies consume a closed-bar DataFrame and emit at most one :class:`Signal`
per instrument per call. Sizing and risk are applied upstream — strategies
never call the broker directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from src.oanda.orders import OrderRequest

Side = Literal["LONG", "SHORT"]


@dataclass(frozen=True, slots=True)
class Signal:
    """A strategy's intent to enter a new trade.

    Attributes:
        strategy: Strategy name — also used as OANDA ``clientExtensions.tag``.
        instrument: OANDA instrument (e.g. ``"EUR_USD"``).
        side: ``"LONG"`` or ``"SHORT"``.
        units: Integer unit count (positive). The order client signs based on side.
        sl_price: Stop loss as an absolute price.
        tp_price: Take profit as an absolute price, or None.
        trailing_stop_distance: Server-side trail distance in price units,
            or None to manage client-side via :mod:`src.live.trailing`.
    """

    strategy: str
    instrument: str
    side: Side
    units: int
    sl_price: float | None
    tp_price: float | None
    trailing_stop_distance: float | None = None

    def to_order_request(self) -> OrderRequest:
        """Convert to an :class:`OrderRequest` the order client can submit."""
        return OrderRequest(
            strategy=self.strategy,
            instrument=self.instrument,
            side=self.side,
            units=self.units,
            sl_price=self.sl_price,
            tp_price=self.tp_price,
            trailing_stop_distance=self.trailing_stop_distance,
        )


class Strategy(ABC):
    """Abstract base. Concrete strategies implement :meth:`generate_signal`."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier — used in journal + order tagging."""

    @abstractmethod
    def generate_signal(
        self,
        *,
        instrument: str,
        candles: pd.DataFrame,
    ) -> Signal | None:
        """Return a new :class:`Signal` for ``instrument``, or None.

        Args:
            instrument: OANDA instrument name.
            candles: DataFrame of *closed* bars. Never use the last row if it's
                the forming bar — caller has already filtered.

        Returns:
            Signal if entry conditions met, else None.
        """
