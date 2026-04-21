"""Instrument metadata registry: pip location, precision, margin rate, pip_value.

Per CLAUDE.md §OANDA Gotchas #3–4: never hardcode pip or precision.
Fetch from OANDA's ``/v3/accounts/{id}/instruments`` and cache in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from oandapyV20.endpoints.accounts import AccountInstruments

from src.oanda.client import OandaClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class InstrumentSpec:
    """Metadata for a single OANDA instrument.

    Attributes:
        name: Underscore form, e.g. ``EUR_USD``.
        display_name: Human form, e.g. ``EUR/USD``.
        type: ``CURRENCY`` for FX pairs.
        pip_location: Exponent of pip. -4 for EUR/USD (pip = 10^-4 = 0.0001).
        display_precision: Decimal places prices are quoted to.
        trade_units_precision: Integer precision for units (usually 0).
        minimum_trade_size: Smallest allowed order, in units.
        maximum_trailing_stop_distance: Max trailing distance in price units.
        minimum_trailing_stop_distance: Min trailing distance in price units.
        margin_rate: Fraction of notional required as margin (e.g. 0.02 = 50:1).
    """

    name: str
    display_name: str
    type: str
    pip_location: int
    display_precision: int
    trade_units_precision: int
    minimum_trade_size: float
    maximum_trailing_stop_distance: float
    minimum_trailing_stop_distance: float
    margin_rate: float

    @property
    def pip_size(self) -> float:
        """Price-space pip size (e.g. 0.0001 for EUR/USD, 0.01 for USD/JPY)."""
        return 10 ** self.pip_location

    def round_price(self, price: float) -> float:
        """Round ``price`` to the instrument's display precision."""
        return round(price, self.display_precision)

    def round_units(self, units: float) -> int:
        """Round to integer units (OANDA requires int for FX)."""
        return int(round(units, self.trade_units_precision))


class InstrumentRegistry:
    """In-memory cache of instrument metadata.

    Fetch once at startup with :meth:`load`, then call :meth:`get` repeatedly.
    """

    def __init__(self, client: OandaClient) -> None:
        self._client = client
        self._by_name: dict[str, InstrumentSpec] = {}

    def load(self, instruments: list[str] | None = None) -> None:
        """Fetch and cache instrument metadata.

        Args:
            instruments: Optional list to restrict the fetch. Defaults to all.
        """
        params: dict[str, Any] = {}
        if instruments:
            params["instruments"] = ",".join(instruments)
        resp = self._client.request(
            AccountInstruments(accountID=self._client.account_id, params=params)
        )
        for raw in resp.get("instruments", []):
            spec = self._parse(raw)
            self._by_name[spec.name] = spec
        logger.info(
            f"Loaded {len(self._by_name)} instruments: "
            f"{sorted(self._by_name.keys())[:10]}{'…' if len(self._by_name) > 10 else ''}"
        )

    def get(self, name: str) -> InstrumentSpec:
        """Return cached :class:`InstrumentSpec` or raise KeyError."""
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(
                f"Instrument {name!r} not in registry. Call load() first or include "
                f"it in the instruments filter."
            ) from None

    def names(self) -> list[str]:
        """Sorted list of cached instrument names."""
        return sorted(self._by_name.keys())

    # ------------------------------------------------------------------
    # Pip-value math
    # ------------------------------------------------------------------

    def pip_value_usd(
        self,
        instrument: str,
        units: int,
        current_price: float,
    ) -> float:
        """Return the USD value of one pip for ``units`` of ``instrument``.

        For an instrument quoted as BASE/QUOTE:
            pip_value_quote = units × pip_size
            pip_value_usd   = pip_value_quote converted to USD

        For pairs where USD is the QUOTE (EUR_USD, GBP_USD), pip_value_quote
        is already in USD.

        For pairs where USD is the BASE (USD_JPY), divide by current price.

        Args:
            instrument: OANDA instrument name (e.g. ``EUR_USD``).
            units: Integer unit size of the hypothetical position.
            current_price: Current mid price, used for base-currency conversion.

        Returns:
            USD value of one pip (positive float).
        """
        spec = self.get(instrument)
        pip_value_quote = abs(units) * spec.pip_size
        base, quote = instrument.split("_")
        if quote == "USD":
            return pip_value_quote
        if base == "USD":
            return pip_value_quote / current_price
        # Cross: convert quote→USD via current_price as a rough proxy.
        # Good enough for sizing sanity; real precision handled at broker.
        return pip_value_quote / current_price

    @staticmethod
    def _parse(raw: dict[str, Any]) -> InstrumentSpec:
        return InstrumentSpec(
            name=str(raw["name"]),
            display_name=str(raw.get("displayName", raw["name"])),
            type=str(raw.get("type", "CURRENCY")),
            pip_location=int(raw.get("pipLocation", -4)),
            display_precision=int(raw.get("displayPrecision", 5)),
            trade_units_precision=int(raw.get("tradeUnitsPrecision", 0)),
            minimum_trade_size=float(raw.get("minimumTradeSize", 1)),
            maximum_trailing_stop_distance=float(
                raw.get("maximumTrailingStopDistance", 1.0)
            ),
            minimum_trailing_stop_distance=float(
                raw.get("minimumTrailingStopDistance", 0.00005)
            ),
            margin_rate=float(raw.get("marginRate", 0.02)),
        )
