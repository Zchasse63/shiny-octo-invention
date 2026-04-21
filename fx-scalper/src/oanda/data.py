"""Candle fetchers and price utilities.

Per CLAUDE.md §Day 2: use price type ``BA`` (bid + ask) for spread modeling.
Never compute signals on the forming bar — caller is responsible for
slicing to the last *closed* bar (``iloc[-2]`` if ``complete==False`` on last).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

import pandas as pd
from oandapyV20.endpoints.instruments import InstrumentsCandles
from oandapyV20.endpoints.pricing import PricingInfo

from src.oanda.client import OandaClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


Granularity = Literal[
    "S5", "S10", "S15", "S30", "M1", "M2", "M5", "M15", "M30",
    "H1", "H2", "H4", "D", "W",
]


class DataClient:
    """Wraps OANDA candle + pricing endpoints.

    Args:
        client: Authenticated :class:`OandaClient`.
    """

    def __init__(self, client: OandaClient) -> None:
        self._client = client

    def get_candles(
        self,
        instrument: str,
        granularity: Granularity,
        count: int = 500,
        price: str = "BA",
        include_incomplete: bool = False,
    ) -> pd.DataFrame:
        """Fetch OHLCV candles.

        Args:
            instrument: OANDA instrument name (e.g. ``EUR_USD``).
            granularity: OANDA granularity code (``M1``, ``M5``, ``M15``, …).
            count: Number of candles to fetch (1–5000).
            price: ``"M"`` (mid), ``"B"`` (bid), ``"A"`` (ask), ``"BA"`` (both).
                Default is ``"BA"`` for spread-aware signals.
            include_incomplete: If False (default), drop the forming bar.

        Returns:
            DataFrame indexed by UTC timestamp, with columns keyed by price type
            and OHLCV. For ``price="BA"`` the columns include both
            ``bid_open``/``bid_high``/... and ``ask_open``/``ask_high``/... plus
            ``volume`` and ``complete``.
        """
        params = {
            "granularity": granularity,
            "count": count,
            "price": price,
        }
        resp = self._client.request(
            InstrumentsCandles(instrument=instrument, params=params)
        )
        candles = resp.get("candles", [])
        if not candles:
            return _empty_candles_frame(price)

        rows: list[dict[str, Any]] = []
        for c in candles:
            row: dict[str, Any] = {
                "time": _parse_oanda_time(c["time"]),
                "volume": int(c.get("volume", 0)),
                "complete": bool(c.get("complete", False)),
            }
            if "mid" in c:
                _expand(row, c["mid"], prefix="mid_")
            if "bid" in c:
                _expand(row, c["bid"], prefix="bid_")
            if "ask" in c:
                _expand(row, c["ask"], prefix="ask_")
            rows.append(row)

        df = pd.DataFrame(rows).set_index("time").sort_index()
        # Guarantee tz-aware UTC per CLAUDE.md §Code Standards.
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        if not include_incomplete and not df.empty:
            df = df[df["complete"]].copy()

        return df

    def get_current_price(self, instrument: str) -> dict[str, float]:
        """Return the latest bid/ask/mid for ``instrument``.

        Returns:
            Dict with keys ``bid``, ``ask``, ``mid``, ``spread``.
        """
        params = {"instruments": instrument}
        resp = self._client.request(
            PricingInfo(accountID=self._client.account_id, params=params)
        )
        prices = resp.get("prices", [])
        if not prices:
            raise RuntimeError(f"No price returned for {instrument}")
        p = prices[0]
        # OANDA returns a list of prices per liquidity; take the best (first).
        bid = float(p["bids"][0]["price"])
        ask = float(p["asks"][0]["price"])
        mid = (bid + ask) / 2.0
        return {"bid": bid, "ask": ask, "mid": mid, "spread": ask - bid}


def _parse_oanda_time(t: str) -> datetime:
    """Parse OANDA's RFC3339 timestamp into a tz-aware UTC datetime."""
    # OANDA emits e.g. "2024-05-01T14:30:00.000000000Z" — truncate nanos.
    if t.endswith("Z"):
        t = t[:-1]
    # Trim sub-microsecond precision if present.
    if "." in t:
        head, frac = t.split(".", 1)
        frac = frac[:6]
        t = f"{head}.{frac}"
    dt = datetime.fromisoformat(t)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _expand(row: dict[str, Any], candle_price: dict[str, str], prefix: str) -> None:
    """Expand OANDA candle price dict into row keys."""
    row[f"{prefix}open"] = float(candle_price["o"])
    row[f"{prefix}high"] = float(candle_price["h"])
    row[f"{prefix}low"] = float(candle_price["l"])
    row[f"{prefix}close"] = float(candle_price["c"])


def _empty_candles_frame(price: str) -> pd.DataFrame:
    """Produce an empty DataFrame with the right columns for the price type."""
    cols = ["volume", "complete"]
    prefixes: list[str] = []
    if "M" in price:
        prefixes.append("mid_")
    if "B" in price:
        prefixes.append("bid_")
    if "A" in price:
        prefixes.append("ask_")
    for p in prefixes:
        for f in ("open", "high", "low", "close"):
            cols.append(f"{p}{f}")
    df = pd.DataFrame(columns=cols)
    df.index = pd.DatetimeIndex([], tz="UTC", name="time")
    return df
