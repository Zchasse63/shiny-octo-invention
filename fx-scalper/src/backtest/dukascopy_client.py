"""Minimal synchronous Dukascopy tick downloader.

Written in-house because duka==0.2.0's asyncio+requests pipeline silently
drops most hours. We need reliable bid/ask data for the Day 3 harness
honest-backtest mandate.

Format reference:
- URL: ``https://datafeed.dukascopy.com/datafeed/{SYM}/{YYYY}/{MM0}/{DD}/{HH}h_ticks.bi5``
  where ``MM0`` is zero-indexed (Jan=00).
- Payload: LZMA-compressed binary, 20-byte records of ``!IIIff`` =
  (ms_into_hour, ask_int, bid_int, ask_volume, bid_volume).
- Prices stored as int × 100000 for non-JPY pairs; int × 1000 for JPY pairs
  (pip location -4 vs -2). Callers must divide by the right factor.
- Missing hour (e.g. weekend) returns HTTP 200 with 0 bytes, not a 404.
"""

from __future__ import annotations

import struct
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from lzma import FORMAT_AUTO, LZMADecompressor, LZMAError

import pandas as pd
import requests

from src.utils.logger import get_logger

logger = get_logger(__name__)


_BASE_URL = "https://datafeed.dukascopy.com/datafeed"
_TIMEOUT_SECONDS = 20.0
_MAX_RETRIES = 3
_BACKOFF_SECONDS = 0.5

# Pairs quoted vs JPY use price scale factor 1000 (pip location -2).
_JPY_QUOTED = frozenset({"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY",
                          "CHFJPY", "NZDJPY"})


def _price_scale(symbol: str) -> float:
    return 1000.0 if symbol in _JPY_QUOTED else 100000.0


@dataclass(frozen=True, slots=True)
class HourFetch:
    """Raw fetch result for one hour bucket."""

    url: str
    status_code: int
    payload: bytes


def _build_url(symbol: str, d: date, hour: int) -> str:
    # Dukascopy month is zero-indexed.
    return (
        f"{_BASE_URL}/{symbol}/{d.year:04d}/{d.month - 1:02d}/"
        f"{d.day:02d}/{hour:02d}h_ticks.bi5"
    )


def fetch_hour(symbol: str, d: date, hour: int) -> HourFetch:
    """Fetch a single hour's compressed bi5 payload with retry + backoff."""
    url = _build_url(symbol, d, hour)
    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            r = requests.get(url, timeout=_TIMEOUT_SECONDS)
            if r.status_code == 200:
                return HourFetch(url=url, status_code=200, payload=r.content)
            if r.status_code in {429, 503}:
                last_error = RuntimeError(f"{r.status_code} on {url}")
                time.sleep(_BACKOFF_SECONDS * (2 ** attempt))
                continue
            # Other non-200s: give up fast.
            return HourFetch(url=url, status_code=r.status_code, payload=b"")
        except requests.RequestException as e:
            last_error = e
            time.sleep(_BACKOFF_SECONDS * (2 ** attempt))
    assert last_error is not None  # unreachable otherwise
    raise last_error


def _decompress(blob: bytes) -> bytes:
    if not blob:
        return b""
    out: list[bytes] = []
    data = blob
    while True:
        dec = LZMADecompressor(FORMAT_AUTO, None, None)
        try:
            piece = dec.decompress(data)
        except LZMAError:
            if out:
                break  # trailing noise; accept what we have.
            raise
        out.append(piece)
        data = dec.unused_data
        if not data:
            break
        if not dec.eof:
            raise LZMAError("Truncated LZMA stream")
    return b"".join(out)


def _parse_records(
    buf: bytes,
    symbol: str,
    d: date,
    hour: int,
) -> Iterator[tuple[datetime, float, float, float, float]]:
    """Yield (ts_utc, ask, bid, ask_vol, bid_vol) tuples."""
    if not buf:
        return
    record_size = 20
    price_scale = _price_scale(symbol)
    hour_base = datetime(d.year, d.month, d.day, hour, 0, tzinfo=UTC)
    n = len(buf) // record_size
    for i in range(n):
        ms_into_hour, ask_int, bid_int, ask_vol, bid_vol = struct.unpack(
            "!IIIff", buf[i * record_size : (i + 1) * record_size]
        )
        ts = hour_base + timedelta(milliseconds=ms_into_hour)
        yield ts, ask_int / price_scale, bid_int / price_scale, ask_vol, bid_vol


def fetch_day_ticks(symbol: str, d: date) -> pd.DataFrame:
    """Fetch all 24 hour-buckets for ``d`` and return a unified tick DataFrame.

    Hours that have no data (weekend boundary) contribute zero rows silently.

    Returns:
        DataFrame with columns ``bid, ask, ask_volume, bid_volume, volume``,
        indexed by tz-aware UTC timestamp. ``volume`` is ``ask_volume + bid_volume``.
    """
    rows: list[tuple[datetime, float, float, float, float]] = []
    missing_hours = 0
    for hour in range(24):
        try:
            fetch = fetch_hour(symbol, d, hour)
        except Exception as e:
            logger.warning(f"{symbol} {d} {hour:02d}h: fetch failed ({e})")
            missing_hours += 1
            continue
        if fetch.status_code != 200 or not fetch.payload:
            continue
        try:
            decompressed = _decompress(fetch.payload)
        except LZMAError as e:
            logger.warning(f"{symbol} {d} {hour:02d}h: decompress failed ({e})")
            missing_hours += 1
            continue
        rows.extend(_parse_records(decompressed, symbol, d, hour))

    if missing_hours:
        logger.warning(f"{symbol} {d}: {missing_hours}/24 hours missing")
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=["time", "ask", "bid", "ask_volume", "bid_volume"],
    )
    df["volume"] = df["ask_volume"] + df["bid_volume"]
    df = df.set_index("time").sort_index()
    return df


def fetch_range(
    symbol: str,
    start: date,
    end: date,
) -> Iterator[tuple[date, pd.DataFrame]]:
    """Yield ``(day, ticks_df)`` pairs for each weekday between ``start`` and ``end`` inclusive."""
    cur = start
    while cur <= end:
        if cur.weekday() < 5:  # Mon-Fri
            ticks = fetch_day_ticks(symbol, cur)
            logger.info(f"{symbol} {cur}: {len(ticks):,} ticks")
            yield cur, ticks
        cur += timedelta(days=1)
