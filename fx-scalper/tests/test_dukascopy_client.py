"""Tests for the in-house Dukascopy downloader (URL build + parser)."""

from __future__ import annotations

import struct
from datetime import UTC, date, datetime
from lzma import FORMAT_XZ, LZMACompressor
from unittest.mock import MagicMock, patch

from src.backtest.dukascopy_client import (
    _build_url,
    _decompress,
    _parse_records,
    _price_scale,
    fetch_day_ticks,
)


def test_url_uses_zero_indexed_month() -> None:
    """Dukascopy months are 0-indexed: Jan = 00, Dec = 11."""
    url = _build_url("EURUSD", date(2024, 1, 2), 10)
    assert "/2024/00/02/10h_ticks.bi5" in url

    url = _build_url("EURUSD", date(2024, 12, 15), 8)
    assert "/2024/11/15/08h_ticks.bi5" in url


def test_url_has_datafeed_host() -> None:
    url = _build_url("EURUSD", date(2024, 1, 2), 10)
    assert url.startswith("https://datafeed.dukascopy.com/datafeed/")


def test_price_scale_jpy_vs_non_jpy() -> None:
    assert _price_scale("USDJPY") == 1000.0
    assert _price_scale("EURJPY") == 1000.0
    assert _price_scale("EURUSD") == 100000.0
    assert _price_scale("GBPUSD") == 100000.0


def test_decompress_empty_returns_empty() -> None:
    assert _decompress(b"") == b""


def test_decompress_roundtrip() -> None:
    payload = b"some binary payload \x00\xff\xde\xad\xbe\xef" * 4
    comp = LZMACompressor(format=FORMAT_XZ)
    blob = comp.compress(payload) + comp.flush()
    assert _decompress(blob) == payload


def test_parse_records_produces_expected_values() -> None:
    """Build a synthetic hour of records and verify scaling + timestamps."""
    records = [
        (500, 108050, 108040, 1.5, 2.0),  # 0.5 sec into hour
        (1500, 108060, 108050, 1.0, 1.0),  # 1.5 sec into hour
    ]
    buf = b"".join(
        struct.pack("!IIIff", ms, ai, bi, av, bv) for ms, ai, bi, av, bv in records
    )
    d = date(2024, 1, 2)
    hour = 10
    parsed = list(_parse_records(buf, "EURUSD", d, hour))
    assert len(parsed) == 2

    t0, ask0, bid0, av0, bv0 = parsed[0]
    assert t0 == datetime(2024, 1, 2, 10, 0, 0, 500_000, tzinfo=UTC)
    assert ask0 == 108050 / 100000  # 1.08050
    assert bid0 == 108040 / 100000
    assert av0 == 1.5
    assert bv0 == 2.0

    t1, _, _, _, _ = parsed[1]
    assert t1 == datetime(2024, 1, 2, 10, 0, 1, 500_000, tzinfo=UTC)


def test_parse_records_usdjpy_scaling() -> None:
    records = [(0, 155_200, 155_180, 1.0, 1.0)]  # USD/JPY = 155.200 / 155.180
    buf = struct.pack("!IIIff", *records[0])
    parsed = list(_parse_records(buf, "USDJPY", date(2024, 1, 2), 10))
    _, ask, bid, _, _ = parsed[0]
    assert abs(ask - 155.200) < 1e-9
    assert abs(bid - 155.180) < 1e-9


def test_fetch_day_ticks_handles_empty_hours() -> None:
    """Weekend boundary: empty (200 OK with 0 bytes) shouldn't crash or warn."""
    empty_response = MagicMock()
    empty_response.status_code = 200
    empty_response.content = b""

    with patch(
        "src.backtest.dukascopy_client.requests.get",
        return_value=empty_response,
    ):
        df = fetch_day_ticks("EURUSD", date(2024, 1, 6))  # Saturday
    assert df.empty
