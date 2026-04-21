"""Indicator engine tests — verify columns, non-null coverage, TA-Lib detection."""

from __future__ import annotations

import pandas as pd

from src.indicators.engine import (
    add_adx,
    add_atr,
    add_bbands,
    add_ema,
    add_indicators,
    add_rsi,
    talib_available,
)


class TestAddRsi:
    def test_adds_column(self, synthetic_m5_candles: pd.DataFrame) -> None:
        out = add_rsi(synthetic_m5_candles, length=14)
        assert "rsi_14" in out.columns
        # RSI is bounded 0-100 and should be non-null after warmup.
        tail = out["rsi_14"].iloc[50:]
        assert tail.notna().all()
        assert (tail >= 0).all() and (tail <= 100).all()


class TestAddBbands:
    def test_adds_three_columns(self, synthetic_m5_candles: pd.DataFrame) -> None:
        out = add_bbands(synthetic_m5_candles, length=20, std=2.0)
        for col in ("bb_lower_20_2.0", "bb_middle_20_2.0", "bb_upper_20_2.0"):
            assert col in out.columns
        # Upper > middle > lower after warmup.
        tail = out.iloc[50:]
        assert (tail["bb_upper_20_2.0"] >= tail["bb_middle_20_2.0"]).all()
        assert (tail["bb_middle_20_2.0"] >= tail["bb_lower_20_2.0"]).all()


class TestAddAtr:
    def test_atr_is_positive(self, synthetic_m5_candles: pd.DataFrame) -> None:
        out = add_atr(synthetic_m5_candles, length=14)
        assert "atr_14" in out.columns
        tail = out["atr_14"].iloc[50:]
        assert (tail > 0).all()


class TestAddAdx:
    def test_adx_is_bounded(self, synthetic_m5_candles: pd.DataFrame) -> None:
        out = add_adx(synthetic_m5_candles, length=14)
        assert "adx_14" in out.columns
        tail = out["adx_14"].iloc[100:]
        assert (tail >= 0).all() and (tail <= 100).all()


class TestAddEma:
    def test_ema_tracks_price(self, synthetic_m5_candles: pd.DataFrame) -> None:
        out = add_ema(synthetic_m5_candles, length=50)
        assert "ema_50" in out.columns
        tail = out["ema_50"].iloc[200:]
        price_tail = out["mid_close"].iloc[200:]
        # EMA should be within ±5% of price (generous bound).
        diff = (tail - price_tail).abs() / price_tail
        assert (diff < 0.05).all()


class TestAddIndicators:
    def test_batch_wires_up_known_names(
        self, synthetic_m5_candles: pd.DataFrame
    ) -> None:
        out = add_indicators(
            synthetic_m5_candles,
            indicators=("rsi", "bbands", "atr", "adx"),
        )
        assert "rsi_14" in out.columns
        assert "bb_middle_20_2.0" in out.columns
        assert "atr_14" in out.columns
        assert "adx_14" in out.columns


def test_talib_available() -> None:
    """Should return True when TA-Lib is installed (via brew + pip).

    We don't assert True — the test is informational. If it returns False,
    the indicator compute is still correct, just slower.
    """
    # Just smoke-test it doesn't raise.
    _ = talib_available()
