"""SQLite journal tests."""

from __future__ import annotations

from pathlib import Path

from src.utils.journal import Journal


def test_journal_creates_schema(tmp_path: Path) -> None:
    db = tmp_path / "j.db"
    Journal(db)
    assert db.exists()


def test_api_call_roundtrip(tmp_journal: Journal) -> None:
    tmp_journal.record_api_call(
        endpoint="AccountSummary",
        method="GET",
        request={"accountID": "001-001-123"},
        response={"account": {"balance": "500.00"}},
        status_code=200,
        duration_ms=42,
    )
    # No API for reading yet, but SQLite file must have grown.


def test_trade_open_and_close(tmp_journal: Journal) -> None:
    tmp_journal.record_trade_open(
        trade_uuid="uuid-1",
        strategy="bb_rsi_mr",
        instrument="EUR_USD",
        side="LONG",
        units=4629,
        entry_price=1.0800,
        sl_price=1.0780,
        tp_price=1.0820,
    )
    tmp_journal.record_trade_close(
        trade_uuid="uuid-1",
        exit_price=1.0815,
        pl_realized=+6.94,
        commission_total=0.0,
        financing_total=0.0,
        close_reason="TP",
    )
    trades = tmp_journal.closed_trades_descending()
    assert len(trades) == 1
    assert trades[0]["trade_uuid"] == "uuid-1"
    assert trades[0]["close_reason"] == "TP"
    assert trades[0]["pl_realized"] > 0


def test_consecutive_losses_count(tmp_journal: Journal) -> None:
    # Two losses, then one win, then one loss → consec should be 1.
    for i, (reason, pl) in enumerate(
        [("SL", -10.0), ("SL", -10.0), ("TP", +15.0), ("SL", -10.0)]
    ):
        tmp_journal.record_trade_open(
            trade_uuid=f"u{i}",
            strategy="s",
            instrument="EUR_USD",
            side="LONG",
            units=1,
            entry_price=1.0,
            sl_price=None,
            tp_price=None,
        )
        tmp_journal.record_trade_close(
            trade_uuid=f"u{i}",
            exit_price=1.0,
            pl_realized=pl,
            commission_total=0.0,
            financing_total=0.0,
            close_reason=reason,
        )
    assert tmp_journal.consecutive_losses() == 1


def test_record_trade_close_raises_on_missing_uuid(tmp_journal: Journal) -> None:
    """Silent UPDATE would let a missed open slip through reconciliation."""
    import pytest as _pytest

    with _pytest.raises(ValueError, match="not found"):
        tmp_journal.record_trade_close(
            trade_uuid="does-not-exist",
            exit_price=1.0,
            pl_realized=0.0,
            commission_total=0.0,
            financing_total=0.0,
            close_reason="TP",
        )


def test_realized_pl_since(tmp_journal: Journal) -> None:
    tmp_journal.record_trade_open(
        trade_uuid="u1",
        strategy="s",
        instrument="EUR_USD",
        side="LONG",
        units=1,
        entry_price=1.0,
        sl_price=None,
        tp_price=None,
    )
    tmp_journal.record_trade_close(
        trade_uuid="u1",
        exit_price=1.0,
        pl_realized=+25.0,
        commission_total=0.0,
        financing_total=0.0,
        close_reason="TP",
    )
    # "Since the epoch" — should see the +25.
    assert tmp_journal.realized_pl_since("1970-01-01T00:00:00") == 25.0
