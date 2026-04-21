"""SQLite-backed audit journal for every OANDA request/response and trade event.

Per CLAUDE.md §Code Standards: *every* order request AND response must be
logged here. The journal is the single source of truth when reconciling
paper/live results against backtests.

Schema (kept deliberately simple — we query with pandas):

* ``api_calls`` — raw request/response for any OANDA endpoint
* ``orders`` — order intents we sent, with magic-number + strategy tags
* ``fills`` — reported fills/transactions from OANDA
* ``trades`` — lifecycle: open→close with PnL
* ``risk_events`` — every circuit-breaker trip
* ``equity_snapshots`` — periodic NAV/balance/margin snapshots
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS api_calls (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc          TEXT NOT NULL,
    endpoint        TEXT NOT NULL,
    method          TEXT NOT NULL,
    request_json    TEXT,
    response_json   TEXT,
    status_code     INTEGER,
    error_code      TEXT,
    duration_ms     INTEGER
);
CREATE INDEX IF NOT EXISTS idx_api_calls_ts ON api_calls(ts_utc);
CREATE INDEX IF NOT EXISTS idx_api_calls_endpoint ON api_calls(endpoint);

CREATE TABLE IF NOT EXISTS orders (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc              TEXT NOT NULL,
    strategy            TEXT NOT NULL,
    magic_id            TEXT NOT NULL,
    trade_uuid          TEXT NOT NULL,
    instrument          TEXT NOT NULL,
    side                TEXT NOT NULL CHECK(side IN ('LONG','SHORT')),
    units               INTEGER NOT NULL,
    entry_price_req     REAL,
    entry_price_fill    REAL,
    sl_price            REAL,
    tp_price            REAL,
    trailing_distance   REAL,
    oanda_order_id      TEXT,
    oanda_trade_id      TEXT,
    request_json        TEXT NOT NULL,
    response_json       TEXT,
    status              TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_orders_ts ON orders(ts_utc);
CREATE INDEX IF NOT EXISTS idx_orders_strategy ON orders(strategy);
CREATE INDEX IF NOT EXISTS idx_orders_trade_uuid ON orders(trade_uuid);

CREATE TABLE IF NOT EXISTS fills (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc              TEXT NOT NULL,
    oanda_transaction_id TEXT,
    oanda_trade_id      TEXT,
    instrument          TEXT NOT NULL,
    units               INTEGER NOT NULL,
    price               REAL NOT NULL,
    commission          REAL,
    financing           REAL,
    pl                  REAL,
    reason              TEXT,
    raw_json            TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fills_ts ON fills(ts_utc);
CREATE INDEX IF NOT EXISTS idx_fills_oanda_trade_id ON fills(oanda_trade_id);

CREATE TABLE IF NOT EXISTS trades (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_uuid          TEXT NOT NULL UNIQUE,
    strategy            TEXT NOT NULL,
    instrument          TEXT NOT NULL,
    side                TEXT NOT NULL,
    units               INTEGER NOT NULL,
    ts_open_utc         TEXT NOT NULL,
    ts_close_utc        TEXT,
    entry_price         REAL NOT NULL,
    exit_price          REAL,
    sl_price            REAL,
    tp_price            REAL,
    pl_realized         REAL,
    commission_total    REAL,
    financing_total     REAL,
    close_reason        TEXT
);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);
CREATE INDEX IF NOT EXISTS idx_trades_ts_open ON trades(ts_open_utc);

CREATE TABLE IF NOT EXISTS risk_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc      TEXT NOT NULL,
    breaker     TEXT NOT NULL,
    state       TEXT NOT NULL,
    detail      TEXT,
    context_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_risk_events_ts ON risk_events(ts_utc);

CREATE TABLE IF NOT EXISTS equity_snapshots (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc              TEXT NOT NULL,
    balance             REAL NOT NULL,
    nav                 REAL NOT NULL,
    margin_used         REAL NOT NULL,
    margin_available    REAL NOT NULL,
    open_position_count INTEGER NOT NULL,
    unrealized_pl       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity_snapshots(ts_utc);

CREATE TABLE IF NOT EXISTS backtest_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT NOT NULL UNIQUE,
    ts_utc              TEXT NOT NULL,
    git_sha             TEXT,
    strategy            TEXT NOT NULL,
    params_json         TEXT NOT NULL,
    data_range_start    TEXT NOT NULL,
    data_range_end      TEXT NOT NULL,
    walk_forward_split  TEXT,
    sharpe              REAL,
    sortino             REAL,
    profit_factor       REAL,
    win_rate            REAL,
    total_trades        INTEGER,
    max_drawdown_pct    REAL,
    expectancy_usd      REAL,
    verdict             TEXT,
    notes               TEXT,
    artifacts_dir       TEXT
);
CREATE INDEX IF NOT EXISTS idx_backtest_runs_ts ON backtest_runs(ts_utc);
CREATE INDEX IF NOT EXISTS idx_backtest_runs_strategy ON backtest_runs(strategy);
CREATE INDEX IF NOT EXISTS idx_backtest_runs_verdict ON backtest_runs(verdict);
"""


def _utcnow_iso() -> str:
    """Current UTC timestamp as ISO-8601 string with microseconds."""
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _dumps(obj: Any) -> str | None:
    """JSON-serialize any obj; returns None for None (so SQLite stores NULL)."""
    if obj is None:
        return None
    return json.dumps(obj, default=str, sort_keys=True)


class Journal:
    """Thread-safe SQLite journal.

    One instance per process; share via dependency injection. Wrap every
    OANDA call and every risk event.

    Args:
        db_path: Path to the SQLite file. Parent dir will be created.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection with WAL mode and row dicts."""
        conn = sqlite3.connect(
            str(self._db_path),
            timeout=5.0,
            isolation_level=None,  # Autocommit; we manage txn explicitly.
        )
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Public recording methods
    # ------------------------------------------------------------------

    def record_api_call(
        self,
        *,
        endpoint: str,
        method: str,
        request: Any,
        response: Any,
        status_code: int | None = None,
        error_code: str | None = None,
        duration_ms: int | None = None,
    ) -> None:
        """Log a single OANDA API call (request + response)."""
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO api_calls
                  (ts_utc, endpoint, method, request_json, response_json,
                   status_code, error_code, duration_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _utcnow_iso(),
                    endpoint,
                    method,
                    _dumps(request),
                    _dumps(response),
                    status_code,
                    error_code,
                    duration_ms,
                ),
            )

    def record_order(
        self,
        *,
        strategy: str,
        magic_id: str,
        trade_uuid: str,
        instrument: str,
        side: str,
        units: int,
        entry_price_req: float | None,
        sl_price: float | None,
        tp_price: float | None,
        trailing_distance: float | None,
        request: Any,
        response: Any = None,
        oanda_order_id: str | None = None,
        oanda_trade_id: str | None = None,
        entry_price_fill: float | None = None,
        status: str = "SUBMITTED",
    ) -> None:
        """Log an order submission with magic-number + trade UUID tagging."""
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO orders
                  (ts_utc, strategy, magic_id, trade_uuid, instrument, side, units,
                   entry_price_req, entry_price_fill, sl_price, tp_price,
                   trailing_distance, oanda_order_id, oanda_trade_id,
                   request_json, response_json, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _utcnow_iso(),
                    strategy,
                    magic_id,
                    trade_uuid,
                    instrument,
                    side,
                    units,
                    entry_price_req,
                    entry_price_fill,
                    sl_price,
                    tp_price,
                    trailing_distance,
                    oanda_order_id,
                    oanda_trade_id,
                    _dumps(request),
                    _dumps(response),
                    status,
                ),
            )

    def record_fill(
        self,
        *,
        oanda_transaction_id: str | None,
        oanda_trade_id: str | None,
        instrument: str,
        units: int,
        price: float,
        commission: float | None,
        financing: float | None,
        pl: float | None,
        reason: str | None,
        raw: Any,
    ) -> None:
        """Log a fill/transaction reported by OANDA."""
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO fills
                  (ts_utc, oanda_transaction_id, oanda_trade_id, instrument,
                   units, price, commission, financing, pl, reason, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _utcnow_iso(),
                    oanda_transaction_id,
                    oanda_trade_id,
                    instrument,
                    units,
                    price,
                    commission,
                    financing,
                    pl,
                    reason,
                    _dumps(raw),
                ),
            )

    def record_trade_open(
        self,
        *,
        trade_uuid: str,
        strategy: str,
        instrument: str,
        side: str,
        units: int,
        entry_price: float,
        sl_price: float | None,
        tp_price: float | None,
    ) -> None:
        """Insert a new row into ``trades`` at open."""
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trades
                  (trade_uuid, strategy, instrument, side, units, ts_open_utc,
                   entry_price, sl_price, tp_price)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade_uuid,
                    strategy,
                    instrument,
                    side,
                    units,
                    _utcnow_iso(),
                    entry_price,
                    sl_price,
                    tp_price,
                ),
            )

    def record_trade_close(
        self,
        *,
        trade_uuid: str,
        exit_price: float,
        pl_realized: float,
        commission_total: float | None,
        financing_total: float | None,
        close_reason: str,
    ) -> None:
        """Update a ``trades`` row at close with realized PnL.

        Raises:
            ValueError: If no row exists for ``trade_uuid``. A silent no-op
                here would let a mis-reconciliation hide unreported closes.
        """
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE trades
                   SET ts_close_utc = ?, exit_price = ?, pl_realized = ?,
                       commission_total = ?, financing_total = ?, close_reason = ?
                 WHERE trade_uuid = ?
                """,
                (
                    _utcnow_iso(),
                    exit_price,
                    pl_realized,
                    commission_total,
                    financing_total,
                    close_reason,
                    trade_uuid,
                ),
            )
            if cursor.rowcount == 0:
                raise ValueError(
                    f"record_trade_close: trade_uuid {trade_uuid!r} not found in "
                    f"trades. record_trade_open must be called first."
                )

    def record_risk_event(
        self,
        *,
        breaker: str,
        state: str,
        detail: str | None,
        context: Any = None,
    ) -> None:
        """Log a circuit-breaker trip or state change."""
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO risk_events (ts_utc, breaker, state, detail, context_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (_utcnow_iso(), breaker, state, detail, _dumps(context)),
            )

    def record_equity_snapshot(
        self,
        *,
        balance: float,
        nav: float,
        margin_used: float,
        margin_available: float,
        open_position_count: int,
        unrealized_pl: float,
    ) -> None:
        """Record NAV/balance snapshot for equity-curve reconstruction."""
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO equity_snapshots
                  (ts_utc, balance, nav, margin_used, margin_available,
                   open_position_count, unrealized_pl)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _utcnow_iso(),
                    balance,
                    nav,
                    margin_used,
                    margin_available,
                    open_position_count,
                    unrealized_pl,
                ),
            )

    # ------------------------------------------------------------------
    # Convenience queries
    # ------------------------------------------------------------------

    def closed_trades_descending(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return the most recent ``limit`` closed trades, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM trades
                 WHERE ts_close_utc IS NOT NULL
                 ORDER BY ts_close_utc DESC
                 LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def consecutive_losses(self) -> int:
        """Count consecutive losing closed trades from most recent going back."""
        trades = self.closed_trades_descending(limit=50)
        count = 0
        for t in trades:
            pl = t.get("pl_realized")
            if pl is None:
                break
            if pl < 0:
                count += 1
            else:
                break
        return count

    def realized_pl_since(self, since_utc_iso: str) -> float:
        """Sum realized PnL for trades closed at/after ``since_utc_iso``."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COALESCE(SUM(pl_realized), 0.0) AS total
                  FROM trades
                 WHERE ts_close_utc >= ?
                """,
                (since_utc_iso,),
            ).fetchone()
            return float(row["total"]) if row else 0.0
