"""Registry of backtest runs — SQLite-backed + JOURNAL event emission.

Every ``scripts/run_backtest.py`` invocation calls :func:`record_run` with
its params + metrics + verdict. The row lands in ``journal.db`` and also
emits a ``backtest_run`` event to ``logs/events.jsonl`` for JOURNAL.md
rendering.

Use :func:`query_runs` from research notebooks to slice candidates.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config.secrets import get_journal_db_path
from src.backtest.metrics import BacktestMetrics
from src.utils.diary import log_event


def generate_run_id(strategy: str) -> str:
    """Create a stable, sortable run id: ``{strategy}-{YYYYMMDDThhmmss}-{uuid8}``."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    short = uuid.uuid4().hex[:8]
    return f"{strategy}-{stamp}-{short}"


@contextmanager
def _connect(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    p = db_path or get_journal_db_path()
    conn = sqlite3.connect(str(p), timeout=5.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def record_run(
    *,
    run_id: str,
    strategy: str,
    params: dict[str, Any],
    data_range: tuple[str, str],
    metrics: BacktestMetrics,
    walk_forward_split: str | None = None,
    verdict: str = "unclassified",
    notes: str | None = None,
    artifacts_dir: Path | None = None,
    db_path: Path | None = None,
) -> None:
    """Persist one backtest run + emit a JOURNAL event.

    Args:
        run_id: Stable identifier — see :func:`generate_run_id`.
        strategy: Strategy name (e.g. ``bb_rsi_mr``).
        params: Strategy parameters as a JSON-serializable dict.
        data_range: ``(start_iso, end_iso)`` tuple of the backtest window.
        metrics: Computed :class:`BacktestMetrics`.
        walk_forward_split: Optional "train/test" split label.
        verdict: ``"candidate" | "rejected" | "fragile" | "unclassified"``.
        notes: Free-form notes.
        artifacts_dir: Path to the per-run artifacts directory (equity curve,
            trades.parquet, plots). Stored as a string.
        db_path: Override the default journal DB path.
    """
    params_json = json.dumps(params, sort_keys=True, default=str)
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO backtest_runs
              (run_id, ts_utc, git_sha, strategy, params_json,
               data_range_start, data_range_end, walk_forward_split,
               sharpe, sortino, profit_factor, win_rate, total_trades,
               max_drawdown_pct, expectancy_usd, verdict, notes, artifacts_dir)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                datetime.now(UTC).isoformat(timespec="microseconds"),
                None,  # git_sha captured by diary.log_event below.
                strategy,
                params_json,
                data_range[0],
                data_range[1],
                walk_forward_split,
                metrics.sharpe,
                metrics.sortino,
                metrics.profit_factor,
                metrics.win_rate,
                metrics.total_trades,
                metrics.max_drawdown_pct,
                metrics.expectancy_usd,
                verdict,
                notes,
                str(artifacts_dir) if artifacts_dir else None,
            ),
        )

    log_event(
        "backtest_run",
        run_id=run_id,
        strategy=strategy,
        params=params,
        metrics=asdict(metrics),
        data_range=f"{data_range[0]}..{data_range[1]}",
        walk_forward_split=walk_forward_split,
        verdict=verdict,
        artifacts_dir=str(artifacts_dir) if artifacts_dir else None,
    )


def query_runs(
    *,
    strategy: str | None = None,
    verdict: str | None = None,
    min_sharpe: float | None = None,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Query stored runs with simple filters."""
    clauses: list[str] = []
    args: list[Any] = []
    if strategy:
        clauses.append("strategy = ?")
        args.append(strategy)
    if verdict:
        clauses.append("verdict = ?")
        args.append(verdict)
    if min_sharpe is not None:
        clauses.append("sharpe >= ?")
        args.append(min_sharpe)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"SELECT * FROM backtest_runs {where} ORDER BY ts_utc DESC"
    with _connect(db_path) as conn:
        return [dict(r) for r in conn.execute(sql, args).fetchall()]
