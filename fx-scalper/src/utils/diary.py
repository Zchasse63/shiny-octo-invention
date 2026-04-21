"""Event log for project memory.

Every significant action — backtest run, config change, deploy, incident,
decision — appends a JSONL record to ``logs/events.jsonl``. The file is the
single source of truth for JOURNAL.md (rendered by ``scripts/render_journal.py``).

Usage:

    from src.utils.diary import log_event

    log_event(
        "backtest_run",
        run_id="r-0001",
        strategy="bb_rsi_mr",
        params={"bb_length": 20, "bb_std": 2.0},
        metrics={"sharpe": 1.2, "dd": 0.08},
    )

Events are write-once and ordered by timestamp. Do NOT delete or edit the
``events.jsonl`` file directly — doing so breaks the JOURNAL audit chain.
If a recorded event was wrong, append a correction event, don't rewrite.
"""

from __future__ import annotations

import json
import subprocess
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


def _project_root() -> Path:
    """Walk up from this file to find the fx-scalper project root."""
    return Path(__file__).resolve().parent.parent.parent


def _events_path() -> Path:
    path = _project_root() / "logs" / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _git_sha() -> str | None:
    """Current git HEAD SHA, or None if git/repo unavailable."""
    try:
        out = subprocess.run(
            ["git", "-C", str(_project_root().parent), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def log_event(kind: str, **fields: Any) -> None:
    """Append a structured event to ``logs/events.jsonl``.

    Args:
        kind: Event type. Known kinds (use these when possible so the renderer
            can format them richly):

            * ``decision`` — architectural / scope decision. Fields: ``title``,
              ``adr`` (optional ADR id), ``rationale``.
            * ``backtest_run`` — one run of ``scripts/run_backtest.py``.
              Fields: ``run_id``, ``strategy``, ``params``, ``metrics``,
              ``data_range``, ``walk_forward_split``, ``verdict``.
            * ``config_change`` — a value in ``config/settings.py`` changed.
              Fields: ``file``, ``old``, ``new``, ``reason``.
            * ``paper_start`` / ``paper_stop`` — paper trading session.
              Fields: ``session_id``, ``strategy``, plus context.
            * ``live_start`` / ``live_stop`` — live trading session. Same.
            * ``incident`` — production anomaly. Fields: ``severity``,
              ``summary``, ``resolution``, ``link`` (optional).
            * ``learning`` — a new thing learned that should be remembered.
              Fields: ``title``, ``detail``.
            * ``risk_event`` — a circuit breaker tripped. Fields mirror the
              RiskGuard context.

        **fields: Arbitrary structured data for this event.
    """
    record: dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat(timespec="microseconds"),
        "kind": kind,
    }
    sha = _git_sha()
    if sha:
        record["git_sha"] = sha
    record.update(fields)
    line = json.dumps(record, default=str, sort_keys=True) + "\n"
    with _LOCK, _events_path().open("a", encoding="utf-8") as f:
        f.write(line)


def read_events() -> list[dict[str, Any]]:
    """Return all logged events in chronological order."""
    path = _events_path()
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # Ignore corrupt lines rather than crashing renderer.
    return events
