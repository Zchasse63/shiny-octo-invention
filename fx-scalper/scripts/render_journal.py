"""Render JOURNAL.md from ``logs/events.jsonl`` + ``docs/journal_manual.md``.

JOURNAL.md is **generated** — do not hand-edit. For narrative context that
can't be derived from structured events, add entries to ``docs/journal_manual.md``
in the expected format and they will be merged in.

Run manually, or (automatically) via the ``pre-commit`` hook.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.diary import read_events  # noqa: E402


def _forex_root() -> Path:
    """The outer repo root (contains CLAUDE.md, JOURNAL.md, fx-scalper/)."""
    return PROJECT_ROOT.parent


def _load_manual_overlay() -> str:
    path = PROJECT_ROOT / "docs" / "journal_manual.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").rstrip() + "\n"


def _group_by_date(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ev in events:
        ts = ev.get("ts", "")
        date = ts.split("T", 1)[0] if "T" in ts else "unknown"
        out[date].append(ev)
    return out


def _fmt_kv(d: dict[str, Any]) -> str:
    if not d:
        return "_(no data)_"
    return ", ".join(f"`{k}={v}`" for k, v in d.items())


def _render_event(ev: dict[str, Any]) -> str:
    kind = ev.get("kind", "unknown")
    ts = ev.get("ts", "")
    time_part = ts.split("T", 1)[1][:8] if "T" in ts else ""
    sha = ev.get("git_sha")
    sha_suffix = f" `({sha})`" if sha else ""

    match kind:
        case "decision":
            title = ev.get("title", "Untitled decision")
            adr = ev.get("adr")
            adr_suffix = f" — see `DECISIONS/{adr}.md`" if adr else ""
            rationale = ev.get("rationale", "_(no rationale given)_")
            return (
                f"- **{time_part} DECISION{sha_suffix}: {title}**{adr_suffix}  \n"
                f"  {rationale}"
            )
        case "backtest_run":
            run_id = ev.get("run_id", "?")
            strategy = ev.get("strategy", "?")
            verdict = ev.get("verdict", "unclassified")
            metrics = ev.get("metrics", {})
            m = " | ".join(f"{k}={v}" for k, v in metrics.items())
            return (
                f"- **{time_part} BACKTEST `{run_id}` [{strategy}] → {verdict}**{sha_suffix}  \n"
                f"  {m}"
            )
        case "config_change":
            file = ev.get("file", "?")
            reason = ev.get("reason", "(no reason given)")
            diff = []
            old = ev.get("old", {})
            new = ev.get("new", {})
            for k in set(old) | set(new):
                diff.append(f"`{k}`: `{old.get(k, '∅')}` → `{new.get(k, '∅')}`")
            diff_str = "; ".join(diff)
            return (
                f"- **{time_part} CONFIG `{file}`**{sha_suffix}  \n"
                f"  {diff_str} — {reason}"
            )
        case "paper_start" | "paper_stop" | "live_start" | "live_stop":
            session_id = ev.get("session_id", "?")
            strategy = ev.get("strategy", "?")
            return (
                f"- **{time_part} {kind.upper()} session=`{session_id}` "
                f"strategy=`{strategy}`**{sha_suffix}"
            )
        case "incident":
            severity = ev.get("severity", "?")
            summary = ev.get("summary", "?")
            resolution = ev.get("resolution")
            resolution_suffix = f"  \n  Resolution: {resolution}" if resolution else ""
            return (
                f"- **{time_part} INCIDENT [{severity}]: {summary}**{sha_suffix}"
                f"{resolution_suffix}"
            )
        case "learning":
            title = ev.get("title", "Untitled learning")
            detail = ev.get("detail", "")
            return f"- **{time_part} LEARNING: {title}**{sha_suffix}  \n  {detail}"
        case "risk_event":
            breaker = ev.get("breaker", "?")
            state = ev.get("state", "?")
            detail = ev.get("detail", "")
            return (
                f"- **{time_part} RISK `{breaker}` → {state}**{sha_suffix}  \n"
                f"  {detail}"
            )
        case _:
            keys = {k: v for k, v in ev.items() if k not in {"ts", "kind", "git_sha"}}
            return f"- **{time_part} {kind.upper()}**{sha_suffix} — {_fmt_kv(keys)}"


def render() -> str:
    events = read_events()
    grouped = _group_by_date(events)

    lines: list[str] = []
    lines.append("# JOURNAL")
    lines.append("")
    lines.append(
        "Auto-generated from `fx-scalper/logs/events.jsonl` and "
        "`fx-scalper/docs/journal_manual.md`. "
        "**Do not hand-edit this file** — regenerate with "
        "`python fx-scalper/scripts/render_journal.py` (the pre-commit hook also runs this)."
    )
    lines.append("")
    lines.append(
        f"Last rendered: `{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}` UTC  \n"
        f"Events logged: **{len(events)}**"
    )
    lines.append("")

    manual = _load_manual_overlay()
    if manual:
        lines.append("## Narrative overlay")
        lines.append("")
        lines.append("_From `fx-scalper/docs/journal_manual.md` — human-authored context._")
        lines.append("")
        lines.append(manual)
        lines.append("")

    lines.append("## Event log")
    lines.append("")

    if not events:
        lines.append("_(no events yet — run a script that calls `src.utils.diary.log_event`)_")
    else:
        for date in sorted(grouped.keys(), reverse=True):
            lines.append(f"### {date}")
            lines.append("")
            for ev in grouped[date]:
                lines.append(_render_event(ev))
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    target = _forex_root() / "JOURNAL.md"
    target.write_text(render(), encoding="utf-8")
    print(f"Wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
