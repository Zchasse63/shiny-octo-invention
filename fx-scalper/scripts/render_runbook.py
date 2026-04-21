"""Render RUNBOOK.md — auto-generated config snapshot + manual operational prose.

The config snapshot is introspected from ``config/settings.py``. The
human-authored procedures live in ``docs/runbook_manual.md`` and are
appended verbatim.
"""

from __future__ import annotations

import importlib
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _forex_root() -> Path:
    return PROJECT_ROOT.parent


def _is_public_constant(name: str, value: Any) -> bool:
    if name.startswith("_"):
        return False
    if callable(value):
        return False
    if isinstance(value, type):
        return False
    return name.isupper()


def _render_settings_section() -> str:
    settings = importlib.import_module("config.settings")
    lines: list[str] = []
    for name in sorted(dir(settings)):
        value = getattr(settings, name)
        if not _is_public_constant(name, value):
            continue
        if isinstance(value, str) and "\n" in value:
            value = value.replace("\n", " ")
        lines.append(f"- `{name}` = `{value!r}`")
    return "\n".join(lines)


def _load_manual_overlay() -> str:
    path = PROJECT_ROOT / "docs" / "runbook_manual.md"
    if not path.exists():
        return "_(no manual procedures yet — add them to `fx-scalper/docs/runbook_manual.md`)_"
    return path.read_text(encoding="utf-8").rstrip() + "\n"


def render() -> str:
    lines: list[str] = []
    lines.append("# RUNBOOK")
    lines.append("")
    lines.append(
        "Partially auto-generated. The **config snapshot** below is rendered "
        "from `fx-scalper/config/settings.py`. The **operational procedures** "
        "section is merged from `fx-scalper/docs/runbook_manual.md`."
    )
    lines.append("")
    lines.append(f"Last rendered: `{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}` UTC")
    lines.append("")
    lines.append("## Config snapshot")
    lines.append("")
    lines.append("_Derived from `config/settings.py`. All risk params + circuit-breaker "
                 "values + session windows._")
    lines.append("")
    lines.append(_render_settings_section())
    lines.append("")
    lines.append("## Operational procedures")
    lines.append("")
    lines.append(_load_manual_overlay())
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    target = _forex_root() / "RUNBOOK.md"
    target.write_text(render(), encoding="utf-8")
    print(f"Wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
