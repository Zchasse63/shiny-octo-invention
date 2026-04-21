"""Render STRATEGIES.md by introspecting ``src/strategies/``.

For each concrete ``Strategy`` subclass:

* Pull its class docstring (summary of signal logic).
* Pull the fields of its ``*Params`` dataclass (parameter table).
* Mark whether it's implemented or a stub (``NotImplementedError`` in ``generate_signal``).

Manual narrative — "why we picked this strategy", "known failure modes" —
lives in ``docs/strategies_manual.md`` and is appended per-strategy if a
matching ``## strategy-name`` section exists there.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import fields, is_dataclass
from datetime import datetime
from importlib import import_module
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.strategies.base import Strategy  # noqa: E402


def _forex_root() -> Path:
    return PROJECT_ROOT.parent


def _strategies_dir() -> Path:
    return PROJECT_ROOT / "src" / "strategies"


def _find_strategy_modules() -> list[str]:
    out: list[str] = []
    for py in sorted(_strategies_dir().glob("*.py")):
        if py.stem in {"__init__", "base"}:
            continue
        out.append(f"src.strategies.{py.stem}")
    return out


def _is_stub(strategy_cls: type[Strategy]) -> bool:
    """A strategy is a stub if its generate_signal raises NotImplementedError."""
    source_file = Path(sys.modules[strategy_cls.__module__].__file__ or "")
    if not source_file.exists():
        return False
    try:
        tree = ast.parse(source_file.read_text(encoding="utf-8"))
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name != "generate_signal":
            continue
        # Look at only direct raise statements in the body.
        for stmt in node.body:
            if isinstance(stmt, ast.Raise) and isinstance(stmt.exc, ast.Call):
                func = stmt.exc.func
                if isinstance(func, ast.Name) and func.id == "NotImplementedError":
                    return True
    return False


def _find_params_cls(strategy_cls: type[Strategy]) -> type | None:
    mod = sys.modules[strategy_cls.__module__]
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, type) and is_dataclass(obj) and name.endswith("Params"):
            return obj
    return None


def _render_params_table(params_cls: type) -> str:
    if params_cls is None:
        return "_(no Params dataclass found)_"
    lines = [
        "| Parameter | Type | Default | Description |",
        "|---|---|---|---|",
    ]
    for f in fields(params_cls):
        type_repr = getattr(f.type, "__name__", str(f.type))
        default = f.default if f.default is not object() else "—"
        desc = ""
        doc = (params_cls.__doc__ or "")
        # Try to pull the description from the class docstring's Attributes block.
        for line in doc.splitlines():
            stripped = line.strip()
            if stripped.startswith(f"{f.name}:"):
                desc = stripped[len(f.name) + 1 :].strip()
                break
        lines.append(f"| `{f.name}` | `{type_repr}` | `{default!r}` | {desc} |")
    return "\n".join(lines)


def _load_manual_overlay() -> dict[str, str]:
    """Parse docs/strategies_manual.md into {strategy_name: section_text}."""
    path = PROJECT_ROOT / "docs" / "strategies_manual.md"
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    sections: dict[str, str] = {}
    current_name: str | None = None
    current_buf: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if current_name is not None:
                sections[current_name] = "\n".join(current_buf).strip()
            current_name = line[3:].strip()
            current_buf = []
        else:
            current_buf.append(line)
    if current_name is not None:
        sections[current_name] = "\n".join(current_buf).strip()
    return sections


def _discover_strategies() -> list[type[Strategy]]:
    out: list[type[Strategy]] = []
    seen: set[str] = set()
    for mod_name in _find_strategy_modules():
        try:
            mod = import_module(mod_name)
        except Exception:  # pragma: no cover — render-time robustness.
            continue
        for obj_name in dir(mod):
            obj = getattr(mod, obj_name)
            if (
                isinstance(obj, type)
                and issubclass(obj, Strategy)
                and obj is not Strategy
                and obj.__name__ not in seen
            ):
                seen.add(obj.__name__)
                out.append(obj)
    return out


def _render_strategy(cls: type[Strategy], manual: dict[str, str]) -> str:
    name_attr = getattr(cls, "NAME", cls.__name__)
    status = "🚧 stub" if _is_stub(cls) else "✅ implemented"
    doc = (cls.__doc__ or "").strip()
    params_cls = _find_params_cls(cls)

    parts: list[str] = []
    parts.append(f"## {name_attr}  \n**Class:** `{cls.__module__}.{cls.__name__}` — {status}")
    parts.append("")
    if doc:
        parts.append(doc)
        parts.append("")
    parts.append("### Parameters")
    parts.append("")
    parts.append(_render_params_table(params_cls) if params_cls else "_(none)_")
    parts.append("")

    narrative = manual.get(name_attr, "") or manual.get(cls.__name__, "")
    if narrative:
        parts.append("### Notes (human-authored overlay)")
        parts.append("")
        parts.append(narrative)
        parts.append("")
    return "\n".join(parts)


def render() -> str:
    strategies = _discover_strategies()
    manual = _load_manual_overlay()

    lines: list[str] = []
    lines.append("# STRATEGIES")
    lines.append("")
    lines.append(
        "Auto-generated from `fx-scalper/src/strategies/`. "
        "**Do not hand-edit this file** — add narrative to "
        "`fx-scalper/docs/strategies_manual.md` and re-run "
        "`python fx-scalper/scripts/render_strategies.py`."
    )
    lines.append("")
    lines.append(f"Last rendered: `{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}` UTC")
    lines.append("")
    lines.append("## Strategies registered")
    lines.append("")
    if not strategies:
        lines.append("_(no strategies detected — add one to `src/strategies/`)_")
    else:
        for cls in strategies:
            lines.append(_render_strategy(cls, manual))
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    target = _forex_root() / "STRATEGIES.md"
    target.write_text(render(), encoding="utf-8")
    print(f"Wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
