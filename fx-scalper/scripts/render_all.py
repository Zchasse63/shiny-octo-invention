"""Run every renderer in sequence. Called by the pre-commit hook."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "render_journal.py",
    "render_strategies.py",
    "render_runbook.py",
]


def main() -> int:
    here = Path(__file__).resolve().parent
    failures = 0
    for s in SCRIPTS:
        print(f"==> {s}")
        result = subprocess.run(
            [sys.executable, str(here / s)],
            check=False,
        )
        if result.returncode != 0:
            failures += 1
            print(f"  ✗ {s} exited {result.returncode}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
