"""Day 3/4: run a backtest for a given strategy + parameter grid.

Scaffold only — actual parameter sweep lands in Day 4.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger, init_logger  # noqa: E402


def main() -> int:
    init_logger()
    logger = get_logger(__name__)
    logger.info(
        "run_backtest is a scaffold. Day 4 wires up vectorbt Pro and "
        "parameter sweeps for BB-RSI mean reversion."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
