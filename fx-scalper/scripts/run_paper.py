"""Paper trading entry — Week 2 scaffold."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.secrets import load_oanda_secrets  # noqa: E402
from src.utils.logger import get_logger, init_logger  # noqa: E402


def main() -> int:
    init_logger()
    logger = get_logger(__name__)

    secrets = load_oanda_secrets()
    if secrets.environment != "practice":
        logger.error(
            f"Refusing to run paper trading with OANDA_ENVIRONMENT={secrets.environment!r}. "
            "Set OANDA_ENVIRONMENT=practice."
        )
        return 2

    logger.info(
        "run_paper is a scaffold. Wiring lands in Week 2 after a winning "
        "strategy clears the Day 7 deployment gate."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
