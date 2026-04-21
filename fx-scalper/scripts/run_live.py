"""Live trading entry — Week 3+ only.

Do NOT invoke until at least 14 days of paper results track backtest OOS
within 1 sigma per CLAUDE.md §Week 2+.
"""

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
    if secrets.environment != "live":
        logger.error(
            f"Refusing to run live with OANDA_ENVIRONMENT={secrets.environment!r}. "
            "Set OANDA_ENVIRONMENT=live only after passing the paper-trading gate."
        )
        return 2

    # Explicit user confirmation gate — intentionally simple stdin prompt.
    logger.warning(
        "This will place REAL orders with REAL money on account "
        f"{secrets.account_id}. Type 'I UNDERSTAND' to proceed:"
    )
    response = input("> ").strip()
    if response != "I UNDERSTAND":
        logger.info("Aborted.")
        return 1

    logger.info(
        "run_live is a scaffold. Full wiring lands after paper trading "
        "passes the 14-day tracking gate."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
