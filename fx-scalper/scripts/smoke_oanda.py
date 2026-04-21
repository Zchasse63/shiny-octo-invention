"""Day 1 smoke test — connect to OANDA practice, print balance + positions.

Run after filling in ``.env``:

    python scripts/smoke_oanda.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running this script directly: ``python scripts/smoke_oanda.py``.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.secrets import load_oanda_secrets  # noqa: E402
from src.oanda.account import AccountClient  # noqa: E402
from src.oanda.client import make_client  # noqa: E402
from src.oanda.instruments import InstrumentRegistry  # noqa: E402
from src.utils.logger import get_logger, init_logger  # noqa: E402


def main() -> int:
    init_logger()
    logger = get_logger(__name__)

    try:
        secrets = load_oanda_secrets()
    except RuntimeError as e:
        logger.error(f"Cannot load secrets: {e}")
        logger.error("Copy .env.example → .env and fill in your OANDA creds.")
        return 2

    logger.info(
        f"Connecting to OANDA {secrets.environment} "
        f"(account={secrets.account_id})"
    )

    client = make_client(secrets)
    account = AccountClient(client)
    registry = InstrumentRegistry(client)

    try:
        snap = account.snapshot()
    except Exception as e:
        logger.error(f"AccountSummary failed: {e}")
        return 1

    logger.info(
        f"Balance={snap.balance:.2f} {snap.currency}  "
        f"NAV={snap.nav:.2f}  "
        f"MarginUsed={snap.margin_used:.2f}  "
        f"MarginAvailable={snap.margin_available:.2f}  "
        f"OpenTrades={snap.open_trade_count}"
    )

    positions = account.get_open_positions()
    logger.info(f"Open positions: {len(positions)}")
    for p in positions:
        logger.info(
            f"  {p.get('instrument')} long={p.get('long', {}).get('units')} "
            f"short={p.get('short', {}).get('units')} "
            f"pl={p.get('pl')}"
        )

    registry.load()
    logger.info(f"Instruments loaded: {len(registry.names())}")
    for name in ("EUR_USD", "GBP_USD", "USD_JPY"):
        try:
            spec = registry.get(name)
        except KeyError:
            logger.warning(f"  {name}: not available on this account")
            continue
        logger.info(
            f"  {name}: pip={spec.pip_size} precision={spec.display_precision} "
            f"min_units={spec.minimum_trade_size} margin_rate={spec.margin_rate}"
        )

    logger.info("Smoke test OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
