"""Secret loading from ``.env``.

All runtime secrets live in ``.env`` (gitignored). ``.env.example`` shows the
expected keys. Do not hardcode credentials anywhere else.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class OandaSecrets:
    """Validated OANDA credentials pulled from environment.

    Attributes:
        api_key: Personal access token from OANDA dashboard.
        account_id: Account ID in format ``001-001-XXXXXXX-001``.
        environment: Either ``"practice"`` or ``"live"``.
    """

    api_key: str
    account_id: str
    environment: Literal["practice", "live"]


def _project_root() -> Path:
    """Walk up from this file to find the fx-scalper project root."""
    return Path(__file__).resolve().parent.parent


def load_oanda_secrets(dotenv_path: Path | None = None) -> OandaSecrets:
    """Load and validate OANDA secrets from ``.env``.

    Args:
        dotenv_path: Optional override path to a ``.env`` file. Defaults to
            ``<project_root>/.env``.

    Returns:
        Validated :class:`OandaSecrets`.

    Raises:
        RuntimeError: If any required env var is missing or the environment
            value is not ``"practice"`` or ``"live"``.
    """
    if dotenv_path is None:
        dotenv_path = _project_root() / ".env"

    # load_dotenv is idempotent; missing file is fine — env may be set already.
    load_dotenv(dotenv_path=dotenv_path, override=False)

    api_key = os.environ.get("OANDA_API_KEY")
    account_id = os.environ.get("OANDA_ACCOUNT_ID")
    environment = os.environ.get("OANDA_ENVIRONMENT", "practice").lower()

    missing: list[str] = []
    if not api_key:
        missing.append("OANDA_API_KEY")
    if not account_id:
        missing.append("OANDA_ACCOUNT_ID")
    if missing:
        raise RuntimeError(
            f"Missing required OANDA secrets: {', '.join(missing)}. "
            f"Copy .env.example to .env and fill in values."
        )

    if environment not in {"practice", "live"}:
        raise RuntimeError(
            f"OANDA_ENVIRONMENT must be 'practice' or 'live', got: {environment!r}"
        )

    # mypy: api_key/account_id are not None by this point (checked above).
    assert api_key is not None
    assert account_id is not None

    return OandaSecrets(
        api_key=api_key,
        account_id=account_id,
        environment=environment,  # type: ignore[arg-type]
    )


def get_log_level() -> str:
    """Return log level from env, defaulting to INFO."""
    load_dotenv(_project_root() / ".env", override=False)
    return os.environ.get("LOG_LEVEL", "INFO").upper()


def get_journal_db_path() -> Path:
    """Return SQLite journal path from env, defaulting to project_root/journal.db."""
    load_dotenv(_project_root() / ".env", override=False)
    raw = os.environ.get("JOURNAL_DB_PATH", "journal.db")
    path = Path(raw)
    if not path.is_absolute():
        path = _project_root() / path
    return path
