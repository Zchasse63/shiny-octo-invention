"""Authenticated ``oandapyV20.API`` factory with retry + journal instrumentation.

Every request flows through :meth:`OandaClient.request`, which:

* Times the call for journaling.
* Retries on transient errors with exponential backoff.
* Parses OANDA's structured error envelope (``errorCode``, ``errorMessage``)
  and logs both request and response to SQLite.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from oandapyV20 import API
from oandapyV20.exceptions import V20Error

from config.secrets import OandaSecrets, load_oanda_secrets
from config.settings import (
    OANDA_BACKOFF_SECONDS,
    OANDA_LIVE_HOSTNAME,
    OANDA_MAX_RETRIES,
    OANDA_PRACTICE_HOSTNAME,
    OANDA_REQUEST_TIMEOUT_SECONDS,
)
from src.utils.journal import Journal
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class OandaClient:
    """Thin wrapper around ``oandapyV20.API`` + our journal.

    Attributes:
        api: Authenticated oandapyV20 API instance.
        account_id: Account ID extracted from secrets (convenience accessor).
        environment: Either ``"practice"`` or ``"live"``.
        journal: Shared SQLite journal for audit trail.
    """

    api: API
    account_id: str
    environment: str
    journal: Journal

    @property
    def hostname(self) -> str:
        """REST hostname for the configured environment."""
        return (
            OANDA_LIVE_HOSTNAME if self.environment == "live" else OANDA_PRACTICE_HOSTNAME
        )

    def request(self, endpoint: Any) -> dict[str, Any]:
        """Execute an oandapyV20 endpoint request with retries + journaling.

        Args:
            endpoint: An oandapyV20 endpoint instance (e.g. ``AccountDetails``).

        Returns:
            The parsed response dict.

        Raises:
            V20Error: Last V20Error after all retries exhausted.
        """
        last_error: V20Error | None = None
        endpoint_name = type(endpoint).__name__

        for attempt in range(1, OANDA_MAX_RETRIES + 1):
            start = time.perf_counter()
            try:
                response = self.api.request(endpoint)
                duration_ms = int((time.perf_counter() - start) * 1000)
                self.journal.record_api_call(
                    endpoint=endpoint_name,
                    method=getattr(endpoint, "METHOD", "?"),
                    request=_safe_request_repr(endpoint),
                    response=response,
                    status_code=getattr(endpoint, "STATUS_CODE", None),
                    duration_ms=duration_ms,
                )
                return response  # type: ignore[no-any-return]
            except V20Error as e:
                duration_ms = int((time.perf_counter() - start) * 1000)
                self.journal.record_api_call(
                    endpoint=endpoint_name,
                    method=getattr(endpoint, "METHOD", "?"),
                    request=_safe_request_repr(endpoint),
                    response={"error": str(e)},
                    status_code=getattr(e, "code", None),
                    error_code=getattr(e, "code", None),
                    duration_ms=duration_ms,
                )
                last_error = e
                if not _is_retriable(e):
                    logger.error(
                        f"Non-retriable V20Error on {endpoint_name}: {e}"
                    )
                    raise
                backoff = OANDA_BACKOFF_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    f"Retriable V20Error on {endpoint_name} "
                    f"(attempt {attempt}/{OANDA_MAX_RETRIES}): {e} — "
                    f"sleeping {backoff}s"
                )
                time.sleep(backoff)

        assert last_error is not None
        logger.error(f"All {OANDA_MAX_RETRIES} retries exhausted on {endpoint_name}")
        raise last_error


def _is_retriable(err: V20Error) -> bool:
    """Return True if the V20Error is safe to retry (5xx / rate-limit)."""
    code = getattr(err, "code", None)
    # oandapyV20's V20Error.code is often the HTTP status.
    if isinstance(code, int):
        return code in {429, 500, 502, 503, 504}
    return False


def _safe_request_repr(endpoint: Any) -> dict[str, Any]:
    """Extract a JSON-safe snapshot of the request body + path for journaling."""
    return {
        "endpoint": type(endpoint).__name__,
        "path": getattr(endpoint, "path", None),
        "params": getattr(endpoint, "params", None),
        "data": getattr(endpoint, "data", None),
    }


def make_client(
    secrets: OandaSecrets | None = None,
    journal: Journal | None = None,
    *,
    request_timeout: float = OANDA_REQUEST_TIMEOUT_SECONDS,
) -> OandaClient:
    """Build an :class:`OandaClient` ready for use.

    Args:
        secrets: Pre-loaded secrets; defaults to loading from ``.env``.
        journal: Shared journal; caller should inject the same instance app-wide.
            If None, a journal at the default path is created.
        request_timeout: Per-request HTTP timeout.

    Returns:
        Configured :class:`OandaClient`.
    """
    if secrets is None:
        secrets = load_oanda_secrets()

    if journal is None:
        from config.secrets import get_journal_db_path

        journal = Journal(get_journal_db_path())

    api = API(
        access_token=secrets.api_key,
        environment=secrets.environment,
        request_params={"timeout": request_timeout},
    )
    logger.info(
        f"OANDA client initialized: env={secrets.environment} "
        f"account={secrets.account_id}"
    )
    return OandaClient(
        api=api,
        account_id=secrets.account_id,
        environment=secrets.environment,
        journal=journal,
    )
