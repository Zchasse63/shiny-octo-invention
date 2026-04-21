"""Centralised loguru configuration.

Never use ``print()`` in ``src/``. Import :func:`get_logger` instead.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from loguru import logger as _loguru_logger

from config.secrets import get_log_level

_INITIALIZED = False


def init_logger(
    log_level: str | None = None,
    log_file: Path | None = None,
) -> None:
    """Configure loguru sinks. Idempotent; safe to call from scripts and tests.

    Args:
        log_level: One of TRACE/DEBUG/INFO/WARNING/ERROR. Defaults to env LOG_LEVEL.
        log_file: Optional file sink. If supplied, logs also stream there with
            daily rotation and 14-day retention.
    """
    global _INITIALIZED

    _loguru_logger.remove()  # Drop default stderr sink so format is ours.

    effective_level = (log_level or get_log_level()).upper()
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> "
        "<level>{level: <8}</level> "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
        "- <level>{message}</level>"
    )

    _loguru_logger.add(
        sys.stderr,
        level=effective_level,
        format=fmt,
        colorize=True,
        backtrace=True,
        diagnose=False,  # Never leak variable values to stderr in production.
    )

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        _loguru_logger.add(
            str(log_file),
            level=effective_level,
            format=fmt,
            colorize=False,
            rotation="1 day",
            retention="14 days",
            backtrace=True,
            diagnose=False,
            enqueue=True,  # Thread-safe writes.
        )

    _INITIALIZED = True


def get_logger(name: str) -> Any:
    """Return a loguru logger bound to ``name``.

    Auto-initializes with env-based defaults on first call.

    Args:
        name: Module name (typically ``__name__``).

    Returns:
        Bound loguru logger. Typed as Any because loguru's ``Logger`` is dynamic.
    """
    if not _INITIALIZED:
        init_logger()
    return _loguru_logger.bind(component=name)
