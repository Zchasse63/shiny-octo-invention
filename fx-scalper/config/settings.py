"""Static configuration: account parameters, risk limits, circuit breakers.

All values are hardcoded per CLAUDE.md. Do not change without asking.
No magic numbers anywhere else in the codebase — everything lives here.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Account sizing (CLAUDE.md §Mission, §Position Sizing Logic)
# ---------------------------------------------------------------------------

ACCOUNT_STARTING_BALANCE_USD: Final[float] = 500.0
"""Starting capital in USD. Do not change — governs all sizing math."""

CASH_PER_TRADE_USD: Final[float] = 100.0
"""Cash committed per trade. Notional = this × leverage."""

MAX_LEVERAGE: Final[int] = 50
"""OANDA US regulatory maximum for major FX pairs."""

# Derived: $100 × 50 = $5,000 notional per trade, ~0.05 lots.

# ---------------------------------------------------------------------------
# Circuit breakers (CLAUDE.md §Circuit Breakers — non-negotiable)
# ---------------------------------------------------------------------------

MAX_CONCURRENT_POSITIONS: Final[int] = 2
"""Never more than 2 open positions — caps margin at $200."""

DAILY_LOSS_LIMIT_USD: Final[float] = 50.0
"""If realized + unrealized PnL for current UTC day ≤ -$50, halt until 00:00 UTC next day."""

ACCOUNT_FLOOR_USD: Final[float] = 400.0
"""If account NAV drops below this, halt bot, close all positions, send alert."""

MAX_CONSECUTIVE_LOSSES: Final[int] = 3
"""After N consecutive losing closed trades, pause new entries for 1 hour."""

CONSECUTIVE_LOSS_PAUSE_MINUTES: Final[int] = 60
"""How long to pause new entries after MAX_CONSECUTIVE_LOSSES hit."""

SINGLE_TRADE_MAX_LOSS_USD: Final[float] = 30.0
"""If any open position shows unrealized loss greater than this, halt — do NOT auto-close."""

OANDA_CONSECUTIVE_FAILURE_LIMIT: Final[int] = 3
"""After N consecutive OANDA API failures, halt new entries."""

# ---------------------------------------------------------------------------
# Session windows (CLAUDE.md §Circuit Breakers #6, §Strategy 3)
# All times in UTC unless stated.
# ---------------------------------------------------------------------------

FRIDAY_CLOSE_UTC_HOUR: Final[int] = 21
"""17:00 ET ≈ 21:00 UTC (summer) / 22:00 UTC (winter). We use 21:00 as a conservative edge."""

FRIDAY_FLAT_BY_UTC_HOUR: Final[int] = 20
"""16:55 ET ≈ 20:55 UTC — close all by 20:55, round to 20:00 cutoff hour."""

SUNDAY_OPEN_UTC_HOUR: Final[int] = 22
"""17:00 ET Sunday ≈ 22:00 UTC. No entries before this."""

# Session windows for Strategy 1 (mean reversion, Asian session)
ASIAN_SESSION_START_UTC_HOUR: Final[int] = 23
ASIAN_SESSION_END_UTC_HOUR: Final[int] = 7

# Session windows for Strategy 3 (London-NY range breakout)
LONDON_OPEN_UTC_HOUR: Final[int] = 8
LONDON_CLOSE_UTC_HOUR: Final[int] = 12
NY_OPEN_UTC_HOUR: Final[int] = 12
NY_CLOSE_UTC_HOUR: Final[int] = 16

# ---------------------------------------------------------------------------
# Instruments (OANDA underscore naming per CLAUDE.md §OANDA Gotchas #1)
# ---------------------------------------------------------------------------

INSTRUMENTS: Final[tuple[str, ...]] = ("EUR_USD", "GBP_USD", "USD_JPY")
"""The only three pairs we trade."""

ACCOUNT_CURRENCY: Final[str] = "USD"

# ---------------------------------------------------------------------------
# OANDA API endpoints
# ---------------------------------------------------------------------------

OANDA_PRACTICE_HOSTNAME: Final[str] = "api-fxpractice.oanda.com"
OANDA_LIVE_HOSTNAME: Final[str] = "api-fxtrade.oanda.com"
OANDA_STREAM_PRACTICE_HOSTNAME: Final[str] = "stream-fxpractice.oanda.com"
OANDA_STREAM_LIVE_HOSTNAME: Final[str] = "stream-fxtrade.oanda.com"

OANDA_REQUEST_TIMEOUT_SECONDS: Final[float] = 10.0
OANDA_MAX_RETRIES: Final[int] = 3
OANDA_BACKOFF_SECONDS: Final[float] = 0.5

# Rate limit: CLAUDE.md §Gotcha #6 — 120 req/sec per account
OANDA_RATE_LIMIT_RPS: Final[int] = 120

# ---------------------------------------------------------------------------
# Magic number for order tagging (CLAUDE.md §Code Standards)
# ---------------------------------------------------------------------------

MAGIC_NUMBER: Final[str] = "FXSCALPER-V1"
"""Prefix for every order's clientExtensions.id — lets us identify our own trades."""

# ---------------------------------------------------------------------------
# Backtest defaults
# ---------------------------------------------------------------------------

BACKTEST_START_DATE: Final[str] = "2023-01-01"
"""Historical data pull start — Dukascopy covers 2003+ but 2023 is sufficient."""

BACKTEST_TRAIN_END: Final[str] = "2023-12-31"
BACKTEST_TEST_START: Final[str] = "2024-01-01"
# Test end is "today" at runtime.

# ---------------------------------------------------------------------------
# Data paths (relative to project root)
# ---------------------------------------------------------------------------

DATA_RAW_DIR: Final[str] = "data/raw"
DATA_PROCESSED_DIR: Final[str] = "data/processed"
