# Narrative overlay — Days 0-3

## 2026-04-21 — Scaffold + automation

### What shipped
- Full `fx-scalper/` scaffold per [CLAUDE.md §Project Structure](../../CLAUDE.md).
  Days 1-3 scope from the 7-day plan: OANDA client/account/instruments/data/orders,
  RiskGuard (all 6 circuit breakers), sizing, trailing-stop math, BB-RSI mean
  reversion strategy (Strategy 1), vectorbt Pro harness skeleton, Dukascopy
  downloader, SQLite journal, loguru logging.
- **Stack locked-in:** Python 3.11.15 + TA-Lib (brew) + pandas 2.2.3 + numpy
  2.2.6 + scipy 1.14.1 + pandas-ta-classic 0.4.47 + oandapyV20 0.7.2 +
  loguru 0.7.2 + pydantic 2.9.2.
- **Tests:** 69 passing covering sizing, risk, journal, indicators,
  strategies, metrics, trailing, data loader, Dukascopy client, OANDA mocks.
- **Code review pass:** 3 CRITICAL + 4 HIGH + 1 MEDIUM issues found +
  fixed. See below under "Key decisions."
- **Historical data pipeline verified:** 436,213 EUR/USD ticks (2024-01-02
  → 2024-01-05) → 5,625 M1 bars → partitioned Parquet. Spread on row 0 is
  5×10⁻⁵ (0.5 pip) — realistic.

### Key decisions (also see DECISIONS/)
- **Wrote in-house Dukascopy downloader** ([DECISIONS/0001](../../DECISIONS/0001-in-house-dukascopy-client.md)) —
  `duka==0.2.0`'s asyncio+requests pipeline silently drops hours: its
  `fetch_day()` raises on any single failed task, nuking the whole day.
  Direct synchronous `requests.get` of the bi5 endpoint + in-process LZMA
  decompression is simpler, reliable, and small (~150 LOC in
  `src/backtest/dukascopy_client.py`). Also corrected the base URL to
  `datafeed.dukascopy.com` (the old `www.` path now 301-redirects).
- **Spread-honest SL/TP** — LONG signals anchor SL/TP to `ask_close`, SHORT
  to `bid_close`. Signal trigger still uses mid. Code review caught the
  original (mid-anchored) version as a CLAUDE.md §"Honest backtests"
  violation.
- **Consecutive-loss pause timer** — the first version re-armed the timer
  on every poll cycle while the journal still showed the loss streak, so
  the pause never expired. Fixed to arm only on the threshold edge.
- **Slippage passed as numpy array** — vectorbt Pro expects a numpy array
  for per-bar slippage, not a Series. Switched to `.to_numpy()`.
- **`fillna(method="ffill")` → `.ffill()`** — the keyword-method form
  raises in pandas 2.x.
- **`record_trade_close` raises on missing UUID** — silent UPDATE would
  mask mis-reconciled trades.

### What's deferred
- **vectorbt Pro** — gated behind the user's paid GitHub access. Harness
  imports it lazily and raises a helpful message if absent.
- **Full historical backfill** — only the smoke-test 4 days pulled. Full
  2023+ for all three pairs lands when Day 4 begins.
- **OANDA practice API key / account ID** — user will paste into `.env`
  after the environment checklist clears.
- **Strategies 2 & 3** — stubbed for Days 5-6.

## Convention

Further narrative goes at the TOP of this file, newest first. Structured
events are auto-captured via `src.utils.diary.log_event(...)` and merged
into JOURNAL.md below this overlay.
