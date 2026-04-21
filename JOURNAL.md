# JOURNAL

Auto-generated from `fx-scalper/logs/events.jsonl` and `fx-scalper/docs/journal_manual.md`. **Do not hand-edit this file** — regenerate with `python fx-scalper/scripts/render_journal.py` (the pre-commit hook also runs this).

Last rendered: `2026-04-21 19:36:19` UTC  
Events logged: **8**

## Narrative overlay

_From `fx-scalper/docs/journal_manual.md` — human-authored context._

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


## Event log

### 2026-04-21

- **14:42:05 DECISION: Automation layer + knowledge base added**  
  Render-driven JOURNAL/STRATEGIES/RUNBOOK + local clones of every library we depend on.
- **14:42:05 DECISION: In-house Dukascopy downloader** — see `DECISIONS/0001.md`  
  See DECISIONS/0001-in-house-dukascopy-client.md
- **14:42:05 LEARNING: pandas-ta-classic Imports is a dict**  
  Imports is a dict not an object in v0.4.x. talib_available() handles both.
- **15:18:02 DECISION `(4c0db8c)`: vectorbt Pro integration scope (three-tier plan)** — see `DECISIONS/0002.md`  
  Tier 1 (core harness) lands in Day 4. Tier 2 (Knowledge/MCP for research) post-Day 4. Tier 3 (runtime LLM gating) deferred post-Day 7. See fx-scalper/docs/research/vectorbtpro_capabilities.md for full review.
- **15:18:02 LEARNING: vectorbtpro ships its own MCP server** `(4c0db8c)`  
  vectorbtpro/mcp.py + mcp_server.py — we can register vbt.chat and vbt.search as MCP tools in Claude Code once a GitHub token + LLM API key are configured (Tier 2).
- **15:18:02 CONFIG `fx-scalper/.venv`** `(4c0db8c)`  
  `vectorbtpro`: `None` → `2026.4.7` — Day 4 backtest harness dependency. +120 transitive deps (numba, torch, plotly, openai, sentence-transformers, hyperopt, optuna, etc).
- **19:35:19 DECISION `(331d1f7)`: Reset to exploratory phase; scrap injected strategy pre-commitments** — see `DECISIONS/0003.md`  
  User identified BB-RSI / trend-momentum / session-breakout choices as prior-Claude assumptions. Switched to strategy-agnostic exploration across 6 signal families with common exit framework. See fx-scalper/docs/research/prior_assumptions_archive.md.
- **19:35:19 LEARNING: Exit framework is strategy-agnostic** `(331d1f7)`  
  All 6 families emit only entries; exits (SL, TP, trail variants) come from ExitConfig in src/strategies/exits.py. Lets us sweep all combos uniformly.
