# JOURNAL

Auto-generated from `fx-scalper/logs/events.jsonl` and `fx-scalper/docs/journal_manual.md`. **Do not hand-edit this file** — regenerate with `python fx-scalper/scripts/render_journal.py` (the pre-commit hook also runs this).

Last rendered: `2026-04-21 21:44:00` UTC  
Events logged: **40**

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
- **19:45:08 AI_QUERY** `(f5c893f)` — `artifact=/private/var/folders/xw/wy0mmntn4mv648kngfz29k0w0000gn/T/pytest-of-zach/pytest-19/test_dry_run_does_not_call_pro0/docs/research/ai_queries/20260421T194508-test.md`, `cost_usd=0.0`, `daily_spend_so_far_usd=0.0`, `input_tokens=6`, `model=anthropic/dry-run`, `output_tokens=0`, `provider=anthropic`, `tag=test`
- **19:51:15 AI_QUERY** `(f5c893f)` — `artifact=/private/var/folders/xw/wy0mmntn4mv648kngfz29k0w0000gn/T/pytest-of-zach/pytest-20/test_dry_run_does_not_call_pro0/docs/research/ai_queries/20260421T195115-test.md`, `cost_usd=0.0`, `daily_spend_so_far_usd=0.0`, `input_tokens=6`, `model=anthropic/dry-run`, `output_tokens=0`, `provider=anthropic`, `tag=test`
- **19:51:28 AI_QUERY** `(f5c893f)` — `artifact=/private/var/folders/xw/wy0mmntn4mv648kngfz29k0w0000gn/T/pytest-of-zach/pytest-21/test_dry_run_does_not_call_pro0/docs/research/ai_queries/20260421T195128-test.md`, `cost_usd=0.0`, `daily_spend_so_far_usd=0.0`, `input_tokens=6`, `model=anthropic/dry-run`, `output_tokens=0`, `provider=anthropic`, `tag=test`
- **19:52:06 DECISION `(f5c893f)`: Research loop wired — Tier 2 Knowledge module active**  
  src/utils/ai_research.py wraps vbt.chat/vbt.search with per-UTC-day budget cap ($10 default), event+artifact logging, runtime assertion that blocks import from src/live/. src/backtest/iterate.py formats exploration results into structured prompts. .mcp.json + scripts/run_mcp_server.sh prepared for Claude Code MCP registration (requires session restart to activate). See CONVENTIONS.md new section.
- **19:52:06 CONFIG `fx-scalper/requirements.txt`** `(f5c893f)`  
  `scipy`: `1.14.1 pin` → `>=1.14,<1.17` — scipy 1.17 broke sklearn compat (csr_matrix export moved). Pin below 1.17 until sklearn catches up. Downgraded in venv 1.17.1 to 1.16.3.
- **19:52:55 AI_QUERY** `(f5c893f)` — `artifact=/private/var/folders/xw/wy0mmntn4mv648kngfz29k0w0000gn/T/pytest-of-zach/pytest-22/test_dry_run_does_not_call_pro0/docs/research/ai_queries/20260421T195255-test.md`, `cost_usd=0.0`, `daily_spend_so_far_usd=0.0`, `input_tokens=6`, `model=anthropic/dry-run`, `output_tokens=0`, `provider=anthropic`, `tag=test`
- **20:00:16 AI_QUERY** `(79ab988)` — `artifact=/Users/zach/Desktop/Forex/fx-scalper/docs/research/ai_queries/20260421T200016-smoke_test.md`, `cost_usd=9.7e-05`, `daily_spend_so_far_usd=0.0001`, `input_tokens=39`, `model=openai/default`, `output_tokens=0`, `provider=openai`, `tag=smoke_test`
- **20:02:19 AI_QUERY** `(79ab988)` — `artifact=/private/var/folders/xw/wy0mmntn4mv648kngfz29k0w0000gn/T/pytest-of-zach/pytest-23/test_dry_run_does_not_call_pro0/docs/research/ai_queries/20260421T200219-test.md`, `cost_usd=0.0`, `daily_spend_so_far_usd=0.0`, `input_tokens=6`, `model=anthropic/dry-run`, `output_tokens=0`, `provider=anthropic`, `tag=test`
- **20:02:53 AI_QUERY** `(79ab988)` — `artifact=/private/var/folders/xw/wy0mmntn4mv648kngfz29k0w0000gn/T/pytest-of-zach/pytest-24/test_dry_run_does_not_call_pro0/docs/research/ai_queries/20260421T200253-test.md`, `cost_usd=0.0`, `daily_spend_so_far_usd=0.0`, `input_tokens=6`, `model=anthropic/dry-run`, `output_tokens=0`, `provider=anthropic`, `tag=test`
- **20:03:05 AI_QUERY** `(79ab988)` — `artifact=/private/var/folders/xw/wy0mmntn4mv648kngfz29k0w0000gn/T/pytest-of-zach/pytest-25/test_dry_run_does_not_call_pro0/docs/research/ai_queries/20260421T200305-test.md`, `cost_usd=0.0`, `daily_spend_so_far_usd=0.0`, `input_tokens=6`, `model=anthropic/dry-run`, `output_tokens=0`, `provider=anthropic`, `tag=test`
- **20:03:43 AI_QUERY** `(79ab988)` — `artifact=/Users/zach/Desktop/Forex/fx-scalper/docs/research/ai_queries/20260421T200343-smoke_test_v2.md`, `cost_usd=0.001628`, `daily_spend_so_far_usd=0.0017`, `input_tokens=35`, `model=openai/default`, `output_tokens=154`, `provider=openai`, `tag=smoke_test_v2`
- **20:06:22 AI_QUERY** `(79ab988)` — `artifact=/Users/zach/Desktop/Forex/fx-scalper/docs/research/ai_queries/20260421T200622-smoke_test_v3_anthropic.md`, `cost_usd=0.001689`, `daily_spend_so_far_usd=0.0034`, `input_tokens=23`, `model=anthropic/default`, `output_tokens=108`, `provider=anthropic`, `tag=smoke_test_v3_anthropic`
- **20:06:56 AI_QUERY** `(79ab988)` — `artifact=/private/var/folders/xw/wy0mmntn4mv648kngfz29k0w0000gn/T/pytest-of-zach/pytest-26/test_dry_run_does_not_call_pro0/docs/research/ai_queries/20260421T200656-test.md`, `cost_usd=0.0`, `daily_spend_so_far_usd=0.0`, `input_tokens=6`, `model=anthropic/dry-run`, `output_tokens=0`, `provider=anthropic`, `tag=test`
- **20:07:10 AI_QUERY** `(79ab988)` — `artifact=/private/var/folders/xw/wy0mmntn4mv648kngfz29k0w0000gn/T/pytest-of-zach/pytest-27/test_dry_run_does_not_call_pro0/docs/research/ai_queries/20260421T200710-test.md`, `cost_usd=0.0`, `daily_spend_so_far_usd=0.0`, `input_tokens=6`, `model=anthropic/dry-run`, `output_tokens=0`, `provider=anthropic`, `tag=test`
- **20:07:38 AI_QUERY** `(79ab988)` — `artifact=/private/var/folders/xw/wy0mmntn4mv648kngfz29k0w0000gn/T/pytest-of-zach/pytest-28/test_dry_run_does_not_call_pro0/docs/research/ai_queries/20260421T200738-test.md`, `cost_usd=0.0`, `daily_spend_so_far_usd=0.0`, `input_tokens=6`, `model=anthropic/dry-run`, `output_tokens=0`, `provider=anthropic`, `tag=test`
- **20:51:15 EXPLORATION_COMPLETE** `(bbf209c)` — `artifacts_dir=/Users/zach/Desktop/Forex/fx-scalper/backtest_results/explore_20260421T2051`, `csv_path=/Users/zach/Desktop/Forex/fx-scalper/backtest_results/explore_20260421T2051/full_results.csv`, `families=['pullback_ema', 'range_breakout', 'vwap_deviation', 'ema_cross', 'bb_rsi_mr', 'rsi_extreme']`, `total_runs=0`
- **20:56:46 EXPLORATION_COMPLETE** `(bbf209c)` — `artifacts_dir=/Users/zach/Desktop/Forex/fx-scalper/backtest_results/explore_20260421T2053`, `csv_path=/Users/zach/Desktop/Forex/fx-scalper/backtest_results/explore_20260421T2053/full_results.csv`, `families=['pullback_ema', 'range_breakout', 'vwap_deviation', 'ema_cross', 'bb_rsi_mr', 'rsi_extreme']`, `total_runs=8100`
- **21:04:45 EXPLORATION_COMPLETE** `(bbf209c)` — `artifacts_dir=/Users/zach/Desktop/Forex/fx-scalper/backtest_results/explore_20260421T2059`, `csv_path=/Users/zach/Desktop/Forex/fx-scalper/backtest_results/explore_20260421T2059/full_results.csv`, `families=['pullback_ema', 'range_breakout', 'vwap_deviation', 'ema_cross', 'bb_rsi_mr', 'rsi_extreme']`, `total_runs=8100`
- **21:08:30 AI_QUERY** `(bbf209c)` — `artifact=/Users/zach/Desktop/Forex/fx-scalper/docs/research/ai_queries/20260421T210830-iter1_eurusd_what_went_wrong.md`, `cost_usd=0.070488`, `daily_spend_so_far_usd=0.0739`, `input_tokens=2856`, `model=anthropic/default`, `output_tokens=4128`, `provider=anthropic`, `tag=iter1_eurusd_what_went_wrong`
- **21:09:55 EXPLORATION_COMPLETE** `(bbf209c)` — `best_expectancy_usd=0.32`, `best_pf=0.991`, `configs=1292`, `csv=backtest_results/explore_20260421T2059/full_results.csv`, `profitable_oos=0`, `rationale=Naive signal families all fail OOS. Best: bb_rsi_mr with PF ~0.99 (marginal losing). Spread eats vanilla signals. AI analysis + iteration plan in docs/research/ai_queries/20260421T210830-iter1_eurusd_what_went_wrong.md`, `round=1`
- **21:09:55 LEARNING: Tight-stop high-win-rate trap empirically confirmed** `(bbf209c)`  
  VWAP_deviation with 0.5 ATR stop + 0.75 R TP produced 70%+ win rate but Sharpe -7 to -20 and DD 99%. Rare large losses erase many small wins. Exit symmetry matters more than win rate.
- **21:09:56 AI_QUERY** `(bbf209c)` — `artifact=/private/var/folders/xw/wy0mmntn4mv648kngfz29k0w0000gn/T/pytest-of-zach/pytest-29/test_dry_run_does_not_call_pro0/docs/research/ai_queries/20260421T210956-test.md`, `cost_usd=0.0`, `daily_spend_so_far_usd=0.0`, `input_tokens=6`, `model=anthropic/dry-run`, `output_tokens=0`, `provider=anthropic`, `tag=test`
- **21:10:07 AI_QUERY** `(bbf209c)` — `artifact=/private/var/folders/xw/wy0mmntn4mv648kngfz29k0w0000gn/T/pytest-of-zach/pytest-30/test_dry_run_does_not_call_pro0/docs/research/ai_queries/20260421T211007-test.md`, `cost_usd=0.0`, `daily_spend_so_far_usd=0.0`, `input_tokens=6`, `model=anthropic/dry-run`, `output_tokens=0`, `provider=anthropic`, `tag=test`
- **21:10:36 AI_QUERY** `(bbf209c)` — `artifact=/private/var/folders/xw/wy0mmntn4mv648kngfz29k0w0000gn/T/pytest-of-zach/pytest-31/test_dry_run_does_not_call_pro0/docs/research/ai_queries/20260421T211036-test.md`, `cost_usd=0.0`, `daily_spend_so_far_usd=0.0`, `input_tokens=6`, `model=anthropic/dry-run`, `output_tokens=0`, `provider=anthropic`, `tag=test`
- **21:33:48 EXPLORATION_COMPLETE** `(b479c1c)` — `artifacts_dir=/Users/zach/Desktop/Forex/fx-scalper/backtest_results/explore_20260421T2116`, `csv_path=/Users/zach/Desktop/Forex/fx-scalper/backtest_results/explore_20260421T2116/full_results.csv`, `families=['pullback_ema', 'range_breakout', 'vwap_deviation', 'ema_cross', 'bb_rsi_mr', 'rsi_extreme', 'bb_rsi_mr_filtered', 'rsi_extreme_filtered']`, `total_runs=26640`
- **21:37:18 AI_QUERY** `(b479c1c)` — `artifact=/private/var/folders/xw/wy0mmntn4mv648kngfz29k0w0000gn/T/pytest-of-zach/pytest-32/test_dry_run_does_not_call_pro0/docs/research/ai_queries/20260421T213718-test.md`, `cost_usd=0.0`, `daily_spend_so_far_usd=0.0`, `input_tokens=6`, `model=anthropic/dry-run`, `output_tokens=0`, `provider=anthropic`, `tag=test`
- **21:37:33 AI_QUERY** `(b479c1c)` — `artifact=/private/var/folders/xw/wy0mmntn4mv648kngfz29k0w0000gn/T/pytest-of-zach/pytest-33/test_dry_run_does_not_call_pro0/docs/research/ai_queries/20260421T213733-test.md`, `cost_usd=0.0`, `daily_spend_so_far_usd=0.0`, `input_tokens=6`, `model=anthropic/dry-run`, `output_tokens=0`, `provider=anthropic`, `tag=test`
- **21:37:35 AI_QUERY** `(b479c1c)` — `artifact=/Users/zach/Desktop/Forex/fx-scalper/docs/research/ai_queries/20260421T213735-iter2_round2_winners_analysis.md`, `cost_usd=0.085878`, `daily_spend_so_far_usd=0.1598`, `input_tokens=3256`, `model=anthropic/default`, `output_tokens=5074`, `provider=anthropic`, `tag=iter2_round2_winners_analysis`
- **21:39:59 EXPLORATION_COMPLETE** `(b479c1c)` — `best_expectancy_usd=0.84`, `best_pf=2.01`, `configs=3526`, `csv=backtest_results/explore_20260421T2116/full_results.csv`, `profitable_oos=74`, `rationale=Session filter = edge. 52 of 74 winners in london_ny_overlap. rsi_extreme_filtered dominates. Best PF 2.01 on 95 trades, Sharpe 1.73, DD 18%. See docs/research/round2_findings.md.`, `round=2`
- **21:39:59 LEARNING: ADX filter provides no value in this sweep** `(b479c1c)`  
  70 of 74 round-2 winners had max_adx=None. Session filter (london_ny_overlap) is doing the work, not regime filter.
- **21:39:59 LEARNING: Session filter transforms losing families into winning ones** `(b479c1c)`  
  Same rsi_extreme that lost money unfiltered becomes rsi_extreme_filtered with PF 2.01 after adding london_ny_overlap session filter. Context beats signal design.
- **21:41:53 EXPLORATION_COMPLETE** `(b479c1c)` — `artifacts_dir=/Users/zach/Desktop/Forex/fx-scalper/backtest_results/explore_multi_tf_20260421T2137/tf_5min`, `csv_path=/Users/zach/Desktop/Forex/fx-scalper/backtest_results/explore_multi_tf_20260421T2137/tf_5min/full_results.csv`, `families=['pullback_ema', 'range_breakout', 'vwap_deviation', 'ema_cross', 'bb_rsi_mr', 'rsi_extreme', 'bb_rsi_mr_filtered', 'rsi_extreme_filtered']`, `total_runs=13950`
