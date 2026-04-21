# ADR 0002: vectorbt Pro integration scope

Date: 2026-04-21
Status: accepted

## Context

User's paid access to `polakowo/vectorbt.pro` came through, unblocking Day 4
of the CLAUDE.md 7-day plan (BB-RSI mean reversion parameter sweep + walk-
forward validation). Full capability review is in
[`fx-scalper/docs/research/vectorbtpro_capabilities.md`](../fx-scalper/docs/research/vectorbtpro_capabilities.md).

Key facts from the review:

1. **vectorbtpro 2026.4.7** installed. Adds ~120 transitive deps including
   numba, torch, sentence-transformers, hyperopt, optuna, riskfolio-lib,
   cvxpy, plotly, openai, anthropic, tiktoken, the MCP client lib, and a
   native MCP server (`vectorbtpro.mcp_server`).

2. Core backtesting primitives (`Portfolio.from_signals`, `vbt.Param`,
   `vbt.Splitter.from_expanding`, native `sl_stop` / `tp_stop` / `sl_trail`,
   leverage-aware sizing, per-bar numpy-array slippage) cover everything
   CLAUDE.md Day 4 needs. Our existing harness skeleton is directionally
   correct; Day 4 is ~50-100 LOC on top of it.

3. The **Knowledge module** (`vectorbtpro/knowledge/`) is a full RAG
   pipeline: ingest → embed → store → retrieve → rerank → complete.
   Top-level APIs: `vbt.chat`, `vbt.search`, `vbt.find`. Providers:
   OpenAI / Anthropic / Gemini via LlamaIndex + direct clients. Requires
   `GITHUB_TOKEN` + one LLM API key. Per-call paid API cost.

4. vectorbtpro ships its own MCP server that can expose those APIs as
   MCP tools to Claude Code.

The question: how much of this do we integrate, when, and what do we
defer?

## Decision

Adopt a **three-tier integration**:

### Tier 1 — Core backtest harness (Day 4, immediate)

Wire `Portfolio.from_signals` + `vbt.Param` + `vbt.Splitter.from_expanding`
+ our Dukascopy Parquet via `vbt.ParquetData` into the existing harness
module. Runs are persisted in the `backtest_runs` SQLite table via the
existing [`src/backtest/registry.py`](../fx-scalper/src/backtest/registry.py).
No LLM calls. No MCP.

**First Day-4 PR is a harness calibration test** (buy-and-hold should lose
~1 pip of spread per trade, matching CLAUDE.md §Day 3), not the BB-RSI
sweep itself. Sweep code is meaningless if the harness is arithmetic-wrong.

### Tier 2 — Knowledge-assisted research (post-Day 4, between sessions)

Register vectorbtpro's MCP server in `.claude/settings.json` so `vbt.chat`
/ `vbt.search` become first-class tools during Claude Code research
sessions. Wrap all LLM calls in `src/utils/ai_research.py`, logging each
to `events.jsonl` under `kind="ai_query"` with prompt token count +
provider + estimated cost. Set a $10/month budget cap initially.

All Tier-2 activity is **research-time only** — between live-trading
sessions. Not wired into the trading loop. Not called by any code in
`src/live/`.

### Tier 3 — Runtime Knowledge calls at signal time (deferred to post-Day 7 minimum)

An optional, feature-flagged LLM-gated entry layer. Signal fires →
`vbt.chat` asked for a confidence assessment → trade allowed only if
confidence > threshold.

**Prerequisites for considering this at all:**
- Rule-based strategy has passed the Day 7 NautilusTrader gate.
- ≥14 days of paper trading with rule-based strategy tracking backtest OOS
  within 1σ.
- Paper-trade A/B: rule-based alone vs. rule-based + LLM gate, over
  sufficient volume to measure signal-lift statistically.
- Dedicated follow-up ADR covering non-determinism, provider lock-in, and
  latency on a 30s–5min scalper.

## Consequences

**Gained:**
- Day 4 parameter sweep + walk-forward validation unblocked.
- Research workflow dramatically improves once Tier 2 ships — we can query
  vbtpro's full docs corpus conversationally instead of grepping.
- Tier 3 remains an option for later without dictating current design.

**Given up:**
- `requirements.txt` now pulls ~120 transitive deps. Venv is noticeably
  heavier. Production VPS disk needs are higher. We pin
  `vectorbtpro==2026.4.7` after Day 4 validates so we don't drift.
- Must add explicit `fees=0.0` + session-mask guardrails every time we
  call `Portfolio.from_signals`. vbt's defaults are for crypto / equity
  day bars and are not always FX-appropriate.
- Metric-schema mapping layer. vbt has one schema, our `backtest_runs`
  SQLite has another, and a thin adapter must bridge them.

**Follow-up work:**
- **Day 4 PR 1:** harness calibration test (buy-and-hold lose-one-pip
  validation).
- **Day 4 PR 2:** `src/backtest/sweep.py` — parameterized grid runner.
- **Day 4 PR 3:** `src/backtest/walk_forward.py` — Splitter-driven IS→OOS
  loop.
- **Day 4 PR 4:** `scripts/run_backtest.py` wiring + the top-20 candidates
  CSV.
- **Tier 2 PR (post-Day 4):** `.claude/settings.json` MCP registration +
  `src/utils/ai_research.py` wrapper + budget cap + events-log hook.
- **Tier 3 ADR:** only after the prerequisites above are demonstrably met.

## Alternatives considered

1. **All-in immediately — wire `vbt.chat` into the live signal path now.**
   Rejected. Non-deterministic, adds paid-API dependency before we even
   know if the rule-based strategy works, violates CLAUDE.md's "honest
   backtests" and "circuit breakers have absolute priority" principles.

2. **Skip the Knowledge module entirely — stick to the pure backtesting
   APIs.** Rejected. Tier 2 is genuinely useful for research (the user
   explicitly wants to iterate on indicators and fine-tune over time)
   and the cost is bounded if we cap and wrap calls.

3. **Use `vbt.chat` for the research but not register the MCP server.**
   Considered. Works, but forces a Python REPL for every lookup instead
   of making the capability first-class in Claude Code. MCP registration
   is cheap and removes friction.

4. **Replace our `src/backtest/metrics.py` with vbt's accessors wholesale.**
   Rejected for schema-stability reasons. vbt's stats dict is large,
   broad, and versioned. Our `backtest_runs` table needs a fixed schema
   for cross-run comparison and index stability. We compute our fixed
   schema from vbt outputs at registration time, and optionally store the
   full vbt stats dict as JSON in a `notes` column for deep-dive.

5. **Use `vbt.IF` (indicator factory) to replace our
   `src/indicators/engine.py`.** Rejected. Our engine is the
   strategy-facing API with stable column names our tests depend on.
   Switching would require updating every strategy + test. vbt indicators
   are still available inside the harness where vectorization matters.
