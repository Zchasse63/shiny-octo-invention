# vectorbtpro — capabilities review and integration plan

**Version installed:** `2026.4.7`
**Upstream:** https://github.com/polakowo/vectorbt.pro
**Local clone:** [`fx-scalper/docs/external/vectorbt-pro/`](../external/vectorbt-pro/)
**Docs source used:** Context7 `/llmstxt/vectorbt_pro_pvt_4f8d7c01_llms-full_txt` (13,932 indexed snippets) + direct source inspection

---

## Executive summary

vectorbt Pro is not just a backtesting library — it's a **vectorized research platform** with native parameter sweeping (`vbt.Param`), walk-forward cross-validation (`vbt.Splitter`), leverage-aware stop modelling (`sl_stop`/`tp_stop`/`sl_trail`/`tsl_stop`), Numba-compiled pipelines, and — as of recent releases — a built-in **Knowledge module** (what the user called "Intelligence") that wraps the entire vbtpro docs corpus in a RAG pipeline with OpenAI / Anthropic / Gemini providers, plus a native MCP server so Claude Code can talk to it as a tool. This is substantially more capable than what we assumed when we gated Day 4 on it. Our existing harness skeleton in [`src/backtest/harness.py`](../../src/backtest/harness.py) is correct in approach (lazy import, `Portfolio.from_signals`, time-varying slippage) but the actual CLAUDE.md Day-4 sweep is roughly a dozen lines once `vbt.Param` is used as the grid primitive and `vbt.Splitter` as the walk-forward driver. The Knowledge module is **powerful for researcher-style iteration** (ask vbt the same question three different ways, fine-tune indicator combos, retrieve prior notebook patterns) but **should stay out of the live bot path** until post-Day 7: it carries paid-API cost per call, non-determinism, and LLM-vendor lock-in we don't want in the risk loop.

---

## Capability catalog

### Package structure (from the clone)

Top-level `vectorbtpro/` submodules:

| Module | Purpose | Day-4 relevance |
|---|---|---|
| `portfolio/` | `Portfolio.from_signals`, `.from_orders`, `pfopt/`. The core backtester. | **Core** — our harness sits on this |
| `signals/` | Signal generation primitives (entries/exits), `generators/`. | Core |
| `indicators/` | Indicator factory + built-in indicators; wraps TA-Lib and pandas-ta under one API. | Medium — we already wrap pandas-ta-classic ourselves, but `vbt.IF` (indicator factory) is cleaner for custom indicators |
| `generic/` | Array-level primitives used everywhere — `splitting`, `ranges`, `returns`, `drawdowns`. | **Core** — `vbt.Splitter` lives here |
| `ohlcv/` | OHLCV-specific tooling (resampling, plotting, accessors). | Medium |
| `records/` | Trade/order record tables, typed with Numba. | Core — post-run analysis |
| `returns/` | Returns accessor (Sharpe, Sortino, Calmar, drawdown stats, etc.). | **Core** — replaces our [`src/backtest/metrics.py`](../../src/backtest/metrics.py) for most stats; we keep our thin wrapper to preserve a stable schema |
| `data/` | Data fetchers (Yahoo, Binance, CCXT, CSV, Parquet, HDF5, DuckDB). `vbt.Data.pull`, `vbt.CSVData`, `vbt.ParquetData`. | **Core** — loads our Dukascopy Parquet directly |
| `base/` | Foundational classes (wrapping, chunking, reshaping). | Indirect |
| `utils/` | `Param`, `parameterized` decorator, `ProgressBar`, chunking, template engine, Numba helpers. | **Core** — `vbt.Param` is the sweep primitive |
| `labels/` | Supervised-learning label generation (fixed horizon, triple barrier, etc.). | ML / Tier-2 |
| `px/` | Plotly integration (optional, interactive). | Optional |
| **`knowledge/`** | **The Intelligence module.** | **Deep dive below** |
| `mcp.py` + `mcp_server.py` | Native MCP tool registry + server implementation. | **Integration opportunity** |
| `cli.py` | CLI entry point (`vbt` command) | Optional |
| `_settings.py` | Global settings (flex_cfg), including the entire `chat` config block. | Core |
| `_opt_deps.py` | Optional-dependency resolver. | Indirect |

### Parameter sweeps — `vbt.Param`

Native, vectorized grid search over any subset of arguments. Three usage patterns:

**Pattern 1 — embedded in `Portfolio.from_signals`:**
```python
pf = vbt.Portfolio.from_signals(
    data,
    entries=entries,
    sl_stop=vbt.Param([np.nan, 0.005, 0.01, 0.015]),
    tp_stop=vbt.Param([0.01, 0.02, 0.03]),
    sl_trail=vbt.Param([True, False]),
    leverage=50,
    freq="1min",
)
# pf.sharpe_ratio is now indexed by (sl_stop, tp_stop, sl_trail)
```

**Pattern 2 — custom strategy function with `@vbt.parameterized`:**
```python
@vbt.parameterized(merge_func="concat")
def test_bb_rsi(data, bb_length, bb_std, rsi_length, rsi_thresh, adx_thresh):
    # full strategy body
    return pf.returns_acc.sharpe_ratio

test_bb_rsi(
    data,
    bb_length=vbt.Param([15, 20, 30]),
    bb_std=vbt.Param([1.8, 2.0, 2.2]),
    rsi_length=vbt.Param([10, 14, 21]),
    rsi_thresh=vbt.Param([25, 30, 35]),
    adx_thresh=vbt.Param([18, 20, 22]),
    _random_subset=200,  # sub-sample the 243-combo grid if needed
)
```

**Pattern 3 — multi-level Param (one logical param spans multiple args):**
```python
sl_stop=vbt.Param([np.nan, 0.1], level=0),
tsl_stop=vbt.Param([np.nan, 0.1, 0.1], level=1),
tp_stop=vbt.Param([np.nan, 0.1], level=2),
```

Levels let related params (e.g. RSI long threshold + short threshold) move together.

### Walk-forward — `vbt.Splitter`

The cross-validator. Three constructors matter for us:

| Constructor | Behavior | Fits which CLAUDE.md split? |
|---|---|---|
| `Splitter.from_rolling(index, length=..., split=0.5, set_labels=["IS","OOS"])` | Fixed-length sliding window, split by fraction | Default walk-forward |
| `Splitter.from_n_rolling(index, n=5, split=0.5)` | N rolling windows covering the full range | Small-data validation |
| `Splitter.from_expanding(index, min_length=..., offset=..., split=-180)` | Expanding train / fixed OOS | Best fit to CLAUDE.md: train 2023, test 2024–2025 |

For the CLAUDE.md Day-4 spec ("train 2023, test 2024–2025"), `from_expanding` is the cleanest match.

### Stops, leverage, and spread-honest slippage

Native first-class support — our harness skeleton was already on the right track:
- `sl_stop` — stop-loss as fraction of entry price
- `tp_stop` — take-profit as fraction
- `sl_trail=True` — trailing stop flag
- `tsl_stop` + `tsl_th` — trailing stop distance + activation threshold (separate from `sl_trail`)
- `leverage=50` — applied to sizing automatically
- `slippage=<float | np.ndarray>` — scalar OR per-bar numpy array. Our bid-ask half-spread as a numpy array (already wired in [`src/backtest/harness.py:103`](../../src/backtest/harness.py)) is exactly the right shape.
- `price="nextopen"` or `"open"` / `"close"` / custom series — controls fill price anchor.

The Day-4 candidate criteria (OOS Sharpe > 0.5, max DD < 15%) are produced natively by vbt's `Portfolio` accessors (`pf.sharpe_ratio`, `pf.max_drawdown`). Our thin [`metrics.py`](../../src/backtest/metrics.py) wrapper still earns its keep as a stable schema for the `backtest_runs` SQLite table — vbt's accessor surface is large and evolves.

### Data loading — `vbt.ParquetData`

Our partitioned Parquet under `data/processed/EUR_USD/year=YYYY/month=MM/bars.parquet` loads directly:
```python
data = vbt.ParquetData.pull(
    "fx-scalper/data/processed/EUR_USD",
    start="2023-01-01",
    end="2024-12-31",
)
```
No need to write a custom reader. Our existing [`load_symbol_bars()`](../../src/backtest/data_loader.py) remains useful for non-vbt consumers (notebook exploration) but the vbt path bypasses it.

### Knowledge module (the "Intelligence")

**Location:** `vectorbtpro/knowledge/` — 15 Python files.

**What it is:** A full **Retrieval-Augmented Generation (RAG) pipeline over the vectorbt Pro documentation + example notebooks + Discord history**. Pre-built as first-class APIs on the top-level `vbt` namespace.

**Files:**

| File | Responsibility |
|---|---|
| `asset_pipelines.py` | Orchestrates the ingest → embed → store → retrieve → rerank → complete pipeline |
| `base_assets.py` + `base_asset_funcs.py` | Asset abstraction — "documentation pages", "messages", "examples" |
| `custom_assets.py` + `custom_asset_funcs.py` | Bring your own asset types (we could index our own strategy notebooks here) |
| `completions.py` | LLM completion wrappers — OpenAI, Anthropic, Gemini, LlamaIndex |
| `embeddings.py` | Embedding model wrappers (default: OpenAI `text-embedding-3-large` @ 256 dims; Gemini fallback) |
| `doc_storing.py` | Persistent vector store for embedded docs |
| `doc_ranking.py` + `reranking.py` | First-pass retrieval + cross-encoder reranking (sentence-transformers / external rerankers) |
| `text_splitting.py` | Chunking strategies |
| `tokenization.py` | tiktoken-based token counting |
| `provider_utils.py` | Per-provider config resolver |
| `formatting.py` | Output formatting (markdown / html) |

**Top-level APIs:**
```python
vbt.chat("How do I run a parameter sweep with walk-forward?")
# → LLM-authored answer, streaming by default, using RAG over the docs

vbt.search("trailing stop examples for leveraged FX")
# → retrieval-only, returns HTML page of ranked excerpts

vbt.find("Splitter.from_expanding")
# → direct API lookup
```

**Requirements:**
- `GITHUB_TOKEN` environment variable (to fetch vbtpro's private docs repo on first call)
- One LLM provider API key: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GEMINI_API_KEY`
- First call is slow (builds the vector store); subsequent calls are cached

**Config knobs** (excerpt from `vbt._settings.chat`):
- `stream=True` — streaming responses
- `max_tokens=100000`
- `system_prompt="You are a helpful assistant…"` (overridable)
- `rank_kwargs.top_k` — retrieval depth before rerank
- `quick_mode=False` — quality vs speed knob
- `embeddings='auto'` — OpenAI by default, Gemini fallback
- `chat_dir` — persistent history location

**Cost model:** We pay per call. Embeddings happen once (batch of 256) and are cached. Completions bill per request. No sticker shock if used for manual research ("I want a pattern for walk-forward with leverage"); expensive if wired into a live trading loop (thousands of calls/day).

### MCP server — vbtpro talks to Claude Code natively

`vectorbtpro/mcp.py` + `vectorbtpro/mcp_server.py` implement an MCP server that exposes vbt tools (including `chat` and `search`) to any MCP client. Claude Code supports MCP servers via `settings.json`. This means:

> **We can register vbtpro as an MCP server and call `vbt.chat` / `vbt.search` from within Claude Code without writing Python for every lookup.**

This is a big workflow upgrade. It replaces our current approach of "grep `docs/external/vectorbt-pro/` first" with "ask the vbt MCP server first, fall back to grep."

Registration (not done yet — gated on Tier 2 decision):
```bash
# Pseudocode — exact invocation TBD from vectorbtpro/mcp_server.py
.venv/bin/python -m vectorbtpro.mcp_server --port 3333
# Then add to .claude/settings.json under `mcpServers`
```

---

## Conflicts with existing project structure

Issues to resolve on Day 4, flagged now so we don't stumble:

1. **Indicator wrapping divergence.** Our [`src/indicators/engine.py`](../../src/indicators/engine.py) wraps pandas-ta-classic directly and produces stable column names (`rsi_14`, `bb_upper_20_2.0`, etc.). vbt has its own `vbt.IF` / `vbt.indicator("talib:RSI")` pattern. **Resolution:** keep our engine as the strategy-facing API; use vbt indicators inside the harness when that lets us vectorize faster. Don't mix column-naming conventions in a single strategy file.

2. **Signal-bar semantics.** CLAUDE.md mandates "never compute signals on the forming bar — use `iloc[-2]`." vbt's `Portfolio.from_signals` uses the current bar's signal for the *next* bar's action by default (`price="nextopen"`). **Resolution:** use `price="nextopen"` or `price="open"` with a shifted signal — both are equivalent. Test this is spread-honest against our live bot logic.

3. **Sizing semantics.** vbt has its own sizing model (`size_type="percent"`, `size_type="amount"`, `fees`, etc.). Our CLAUDE.md mandates `$100 × leverage = $5,000 notional per trade`. **Resolution:** pass `size=compute_position_units(...)` per-signal into vbt, OR use `size_type="value"` with `size=5000`. Match exactly so backtest and live produce the same unit counts.

4. **Metric schema drift.** vbt's `Portfolio.stats()` returns dozens of metrics with its own naming. Our `backtest_runs` SQLite table has a fixed schema. **Resolution:** in `src/backtest/registry.py`, compute our fixed schema from vbt outputs (Sharpe, Sortino, profit factor, etc.) and store. vbt stats can optionally be serialized alongside (JSON) for deep-dive.

5. **Session / weekend gating.** Our harness strips signals during Fri 20:00 UTC → Sun 22:00 UTC. vbt has `freq="1min"` but no native session concept. **Resolution:** apply our `_weekend_mask()` to `entries` / `exits` before passing to `Portfolio.from_signals`. Already wired in [`src/backtest/harness.py`](../../src/backtest/harness.py).

6. **Test coverage for Param/Splitter combos.** No current tests exercise vbt param sweeps or walk-forward. **Resolution:** Day-4 first PR must add at least one `test_harness_param_sweep.py` and one `test_harness_walk_forward.py`.

---

## Three-tier integration plan

### Tier 1 — Core backtest harness (Day 4, ship first)

**Scope:** Wire `Portfolio.from_signals` + `vbt.Param` + `vbt.Splitter.from_expanding` into our existing harness. Register every run in the `backtest_runs` table. No LLM calls, no MCP.

**Deliverables:**
- Replace the TODO body in [`src/backtest/harness.py`](../../src/backtest/harness.py) with a real `Portfolio.from_signals` call.
- New `src/backtest/sweep.py` — one function per strategy that takes `data` + `params_grid` + `splitter` and returns a DataFrame of `(param_combo × split) → metrics`.
- New `src/backtest/walk_forward.py` — orchestrates IS-fit → OOS-evaluate loops using `Splitter.from_expanding`.
- Wire into [`scripts/run_backtest.py`](../../scripts/run_backtest.py) as the actual entry point.
- Add `test_harness_param_sweep.py` + `test_harness_walk_forward.py`.

**Success criteria (from CLAUDE.md):** 243 param combos × 3 pairs × walk-forward, producing a CSV of top-20 param sets with OOS Sharpe > 0.5 and max DD < 15%.

### Tier 2 — Knowledge-assisted research loops (post-Day 4, between sessions)

**Scope:** Use `vbt.chat` / `vbt.search` as a research copilot for iterative strategy tuning *outside* the live trading loop. The human (or Claude Code during research sessions) asks questions like "what are best-practice patterns for trailing stops on minute-bar EUR/USD?" and vbt serves answers grounded in its own docs and notebooks.

**Deliverables:**
- Register vbtpro's MCP server in `.claude/settings.json` so `vbt.chat` / `vbt.search` become first-class tools.
- Add `docs/research/chat_recipes.md` — canonical prompts for the kinds of questions that advance Day 5 / 6 strategies.
- Optionally index our own strategy notebooks via `custom_assets.py` so vbt can retrieve our prior work.

**Cost guardrails:**
- Set `OPENAI_API_KEY` usage cap at $10/month initially.
- All `vbt.chat` calls go through a tiny wrapper in `src/utils/ai_research.py` that logs prompt + token count + provider to `events.jsonl` under `kind="ai_query"`. Budget transparency.

**Pre-req:** User provides at least one of `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` + a `GITHUB_TOKEN` with vbtpro repo read access.

### Tier 3 — Runtime Knowledge calls during trading (deferred; post-Day 7 minimum)

**Scope:** LLM-authored reasoning at signal time — "current BB-RSI setup looks borderline, what's the historical precedent for this pattern on GBP/USD?" — returning a confidence score that gates the entry.

**Why deferred:**
- Adds paid-API dependency to the critical path.
- Non-deterministic → reproducibility of live results vs. backtest breaks.
- Lock-in to one LLM provider → strategy behaviour changes if they rev the model.
- Latency: LLM call adds seconds to signal-time path. Unacceptable on a 30s–5min scalper without careful engineering.

**Reversibility:** We can add this layer *without* breaking anything in Tier 1 or 2 — it becomes an optional decorator on the signal function, feature-flagged.

**Pre-req for considering this at all:**
- Strategy has passed Day-7 NautilusTrader gate and 14 days of paper trading.
- We can measure (in paper) whether LLM gating actually reduces loss rate vs. the rule-based strategy alone.
- A dedicated ADR covering the three risks above.

---

## Risks and watch-items

| Risk | Mitigation |
|---|---|
| Knowledge calls billed per request; easy to leak money | Wrapper that logs + caps; alert on daily spend > $X |
| First `vbt.chat` call downloads and embeds the whole vbtpro docs corpus (slow + storage) | Run once in dev environment; persist under `.vbt_cache/`; document in RUNBOOK |
| LLM provider output is non-deterministic | Never use inside the trading loop without feature flag + kill switch |
| vbtpro major version bumps can break our harness | Pin `vectorbtpro==2026.4.7` in requirements.txt after validating Day 4 |
| vbt introduces its own `vbt.Param` grid semantics that differ from naive itertools.product | Use vbt primitives, don't roll our own; tested via grid equivalence test |
| 120 transitive deps now in venv (numba, torch, plotly, hyperopt, optuna, riskfolio-lib, cvxpy, yfinance) | All are optional imports at vbtpro level; review if install footprint matters on production VPS |
| Portfolio.from_signals silently applies fees=0 by default | Our harness must explicitly pass `fees=0.0` (OANDA is commission-free) + verify via a buy-and-hold calibration test |

---

## Recommended Day-4 first step

**Before touching strategy parameters**, write a **harness calibration test**:

1. Load one month of EUR/USD from the Parquet we pulled.
2. Build a buy-and-hold "strategy" (`entries=[True, False, False, …]`, `exits=[False, …, True]`).
3. Pass through `Portfolio.from_signals` with `leverage=50`, `slippage=<our half-spread array>`, `fees=0.0`.
4. Verify final PnL ≈ `(exit_close - entry_close) × units - (half_spread × 2 × units)`.
5. If it matches within 1 pip, the harness is spread-honest and the sweep can run.
6. If it doesn't, we've found the bug before it propagates into 243 × 3 pairs of garbage results.

This is the "buy-and-hold should lose ~1 pip of spread per trade" validation test CLAUDE.md §Day 3 already specifies — we'd just be running it end-to-end now that vbt is installed.

After calibration passes, the actual BB-RSI sweep is maybe 50 lines of code on top of [`src/strategies/bb_rsi_mr.py`](../../src/strategies/bb_rsi_mr.py).

---

**See also:**
- [`DECISIONS/0002-vectorbtpro-integration-scope.md`](../../../DECISIONS/0002-vectorbtpro-integration-scope.md) — the corresponding ADR
- [`../external/vectorbt-pro/README.md`](../external/vectorbt-pro/README.md) — upstream README
- [`../external/vectorbt-pro/vectorbtpro/knowledge/`](../external/vectorbt-pro/vectorbtpro/knowledge/) — the Knowledge module source
- [`../external/vectorbt-pro/tests/`](../external/vectorbt-pro/tests/) — best worked examples in the repo
