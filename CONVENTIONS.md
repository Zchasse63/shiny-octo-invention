# Conventions

Operational rules for this project. Complements [CLAUDE.md](CLAUDE.md)
(which specifies the *what*) by specifying the *how*.

## Memory stack

Four layers, newest always at the top, all committed to git:

| Layer | File | Who writes | Purpose |
|---|---|---|---|
| Plan | [CLAUDE.md](CLAUDE.md) | human only | Architectural North Star — the plan |
| Conventions | [CONVENTIONS.md](CONVENTIONS.md) | human + claude | Operational rules — this file |
| Journal | [JOURNAL.md](JOURNAL.md) | **auto-generated** | Running log of decisions, runs, incidents, learnings |
| Runbook | [RUNBOOK.md](RUNBOOK.md) | **auto-generated + manual overlay** | Operational procedures + config snapshot |
| Strategies | [STRATEGIES.md](STRATEGIES.md) | **auto-generated + manual overlay** | Signal logic spec for each strategy |
| Decisions | [DECISIONS/](DECISIONS/) | claude (+ human review) | ADRs for non-trivial decisions |

## Automation — do not hand-edit generated docs

* **JOURNAL.md** is rebuilt from `fx-scalper/logs/events.jsonl` +
  `fx-scalper/docs/journal_manual.md`. Scripts emit events via
  `src.utils.diary.log_event(...)`.
* **STRATEGIES.md** is rebuilt by introspecting `src/strategies/`.
  Add narrative under `fx-scalper/docs/strategies_manual.md` keyed by
  strategy name.
* **RUNBOOK.md** has a config snapshot pulled from `config/settings.py`
  plus human procedures from `fx-scalper/docs/runbook_manual.md`.

All three are re-rendered by the **pre-commit hook** on every commit, so a
stale JOURNAL/STRATEGIES/RUNBOOK cannot reach the remote. To render locally:

```bash
python fx-scalper/scripts/render_all.py
```

## Events — how to log something worth remembering

Any script or module that performs a notable action MUST call
`src.utils.diary.log_event(kind, **fields)`. This is the only channel
through which JOURNAL.md gets populated.

**Known event kinds** (see `src/utils/diary.py` docstring for full field list):

| Kind | Emitted by |
|---|---|
| `decision` | Manual — when claude makes an architectural call, call `log_event("decision", title=..., adr=..., rationale=...)` |
| `backtest_run` | `src.backtest.registry.record_run()` (automatic) |
| `config_change` | Manual — when a `config/settings.py` value changes, log it |
| `paper_start` / `paper_stop` | `scripts/run_paper.py` (automatic) |
| `live_start` / `live_stop` | `scripts/run_live.py` (automatic) |
| `incident` | Manual — on any production anomaly |
| `learning` | Manual — when a surprising fact emerges |
| `risk_event` | `src.live.risk.RiskGuard` (automatic, also in `risk_events` table) |

## Backtest registry

Every `scripts/run_backtest.py` invocation calls
`src.backtest.registry.record_run(...)`. The run lands in the
`backtest_runs` table in `fx-scalper/journal.db` AND in `events.jsonl`.
The `backtest_results/{run_id}/` directory holds per-run artifacts
(equity curve PNG, trades.parquet, metrics.json, params.yaml).

Query runs from research notebooks:

```python
from src.backtest.registry import query_runs
candidates = query_runs(strategy="bb_rsi_mr", verdict="candidate", min_sharpe=0.5)
```

## Knowledge base — `fx-scalper/docs/external/`

Shallow-clones of every third-party library we depend on live under
`fx-scalper/docs/external/`. See [fx-scalper/docs/external/INDEX.md](fx-scalper/docs/external/INDEX.md).

**Rule: before writing code that uses a library, grep the local clone first.**
Web searches for stale info are the enemy of correctness. If the local clone
is missing something relevant, either:

1. Run `fx-scalper/scripts/refresh_external_docs.sh` to pull latest, or
2. Use the Context7 MCP tool for on-demand doc queries, and note the URL
   in the answer so future-us knows where to look.

### Research loop — exploration ↔ AI iteration

The goal is NOT autonomous pattern-finding. It's a disciplined loop where:

1. **Explorer runs** a broad sweep over signal families × params × exit configs × walk-forward splits
2. **[`iterate.py`](fx-scalper/src/backtest/iterate.py)** loads the results CSV → formats top-N (+ per-family aggregates) as a structured prompt
3. **[`ai_research.ask()`](fx-scalper/src/utils/ai_research.py)** sends the prompt to `vbt.chat`, which does RAG over vectorbtpro's full docs/examples/Discord corpus
4. **Answer** is logged to `events.jsonl` (kind=`ai_query`) AND saved as a Markdown artifact under [`docs/research/ai_queries/<ts>-<tag>.md`](fx-scalper/docs/research/ai_queries/)
5. **Human reviews** (you read the answer). Claude proposes a concrete next iteration: add family X, extend param grid on Y, add filter Z. This goes in an ADR if it's load-bearing.
6. **Re-run** with the new configuration. Loop.

**Prompt kinds** (see [`iterate.py`](fx-scalper/src/backtest/iterate.py) `_PROMPTS`):
- `next_iteration` — "what patterns in the top performers, what should I try next"
- `diagnose_drawdown` — "why did this strategy bleed through period X"
- `compare_families` — "which family is strongest and why"
- `propose_new_family` — "what signal families haven't I tried"
- `explain_anomaly` — "here's a weird result, what could cause it"

**Budget guard rails:**
- Default cap: `$10/day`, enforced at call time by [`ai_research.ask()`](fx-scalper/src/utils/ai_research.py).
- Every call logs estimated cost + cumulative daily spend.
- Provider auto-picked from env; defaults to Anthropic (better for code/strategy questions), falls back to OpenAI.
- First call triggers corpus download + embedding build. Slow (~5-10 min). Subsequent calls cached and fast.

**Hard rule:**
- `src.utils.ai_research` must NEVER be imported from anything in [`src/live/`](fx-scalper/src/live/). Enforced at runtime by a stack-walk check. AI output is non-deterministic and must not be in the trading critical path. See ADR 0002 (Tier 3 deferred).

**What this loop does NOT do:**
- It doesn't "learn from your trading results" — each LLM call is stateless. The **logs** are the memory; the LLM contributes hypotheses, not persistence.
- It doesn't find edge for you. It helps you iterate on your own hypotheses faster.
- It doesn't control for overfitting. That's what walk-forward OOS does. Iteration that ignores OOS degradation is how retail quant strategies die.

### vectorbtpro — where to look first

For backtest harness code, grep [`fx-scalper/docs/external/vectorbt-pro/vectorbtpro/portfolio/`](fx-scalper/docs/external/vectorbt-pro/vectorbtpro/portfolio/)
for `from_signals.py`, [`vectorbtpro/utils/`](fx-scalper/docs/external/vectorbt-pro/vectorbtpro/utils/) for `Param` and `parameterized`,
[`vectorbtpro/generic/splitting/`](fx-scalper/docs/external/vectorbt-pro/vectorbtpro/generic/) for `Splitter`.
For the Intelligence/Knowledge module, see [`vectorbtpro/knowledge/`](fx-scalper/docs/external/vectorbt-pro/vectorbtpro/knowledge/)
(`completions.py`, `embeddings.py`, `reranking.py`, `doc_storing.py`).
For worked examples, [`vectorbt-pro/tests/`](fx-scalper/docs/external/vectorbt-pro/tests/) has
real usage patterns for nearly every feature. Context7 library ID for
remote queries: `/llmstxt/vectorbt_pro_pvt_4f8d7c01_llms-full_txt` (13,932
indexed snippets). Full capability review + integration plan:
[`fx-scalper/docs/research/vectorbtpro_capabilities.md`](fx-scalper/docs/research/vectorbtpro_capabilities.md)
and [`DECISIONS/0002-vectorbtpro-integration-scope.md`](DECISIONS/0002-vectorbtpro-integration-scope.md).

## Decision records (ADRs)

Non-trivial decisions — library swaps, schema changes, scope pivots — get
a numbered ADR in `DECISIONS/NNNN-title.md`. Template:

```markdown
# ADR NNNN: <title>

Date: YYYY-MM-DD
Status: accepted | superseded | deprecated

## Context
What problem are we solving? What constraints exist?

## Decision
What did we choose?

## Consequences
What do we gain? What do we give up? Any follow-up work implied?

## Alternatives considered
Short notes on paths not taken, and why.
```

When an ADR is written, emit a matching `decision` event via `log_event(...)`.

## Git workflow

* Init'd at `/Users/zach/Desktop/Forex`, branch `main`.
* Commits follow the pattern `type(scope): short imperative` — types:
  `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `research`.
* Tag at meaningful milestones: `v0.1-scaffold`, `v0.4-strategy1-walk-forward`,
  `v1.0-first-live-trade`, etc.
* **Pre-commit hook** runs:
  1. `fx-scalper/scripts/render_all.py` (re-renders the three docs)
  2. `fx-scalper/.venv/bin/ruff check fx-scalper` (blocking on errors)
  3. `fx-scalper/.venv/bin/pytest -q fx-scalper` (blocking on failure)
  4. Auto-stages any regenerated JOURNAL/STRATEGIES/RUNBOOK.md

If the hook fails, fix the underlying issue and re-commit. Do NOT bypass
with `--no-verify`.

## Before writing code — checklist

1. Grep `fx-scalper/docs/external/<library>/` for the API surface.
2. Grep existing `fx-scalper/src/` for prior usage patterns.
3. If crossing into a new module, check whether CLAUDE.md already
   specifies the design.
4. If making a non-trivial call, write the ADR first, then the code.
5. If code will change risk params — update `config/settings.py`, emit
   a `config_change` event, and update RUNBOOK via the renderer.

## After doing non-trivial work — checklist

1. Emit `log_event(...)` for anything that would matter in 3 months.
2. Run `fx-scalper/scripts/render_all.py`.
3. Run tests.
4. Commit with a descriptive message.
