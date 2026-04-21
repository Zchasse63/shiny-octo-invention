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
