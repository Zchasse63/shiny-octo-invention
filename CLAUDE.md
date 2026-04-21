# FX Scalper — Project North Star

**Last meaningful edit:** 2026-04-21 (rewrite after user correction; see
[`fx-scalper/docs/research/prior_assumptions_archive.md`](fx-scalper/docs/research/prior_assumptions_archive.md)
for what was stripped and why).

This file contains **only user-stated requirements**. Anything that looks
like a specific strategy, parameter grid, or threshold value should trigger
a question, not an assumption. When in doubt, ask before writing code.

---

## Mission

Build a production-grade Python forex scalping bot trading **EUR/USD,
GBP/USD, and USD/JPY** on **30-second to 5-minute timeframes** through
**OANDA**, aiming for **high-frequency scalping** (multiple trades per hour)
with a **base-hits-plus-trailing-runner exit philosophy** — tight initial
stops, small fixed take-profits on the bulk of trades, and trailing stops
that let winners run when a move has more in it.

Trades are NEVER closed by time. Only by price action against the trail or
by hitting stop loss.

Start tiny ($500). Prove the system doesn't self-destruct. Then scale.

---

## Account parameters (user-stated, hardcoded)

- Starting capital: **$500 USD**
- Cash committed per trade: **$100 margin**
- Leverage: **50:1** (OANDA US regulatory maximum)
- Resulting position size per trade: **$5,000 notional** (~0.05 standard lots, ~$0.50/pip on EUR/USD)
- Broker: **OANDA** — CFTC/NFA regulated, US entity
- Dev platform: **macOS**
- Production runtime: **Linux VPS**

## Non-negotiable principles

1. **Honest backtests over optimistic ones.** Always model real spreads,
   real commission, real slippage. Synthetic assumptions are the enemy.
2. **Walk-forward validation required** before any live deployment.
   Train on one period, validate OOS on the next. No cherry-picking.
3. **Circuit breakers exist in concept.** Specific thresholds (daily-loss,
   consecutive-loss, single-trade blowout, account floor) are TO BE
   DETERMINED after exploration shows what the strategy's normal
   drawdown and loss-streak distribution looks like. Ops-level breakers
   (OANDA-disconnect, weekend flat-by) are sensible regardless.
4. **Magic number discipline on every order.** Every OANDA order carries
   `clientExtensions.id` tagging our project, strategy name, and trade UUID.
5. **Log every request and response to SQLite** for audit. Non-optional.
6. **Never compute signals on the forming bar.** Use the last CLOSED bar.
7. **All timestamps are tz-aware UTC.** No naïve datetimes.

---

## Pre-decided architecture (stack choices, user-confirmed)

### Execution venue
- **OANDA v20 REST + streaming API** via the official `oandapyV20` package
- Standard account (no commission, ~0.6 pip spread on EUR/USD)
- `api-fxtrade.oanda.com` (live) / `api-fxpractice.oanda.com` (paper)

### Core libraries (installed, pinned)
- `oandapyV20` — OANDA execution
- `pandas`, `numpy`, `scipy`, `pyarrow` — data stack
- `pandas-ta-classic` (with `TA-Lib` C backend for speed) — indicators
- `vectorbtpro` — backtesting + parameter sweeps + walk-forward + Knowledge/Intelligence module
- `nautilus_trader` — execution-realism validation gate before paper trading
- `loguru`, `pydantic`, `python-dotenv` — dev utilities
- `pytest`, `ruff`, `mypy` — QA

### Historical data
- **Primary:** Dukascopy tick data (2003–present, institutional ECN bid/ask)
- **Pulled via:** our own in-house downloader at `src/backtest/dukascopy_client.py`
  (duka 0.2.0 had a bug that silently dropped hours; see `DECISIONS/0001`)
- **Period:** 2023-01-01 → today for all three pairs
- **Storage:** Partitioned Parquet at `data/processed/{symbol}/year={YYYY}/month={MM}/bars.parquet`
- **OANDA historical candles** used only for final broker-feed sanity check.

---

## What we're building right now (phased — current phase in bold)

### Phase 0: Scaffolding ✓
Project structure, OANDA wrappers, indicator engine, SQLite journal,
loguru logger, risk-guard skeleton, position-sizing math, Dukascopy
downloader, backtest harness scaffold, CI-style pre-commit automation.

### Phase 1: External knowledge base ✓
Local shallow-clones of every library we depend on under
`fx-scalper/docs/external/`. Context7 MCP for on-demand queries.
See [`fx-scalper/docs/external/INDEX.md`](fx-scalper/docs/external/INDEX.md).

### **Phase 2: Exploratory strategy sweep (in progress)**
See [`DECISIONS/0003-exploratory-phase-scope.md`](DECISIONS/0003-exploratory-phase-scope.md).

Test multiple signal families head-to-head on EUR/USD over 2023-01 →
today. Score each by profit factor, monthly win rate, trades/day,
skewness, recovery factor, total return, Sharpe. Walk-forward required,
but **no hard risk filters during this phase** — report the full
distribution and let the data pick the winner.

Families under test: pullback-to-EMA, ATR-contraction breakout, VWAP
deviation reversion, fast/slow EMA cross, Bollinger + RSI mean reversion
(one of several, not the plan), RSI extreme oscillator.

All share a common exit framework: configurable initial SL, fixed TP,
trailing stop variants, and optional "take partial at 1R, trail rest"
to realize the "base hit + occasional homerun" profile.

### Phase 3: Formalize rules (post-exploration)
Given the ranked results from Phase 2, pick the top 3 candidates and
write their production rules as concrete code. Re-validate each with
a fresh walk-forward pass to rule out selection bias. Compute the
realistic drawdown / loss-streak distribution → set real circuit
breaker thresholds based on those numbers.

### Phase 4: NautilusTrader validation
Port the winning strategy to NautilusTrader with L1 FillModel. If the
vbt-to-Nautilus Sharpe degradation exceeds 30%, the strategy is fragile
— iterate. >50% degradation → kill and return to research.

### Phase 5: Paper trading
OANDA practice account, minimum 14 days. Compare paper results to
backtest OOS within 1σ on all reported metrics. If tracking, proceed.

### Phase 6: Live with $500
Flip `.env` to live. Run with real money. Do NOT add capital until
30+ days of live results track paper results.

### Phase 7+: Scale
Larger account sizes, possibly additional pairs, per results.

---

## Project structure

```
fx-scalper/
├── .env.example
├── .gitignore
├── README.md
├── pyproject.toml
├── requirements.txt
├── config/
│   ├── settings.py           # config + risk params + circuit-breaker placeholders
│   └── secrets.py            # .env loader
├── data/                     # gitignored
│   ├── raw/                  # Dukascopy tick CSVs
│   └── processed/            # Parquet M1 bid/ask bars
├── docs/
│   ├── external/             # cloned upstream library source
│   ├── research/             # strategy exploration write-ups
│   ├── journal_manual.md     # human-authored narrative
│   ├── strategies_manual.md  # per-strategy commentary
│   └── runbook_manual.md     # operational procedures
├── logs/                     # events.jsonl (committed) + rotating logs (not)
├── src/
│   ├── oanda/                # client, data, orders, account, instruments
│   ├── indicators/           # pandas-ta-classic wrapper
│   ├── strategies/           # family modules; common exit framework
│   │   ├── exits.py          # shared SL/TP/trail primitives
│   │   └── families/         # one module per signal family
│   ├── backtest/             # data_loader, dukascopy_client, harness,
│   │   │                      #   explorer, metrics, registry
│   │   └── ...
│   ├── live/                 # bot, trailing, risk, sizing
│   └── utils/                # logger, journal, diary
├── scripts/                  # pull_dukascopy, run_exploration,
│                             #   run_backtest, run_paper, run_live,
│                             #   render_* (JOURNAL/STRATEGIES/RUNBOOK)
└── tests/
```

---

## Position sizing (user-stated, coded verbatim in `src/live/sizing.py`)

Every trade commits **$100 cash**. Leverage is **50:1**. Size is derived
from committed cash, never from risk-percentage formulas.

```python
def compute_position_units(
    cash_committed_usd: float = 100,
    leverage: int = 50,
    *,
    current_price: float,
    instrument: str,
    account_currency: str = "USD",
) -> int:
    notional_usd = cash_committed_usd * leverage   # $5,000
    base_ccy, quote_ccy = instrument.split("_")
    if base_ccy == account_currency:
        return int(notional_usd)                    # USD_JPY → 5,000 USD units
    if quote_ccy == account_currency:
        return int(notional_usd / current_price)    # EUR_USD → 4,630 EUR units at 1.08
    return int(notional_usd / current_price)
```

Expected on EUR/USD at 1.0800: **4,629 units (~$0.46/pip).**

---

## Code standards

- `from __future__ import annotations` at the top of every module.
- **Type hints everywhere.** Google-style docstrings on public functions + classes.
- **No `print()` in `src/`** — use loguru via `src/utils/logger.py`.
- **No magic numbers in strategy files.** All config in `config/settings.py`.
- **Secrets in `.env` only.** `.env.example` shows structure, never values.
- **tz-aware UTC** on every timestamp that crosses a module boundary.
- **Never compute signals on the forming bar** — `iloc[-2]` is the rule when the last bar may be incomplete.
- **Every OANDA order carries** `clientExtensions.id` with our magic number + strategy name + trade UUID.
- **Parse every OANDA error** — they have rich error codes; log them, don't swallow.
- **Journal every order** request AND response to SQLite (`src/utils/journal.py`).
- **Tests that aren't flaky.** Mock OANDA for units. Integration tests hit practice account only.
- **Pydantic models** for any data crossing the OANDA boundary.

---

## OANDA-specific gotchas (factual — keep in module comments)

1. **Instrument naming** is underscore-separated: `EUR_USD`, `GBP_USD`, `USD_JPY`.
2. **Units, not lots.** 100,000 units = 1 standard lot. $5,000 notional on EUR/USD @ 1.08 ≈ 4,630 units.
3. **Pip location varies.** EUR/USD pip = 0.0001 (location −4). USD/JPY pip = 0.01 (location −2). Fetch from instrument metadata, don't hardcode.
4. **Precision on stops/limits.** Submit at instrument's `displayPrecision`.
5. **Streaming API is separate from REST.** `PricingStream` for live prices, REST for orders.
6. **Rate limit: 120 req/sec per account.** Generous; batch where it makes sense.
7. **Trailing stops can be server-side** via `trailingStopLossOnFill` — or client-side for finer control. Decide per strategy.
8. **Margin rates vary by pair.** Fetch current rates, don't assume 2% flat.
9. **FIFO enforcement in US accounts.** Can't partial-close positions arbitrarily. Use full-close + re-enter when needed.
10. **No hedging in US accounts.** Opposing same-instrument positions net out.
11. **Weekend gap risk.** Positions can gap through stops — flat-by-Friday operational breaker handles this.
12. **Practice account resets.** Unlimited. Use aggressively during development.

---

## Environment setup checklist

- [ ] OANDA practice account created (free; separate from live)
- [ ] API token generated (OANDA dashboard → Manage API Access)
- [ ] Account ID noted (format: `001-001-XXXXXXX-001`)
- [ ] Values pasted into `fx-scalper/.env` — OANDA_API_KEY, OANDA_ACCOUNT_ID, OANDA_ENVIRONMENT=practice
- [x] Private GitHub repo created + Zchasse63/shiny-octo-invention
- [x] Python 3.11+ installed via brew
- [x] vectorbt Pro access confirmed; installed into venv
- [ ] Linux VPS provisioned (Contabo, Vultr, etc.) — can wait until Phase 5
- [ ] OANDA live account opened + funded with $500 — only after Phase 5 passes

---

## Reference documents

- [`CLAUDE.md`](CLAUDE.md) — this file, the North Star (user-authored intent only)
- [`CONVENTIONS.md`](CONVENTIONS.md) — operational rules, memory stack, automation, checklists
- [`JOURNAL.md`](JOURNAL.md) — auto-rendered event log
- [`STRATEGIES.md`](STRATEGIES.md) — auto-rendered from code
- [`RUNBOOK.md`](RUNBOOK.md) — auto-rendered config snapshot + human ops prose
- [`DECISIONS/`](DECISIONS/) — numbered ADRs
- [`fx-scalper/docs/external/INDEX.md`](fx-scalper/docs/external/INDEX.md) — local knowledge base
- [`fx-scalper/docs/research/`](fx-scalper/docs/research/) — strategy exploration write-ups
- [`fx-scalper/docs/research/prior_assumptions_archive.md`](fx-scalper/docs/research/prior_assumptions_archive.md) — material stripped from earlier CLAUDE.md rev

## Policy on this file

**CLAUDE.md is authored by the user, not by Claude.** Claude may propose
edits as draft diffs for user review but does NOT silently rewrite it.
Strategy choices, parameter values, threshold settings, or evaluation
criteria do NOT belong in this file unless the user typed them here
themselves. They belong in ADRs or `docs/research/`.
