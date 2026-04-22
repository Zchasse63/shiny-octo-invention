# Session Synthesis — 2026-04-22

**Purpose:** Version-controlled rollup of the EUR/USD exploratory backtesting
session covering rounds 1 → 5, plus a context-independent Recursive Research
Prompt suitable for handing off to a fresh web-enabled model instance.

---

## Part 1 — Research Report

### 1.1 Project context (one paragraph)

`fx-scalper` is a Python trading system targeting EUR/USD, GBP/USD, USD/JPY
on 30-second to 5-minute timeframes through OANDA's v20 API, starting with
$500 of real capital at 50:1 leverage ($100 committed cash per trade →
$5,000 notional, ~$0.46/pip on EUR/USD). Exit philosophy is "base hits plus
trailing runners" — tight initial stops, small fixed take-profits on the
majority of trades, chandelier or ATR-based trailing on the rest. Trades
are never closed by time, only by price. We are in Phase 2 (exploratory
strategy sweep) and have deliberately disabled the production circuit
breakers during discovery per ADR 0003.

### 1.2 Summary of rounds 1 → 5 (EUR/USD only)

| Round | Focus | Families | TFs | Winners (PF>1.2 OOS) | Top PF | Top $/trade |
|---|---|---|---|---|---|---|
| 1 | Unfiltered families | 6 base families | M1 | 0 | <1.0 | -$3.20 |
| 2 | + ADX / session / spread filters | 8 (6 + 2 filtered) | M1 | 11 | 1.41 | +$1.80 |
| 3 | + M5 / M15 / M30 timeframes | 8 | M5, M15, M30 | 63 | 1.83 | +$11.25 |
| 3.5 | Meta-analysis (arithmetic fixup) | — | — | — | — | — |
| 5 | + weekday filter + finer sessions | 8 | M5, M15, M30 | 147 | **2.07** | **+$20.14** |

(Round 4 — cross-pair validation on GBP/USD + USD/JPY — is blocked on
Dukascopy backfills still in progress.)

### 1.3 Consistent patterns across all rounds

1. **Session filter is the dominant edge, not the indicator.** Every profitable
   configuration in rounds 2-5 uses an explicit session window. Unfiltered
   (`session="all"`) configs materially under-perform at every timeframe.
   The `london_ny_overlap` (12-15 UTC) window produced the top PF in every
   round; `active` (7-16 UTC) produced the broadest winner count.

2. **Timeframe determines the winning family.**
   - **M5:** filtered mean-reversion (bb_rsi_mr_filtered + rsi_extreme_filtered)
   - **M15:** bb_rsi_mr dominates (9 of top 10 round-5 configs)
   - **M30:** rsi_extreme unfiltered + bb_rsi_mr mix
   - **M1:** unprofitable after spread regardless of family — confirms the
     M1 spread/edge ratio destroys any naïve entry signal.

3. **Exit convergence.** Nearly every top config uses ATR-based initial SL
   (0.5-1.0× ATR) with chandelier or percent trailing (2× ATR / 2.5% trail).
   Pure fixed-TP-only exits under-perform trailing exits by a consistent
   10-25% in PF.

4. **ADX filter is usually useless.** The `max_adx=None` arm beats
   `max_adx=25` in the majority of round-2+ winners, contradicting the
   round-1→2 hypothesis that ranging-regime gating was the missing piece.
   The regime filter that matters is **time-of-day**, not ADX.

5. **Weekday filter delivers a real (if modest) lift.** `tue_fri` > `all`
   in round 5 by ~0.15 PF points on top configs. `tue_thu` (mid-week-only)
   over-restricts and kills trade count — Monday and Friday carry real
   volume. `mon_thu` is the second-best weekday preset.

### 1.4 Round-5 top production candidate (EUR/USD, not yet cross-pair validated)

| Field | Value |
|---|---|
| Family | `bb_rsi_mr_filtered` |
| Timeframe | **M15** |
| BB length / std | 20 / 2.25 |
| RSI length / long / short | 14 / 25 / 75 |
| ADX filter | off |
| Session | `london_ny_overlap` (12-15 UTC) |
| Weekday | `tue_fri` |
| Spread filter | 0.25× ATR max |
| Stop loss | 0.5× ATR initial |
| Take profit | 1.5R fixed |
| Trail | Chandelier 2× ATR once 1R is reached |
| **OOS PF (mean 3 splits)** | **2.07** |
| **OOS win rate** | **78%** |
| **OOS expectancy** | **+$9.29 / trade** |
| **OOS max DD** | **32%** |
| OOS trades (per split, avg) | 51 |
| Coverage | 1.65 years OOS total |
| Implied annual profit on $500 | ~$1,496 (corrected per round 3.5) |

### 1.5 Information Gap Assessment

What we **don't yet know** that could invalidate the above:

1. **Cross-pair robustness.** Every winning config was selected on EUR/USD.
   Round 4 (GBP/USD + USD/JPY walk-forward) is our binary gate; top configs
   must stay PF > 1.0 with positive expectancy on both pairs, or they're
   overfit.

2. **Per-trade MAE / MFE.** We have never captured `pf.trades.records_readable`,
   so we cannot verify whether stops are sized correctly relative to typical
   adverse excursion. Stop tightening/widening is currently guessed.

3. **Strategy correlation.** We assume M5 + M15 + M30 winners are uncorrelated
   enough to run in parallel — we have not measured this. Requires a single
   `Portfolio.from_signals` call with `cash_sharing=True` across the three
   top configs.

4. **Forward (live or paper) test.** Zero. All performance numbers are
   walk-forward OOS on historical Dukascopy ticks; broker execution realism
   (slippage, requotes, partial fills, weekend gaps) has not been stressed.
   NautilusTrader L1 FillModel validation is Phase 4 and has not started.

5. **Multi-testing honesty.** 147 winners / 8,690 configs = 1.7%. At random
   with a PF > 1.2 gate we'd expect ~5% to pass. Structural patterns
   (bb_rsi_mr + london_ny_overlap + tue_fri concentration) argue for real
   signal, but a bootstrap CI on PF has not been computed.

6. **Capital-scale interaction.** 32-44% max DD on winners would trip the
   current $400 circuit-breaker on a $500 account. Either halve position
   size to $50 margin or start at $1,500+. Not yet decided.

> **Note on news / macro-event overlays:** explicitly **out of scope** for
> now per user direction. Fundamentals and macro-event filters will be
> revisited only after the pure technical / indicator-side system is
> validated end-to-end. The Recursive Research Prompt and the round 6-8
> plan below reflect this.

### 1.6 Items explicitly deferred to later rounds

- **Round 4:** cross-pair validation (blocked on backfills).
- **Round 6:** position-sizing variants via `adjust_func_nb` — Kelly fraction,
  ATR-inverse, losing-streak contraction.
- **Round 7:** per-trade diagnostics — MAE/MFE, R-multiple distribution.
- **Round 8:** native portfolio correlation with `cash_sharing=True`.

---

## Part 2 — Recursive Research Prompt (context-independent)

Copy the block below into a fresh, web-enabled model instance. It is
self-contained — no chat history required.

```
You are a quantitative research reviewer. I am building a Python forex
scalping bot ("fx-scalper") that trades EUR/USD, GBP/USD, USD/JPY through
OANDA's v20 REST/streaming API. The account starts at $500 USD real capital,
commits $100 cash per trade at 50:1 leverage ($5,000 notional, ~$0.46/pip
on EUR/USD), and aims for "base hits + trailing runner" exits — tight
initial stops, small fixed take-profits, chandelier / ATR trailing on the
rest. Trades close only on price (stop or trail hit), never on time.

Stack: vectorbt Pro, pandas-ta-classic + TA-Lib, Dukascopy institutional
bid/ask tick data 2023-01 → 2026-04 resampled to M1/M5/M15/M30, walk-forward
3-window 50/50 OOS validation, costs modelled with realistic OANDA spreads
(~0.6 pip EUR/USD, ~0.9 pip GBP/USD, ~0.9 pip USD/JPY) and slippage.

After 5 rounds of exploration on EUR/USD only, my current best production
candidate is:

  Family:       bb_rsi_mr_filtered (BB + RSI mean reversion)
  Timeframe:    M15
  Entry:        BB(20, 2.25) RSI(14) thresholds 25/75
  Filters:      session=london_ny_overlap (12-15 UTC), weekday=tue_fri,
                spread ≤ 0.25 × ATR, ADX filter DISABLED
  Exits:        SL 0.5× ATR initial, TP 1.5R fixed, trail chandelier 2× ATR
  OOS results:  PF 2.07, WR 78%, expectancy +$9.29/trade, max DD 32%,
                ~51 trades per 0.55-yr split, 1.65-yr total OOS coverage.

Consistent patterns across rounds 1-5:
  - Session filter is the dominant edge (not indicator choice)
  - london_ny_overlap (12-15 UTC) and `active` (7-16 UTC) dominate
  - Timeframe selects the family: MR at M5-M30, momentum/breakout at H1
  - Chandelier / ATR trailing exits beat pure fixed-TP by 10-25% PF
  - Weekday filter tue_fri > all > tue_thu (too restrictive)
  - ADX filter is empirically useless (max_adx=None beats max_adx=25)
  - M1 is unprofitable after spread regardless of family

Known gaps (not yet validated):
  - Cross-pair (GBP/USD, USD/JPY) — walk-forward pending backfills
  - Per-trade MAE/MFE — never captured; stop sizing is guessed
  - Strategy correlation — assumed, not measured
  - Broker execution realism (slippage, requotes, weekend gaps) unstressed
  - Bootstrap CI on PF not computed; multi-testing caveat noted (147/8690 = 1.7%)

IMPORTANT SCOPE CONSTRAINT: news / economic-calendar / macro-event
filtering is DELIBERATELY OUT OF SCOPE for this research pass. The
project is focused on technical / indicator-side fundamentals first;
news overlays will be revisited only after the purely technical system
is validated end-to-end. Do NOT recommend news-based filters, event
calendars, NFP/FOMC blackouts, or similar in your answer.

Please do the following four tasks, citing web sources where relevant:

  TASK 1 — CLAIM VALIDATION.
  Is a BB+RSI mean-reversion strategy with a 12-15 UTC ("London/NY overlap")
  session filter on EUR/USD M15 with PF ~2.0 / WR ~78% / max DD 32% a
  plausible real edge, or is this a known overfitting pattern? Cite
  published research (academic, SSRN, prop-shop blogs, quant Twitter) on:
    (a) whether session-of-day filters genuinely persist out-of-sample
        for FX mean reversion
    (b) what realistic PF / WR / DD ranges are for a ~100-trade/year
        intraday FX MR strategy after real costs
    (c) any evidence that the 12-15 UTC window has structural reason
        (London close + NY lunch liquidity dynamics?) vs being a lucky slice.

  TASK 2 — PURELY-TECHNICAL REGIME DIAGNOSTICS.
  The EUR/USD backtest covers 2023-01 → 2026-04. Without invoking news
  or economic calendars, what PURELY TECHNICAL / PRICE-DERIVED regime
  indicators (realized-vol percentile, ATR regime, range vs trend
  classifiers, autocorrelation of returns, Hurst exponent, volume /
  tick-count regime, session-level range compression, etc.) should I
  monitor to know when this system's edge is degrading in live trading?
  Give me 3-5 concrete price-only diagnostics I can compute every bar.

  TASK 3 — MISSING INFRASTRUCTURE.
  Given the known gaps above, what pieces of infrastructure would you
  build BEFORE committing real $500? Rank them by cost/benefit and give
  concrete suggestions (library names, patterns, papers). Specifically
  address: (a) per-trade MAE/MFE + R-multiple distribution capture,
  (b) strategy correlation measurement across timeframes,
  (c) broker-realism validation beyond vectorbt's cost model (slippage
  models, partial fills, weekend gaps, fill-latency simulation),
  (d) bootstrap / Monte-Carlo confidence intervals on PF and expectancy.
  DO NOT recommend news-calendar or event-overlay infrastructure —
  that is deliberately out of scope for this pass.

  TASK 4 — RED FLAG CHECK.
  What am I missing on the TECHNICAL / EXECUTION side? What would cause
  this to fail in live trading even if backtest numbers are honest?
  Prioritize by likelihood × impact. Consider: overnight roll costs,
  swap / carry rates, OANDA-specific execution quirks (FIFO US rules,
  no hedging), weekend gap risk, the gap between tick-level Dukascopy
  backtest and real OANDA fills, the role of 50:1 leverage in amplifying
  a 32% DD into account blowup, and any behavioral factors unique to
  $500-capital retail FX. Do NOT include news / event-driven factors
  in this answer — those are out of scope for this pass.

Deliverable format:
  A — Verdict on the current candidate: LIKELY_REAL / LIKELY_OVERFIT /
      NEEDS_MORE_DATA, with 3-5 line justification.
  B — Five specific papers / blog posts / open-source projects you'd
      have me read before Round 6.
  C — A prioritized list of missing experiments (with specific parameters)
      that would materially change your verdict.
  D — Top 3 red flags ranked by expected-loss.
  E — Go/Hold/Kill recommendation on proceeding to cross-pair validation
      (Round 4) vs. pivoting to a different approach, with reasoning.
```

---

## Part 3 — What happens next in this codebase

1. **Block on vbt.chat round 5 query** (background) — its output gets
   appended to `docs/research/round5_findings.md` §9.
2. **Block on backfills** — `pull_dukascopy` for GBP/USD and USD/JPY
   must reach 2026-04.
3. **Execute round 4** — `scripts/run_cross_pair_validation.py` on the
   top-10 round-5 configs across all 3 pairs. Hard gate: PF > 1.0 + positive
   expectancy on all 3 pairs.
4. **If round 4 passes,** proceed to round 6 (position sizing). If round 4
   fails on GBP or JPY, the "bb_rsi_mr + overlap" candidate is an EUR/USD
   idiosyncrasy — pivot to per-pair optimization or kill.
5. **If and only if rounds 4 + 6 + 7 + 8 all clear,** freeze params, port
   to NautilusTrader (Phase 4) for execution-realism validation.
