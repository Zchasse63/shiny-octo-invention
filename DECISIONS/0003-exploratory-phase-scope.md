# ADR 0003: Exploratory phase — scope and rules

Date: 2026-04-21
Status: accepted

## Context

Earlier CLAUDE.md revisions pre-committed to three specific strategy
families (BB-RSI mean reversion, trend-filtered momentum, London-NY
breakout) with fixed parameter grids and hard candidate criteria
(OOS Sharpe > 0.5, max DD < 15%). These specifics were added by a
prior Claude session, not specified by the user. Once identified
(2026-04-21), the user directed:

> "Disregard all previous constraints, safety buffers, or 'circuit
> breaker' thresholds established in prior sessions. The current goal
> is purely exploratory: to identify high-probability trading
> strategies through rigorous backtesting without being limited by
> risk-aversion protocols … Do not filter out strategies simply
> because they exhibit high volatility or temporary losses. I want to
> see the raw potential of the data."

The user's stated profile:

- High-frequency scalping, multiple trades/hour
- Base-hits (steady income) + occasional trailing-stop runners for asymmetric upside
- EUR/USD first; GBP/USD + USD/JPY after
- Small account ($500–$2000)
- Data-driven discovery NOW; strict risk management comes AFTER candidate selection

## Decision

Run an **exploratory phase** with the following rules.

### 1. Test multiple signal families, no pre-commitment

Implement at least six signal families, all pluggable into a common
exit framework:

| Family | Signal logic |
|---|---|
| `pullback_ema` | Enter on pullback to EMA(fast) in direction of EMA(slow) slope |
| `range_breakout` | Enter on break of N-bar range after ATR contraction (Bollinger squeeze) |
| `vwap_deviation` | Fade price when it deviates >Nσ from session VWAP |
| `ema_cross` | Enter on fast/slow EMA crossover |
| `bb_rsi_mr` | Bollinger + RSI extreme reversion (baseline carryover from prior plan) |
| `rsi_extreme` | Enter on RSI oversold/overbought crossover |

### 2. Common exit framework

All families share `src/strategies/exits.py` with parameterized:

- Initial stop loss: `sl_atr_mult × ATR(atr_length)`
- Optional fixed take-profit: `tp_r_mult × initial_risk` (configurable or disabled)
- Optional trailing stop variants:
  - ATR trail: `trail_atr_mult × ATR(atr_length)`
  - Chandelier exit: extreme-since-entry minus trail distance
  - Fixed percent trail
- Optional "take partial at 1R, trail the rest" — implements the user's
  "base hit + occasional homerun" exit philosophy

### 3. Parameter exploration

Each family has its own parameter grid with generous ranges — explicitly
wider than "academic defaults" since the goal is discovery. Each axis
has 4–7 values. Full Cartesian grid is not feasible for every family;
use `vbt.Param(..., _random_subset=N)` for coarse passes and Optuna
for Bayesian refinement where the coarse pass shows promise.

### 4. Evaluation metrics (all reported, none gate)

Every strategy × param combo is scored on:

- **Profit factor** (gross wins $ / gross losses $) — primary rank
- **Win rate** (% of trades profitable)
- **Monthly positive rate** (% of calendar months positive) — consistency signal
- **Trades per day** — validates "scalping frequency" claim
- **Recovery factor** (total return / |max drawdown|)
- **Skewness of per-trade returns** — positive = tail favours winners (matches "homerun" intent)
- **Sharpe ratio** (annualized)
- **Sortino ratio**
- **Max drawdown %**
- **Total return %**
- **Avg holding time (bars)** — useful context, not a gate
- **Expectancy per trade ($)**

### 5. No filters during exploration

Explicitly NOT applied during this phase:

- Sharpe ≥ threshold
- Max DD ≤ threshold
- Minimum win rate
- Minimum trade count
- IS/OOS degradation ratio

**Except:** Walk-forward train/test split IS required. OOS metrics are
what's reported, not IS. This is anti-overfitting hygiene, not risk
filtering. Without it the exercise is noise mining.

### 6. Output format

1. **Full ranked table** — every family × every param combo, CSV at
   `fx-scalper/backtest_results/explore_YYYYMMDD/full_results.csv`.
2. **Per-family best performer** — detailed report per family at
   `explore_YYYYMMDD/per_family/<family>.md`.
3. **Top-N overall by each primary metric** — profit factor, monthly hit
   rate, recovery factor, total return (N=10 each).
4. **Equity curves** for top 5 overall (PNG).
5. **Monthly PnL heatmap** for top 5 overall.
6. **Formalized trading rules** document for top 3 — concrete Python
   code snippets with locked parameters, usable for Phase 3 formalization.

### 7. Circuit breaker values deferred

No dollar thresholds, no consecutive-loss counts, no max-drawdown gates
coded into the backtest itself. Circuit breakers remain a concept
(`src/live/risk.py` still exists with placeholder thresholds) but they
are NOT consulted during backtesting. After Phase 2 completes and we
pick candidates, Phase 3 computes realistic drawdown + loss-streak
distributions from those strategies' actual behaviour and sets
breakers based on the data.

### 8. Ops-level breakers stay active regardless

- OANDA API disconnect handling (not strategy risk — infrastructure)
- Weekend flat-by / no-Sunday-re-open entries (avoids gap risk that
  isn't in any backtest)

These fire in live trading, not backtest. No change.

## Consequences

**Gained:**
- Real picture of what works in YOUR data on YOUR pairs for YOUR profile.
- No pre-commitment to a family that may not be the right shape for scalping.
- Traceability: every decision can point to an actual OOS result.

**Given up:**
- Some discipline around "candidate must beat Sharpe X" is gone during
  exploration. Re-added explicitly in Phase 3 based on observed
  distributions, not guessed thresholds.
- Exploration runtime is longer than a single-family sweep. Acceptable
  given the payoff.
- Risk of analysis paralysis if every family produces similar mediocre
  results — mitigated by a firm stop-date: if no family clears a
  minimum Phase-3 filter, we declare "nothing viable, return to
  research with a different architecture" rather than force a pick.

**Follow-up work:**
- Build `src/strategies/exits.py` (common exit framework).
- Implement the six family modules under `src/strategies/families/`.
- Build `src/backtest/explorer.py` (orchestrator).
- `scripts/run_exploration.py` entry point.
- Unit tests on synthetic data for each family.
- Run on EUR/USD once Dukascopy backfill completes.
- Report results + propose Phase 3 candidate shortlist.

## Alternatives considered

1. **Continue with BB-RSI only.** Rejected — user identified this as
   a prior-session assumption, not their intent.
2. **Pick one "best guess" family based on user's description.** Rejected —
   the user explicitly said "do not want to get tunnel vision". Would
   reintroduce the same assumption bias we just removed.
3. **Run the pure search with risk gates (Sharpe > 0.5).** Rejected —
   user explicitly said "do not apply restrictive loss-mitigation rules"
   during exploration. Risk discipline re-enters in Phase 3.
4. **Skip walk-forward and use full-sample metrics.** Rejected. Walk-forward
   is the single most important tool against selection bias in a multi-
   family multi-parameter sweep. Not optional.
