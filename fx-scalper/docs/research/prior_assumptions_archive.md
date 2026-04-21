# Archive: prior-Claude injected material from CLAUDE.md

This file preserves — verbatim — the sections of CLAUDE.md that the user
later identified as prior-Claude-session assumptions, not their stated
requirements. Kept here for traceability and so we can re-examine any
assumption if a later decision surfaces it.

Archive date: 2026-04-21
Trigger: user message "I never asked for the BB-RSI plan, claude assumed
and put that in there. Anything other than what I described was injected
in."

---

## What was user-stated (retained in new CLAUDE.md)

- Mission and target markets (EUR/USD, GBP/USD, USD/JPY, OANDA, 30s–5min)
- Account sizing: $500 start, $100/trade margin, 50:1 leverage, $5,000 notional
- Dynamic trailing stops; exits on price action or SL, never time
- Honest backtests with real spreads + slippage
- Dev on macOS, production on Linux VPS
- Code hygiene: tz-aware UTC, magic number per order, SQLite journal,
  no signals on forming bar, pydantic at API boundaries, type hints,
  google-style docstrings
- High-frequency scalping: multiple trades/hour expected
- "Base hits with occasional trailing-stop runners" exit philosophy

## What was injected and archived below

- Specific starter strategies (BB-RSI MR, trend-filtered momentum,
  London-NY breakout)
- Parameter grids for each strategy
- Candidate criteria (OOS Sharpe > 0.5, max DD < 15%)
- Specific circuit-breaker threshold values ($400 floor, $50 daily,
  $30 single trade, 3 consec losses)
- The day-by-day implementation plan as it pertained to strategy choices
- "Red-flag check: if Strategy 2 OOS beats Strategy 1 OOS, flag overfitting"
  — this was opinion, not fact

---

## Archived section 1: "Starter Strategies" (lines 74–100 of original CLAUDE.md)

> All three have academic or rigorous backtest support. Implement in order.
>
> **Strategy 1: Bollinger-Band + RSI Mean Reversion** (strongest evidence)
> - Pairs: EUR/USD, USD/JPY
> - Timeframe: M5 or M15
> - Session: Asian (23:00–07:00 UTC) preferred
> - Signal:
>   - Long: close < lower BB(20, 2.0) AND RSI(14) < 30 AND ADX(14) < 20
>   - Short: close > upper BB(20, 2.0) AND RSI(14) > 70 AND ADX(14) < 20
> - Exit: BB midline, opposite band, RSI back through 50, or ATR-based TP
> - SL: 1.5× ATR(14)
> - Trail: 2× ATR, tighten to 0.5× ATR if RSI reverses through 50
>
> **Strategy 2: Trend-Filtered M15 Momentum**
> - Pair: GBP/USD primary
> - Filter: price vs EMA200 on H1
> - Signal: RSI(14) crosses 50 in trend direction, ADX(14) > 25
> - SL: 1.5× ATR, TP: 2.5× ATR
> - Trail: Chandelier exit (highest_since_entry − 3× ATR for longs)
>
> **Strategy 3: London–NY Overlap Range Breakout**
> - Pair: GBP/USD primary, EUR/USD secondary
> - Mark London session range: 08:00–12:00 UTC
> - Trade breakout during overlap: 12:00–16:00 UTC
> - Fixed 2:1 reward:risk, hard time exit at 16:00 UTC

These are now three of seven-plus signal families being tested in the
exploratory sweep (see ADR 0003). No strategy is "the plan" anymore.

---

## Archived section 2: Specific circuit-breaker thresholds

> 1. **Account floor:** If account NAV drops below `$400`, halt bot, close all positions, send alert.
> 2. **Daily loss limit:** If realized + unrealized PnL for current UTC day ≤ `-$50`, halt until 00:00 UTC next day.
> 3. **Consecutive losses:** If last 3 trades closed at loss, pause new entries for 1 hour. Trailing on existing positions continues.
> 4. **Single-trade blowout:** If any open position shows unrealized loss > `$30`, halt bot, alert, do NOT auto-close (investigate first).
> 5. **OANDA disconnect:** If API fails 3 consecutive polls, halt new entries, continue trying to manage open positions, alert.
> 6. **Sunday/Friday boundary:** No new entries 17:00 ET Friday → 17:00 ET Sunday. Close all positions by 16:55 ET Friday.

The CONCEPT of circuit breakers is retained. The specific THRESHOLDS
(the dollar values and counts) were never user-approved. They're
parked until exploration results tell us what "normal drawdown" and
"normal consecutive-loss streak" actually look like for the chosen
strategy. Some values (like #5 OANDA disconnect and #6 weekend gap
handling) are operationally sensible regardless of strategy; those
remain active in code but with the threshold values treated as
defaults that may be revised.

---

## Archived section 3: Day-by-day strategy plan

> ### Day 4: Strategy 1 (Bollinger + RSI Mean Reversion)
> - Implement `src/strategies/base.py` abstract class
> - Implement `src/strategies/bb_rsi_mr.py` per spec above
> - Parameter sweep grid:
>   - BB length: {15, 20, 30}
>   - BB std: {1.8, 2.0, 2.2}
>   - RSI length: {10, 14, 21}
>   - RSI threshold: {65/35, 70/30, 75/25}
>   - ADX threshold: {18, 20, 22}
> - 243 combos × 3 pairs = 729 backtests
> - Walk-forward: train 2023, test 2024–2025
> - Candidate criteria: OOS post-cost Sharpe > 0.5, max DD < 15%
> - Output: CSV of top 20 parameter sets, equity curves, drawdown charts
>
> ### Day 5: Strategy 2 (Trend-Filtered Momentum)
> - Implement `src/strategies/trend_momentum.py` per spec
> - Same harness, same sweep structure
> - **Red-flag check:** If Strategy 2 OOS beats Strategy 1 OOS, flag overfitting — literature says simple momentum on minutes is dead. Investigate before trusting.
>
> ### Day 6: Strategy 3 (Session Breakout)
> - Implement `src/strategies/session_breakout.py`
> - Test three variants: London-only, NY-only, London-NY overlap
> - Expect naive London-open on EUR/USD to lose money (calibration test)
> - Single-asset backtests acceptable

Replaced by **Phase 2: Exploratory Strategy Sweep** (see ADR 0003) —
test many signal families head-to-head, let OOS data pick. No
pre-commitment to any specific strategy.

---

## Archived section 4: Opinionated filter criteria

The phrase "Candidate criteria: OOS post-cost Sharpe > 0.5, max DD < 15%"
was a hard gate. Sharpe alone is not appropriate for the "multiple
trades per hour, base hits + trail" profile the user actually wants.
Replaced by a multi-metric report (profit factor, monthly win rate,
trades/day, skewness, recovery factor, total return, Sharpe)
ranked but not gated. See ADR 0003.

---

## The stub strategy files in `src/strategies/`

`src/strategies/bb_rsi_mr.py` was fully implemented based on the above
archived spec. It remains in the codebase but is now **one of several
families being tested**, not the primary strategy. `trend_momentum.py`
and `session_breakout.py` were stubs referencing the archived specs;
they'll be replaced by the six-family exploration in Phase 2.
