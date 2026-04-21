# Strategies — narrative overlay

_Human-authored notes per strategy. The renderer keys each section below by
the strategy's `NAME` attribute and inlines it into `STRATEGIES.md`._

## bb_rsi_mr

**Academic basis.** Mean-reversion on high-frequency FX has the strongest
evidence of the three starter strategies in the literature. Bollinger
extremes combined with RSI give a robust noise filter; the ADX < 20
condition is the no-trend gate that keeps us out of momentum regimes where
MR blows up.

**Failure modes to watch in backtest:**
- **Trend periods.** NFP release, SNB-style interventions, or regime
  breaks can produce sustained BB breakouts. The ADX < 20 filter should
  handle most of these, but OOS drawdowns concentrated around news
  releases are a red flag.
- **Illiquid session boundaries.** 23:00–00:00 UTC Sunday (re-open) has
  wide spreads and stale quotes. Our weekend gate starts Sunday entries
  at `SUNDAY_OPEN_UTC_HOUR` (22:00 UTC) per RUNBOOK, but the first hour
  should be treated skeptically in OOS analysis.
- **Asian session carry.** If London traders revisit overnight levels,
  a MR entry at 06:00 UTC can get stopped out as London trends through.
  Our default `asian_session_only=True` avoids this, but the Day-4 sweep
  will test the `False` case.

**Spread-honest pricing.** SL/TP are anchored to `ask_close` (LONG) or
`bid_close` (SHORT) so backtest metrics match what the broker actually
fills. If the backtest uses mid anywhere for SL/TP, results are ~0.5 pip
optimistic per side, which compounds into a meaningful Sharpe inflation
on 5-minute bars.

**Day-4 sweep targets** (per CLAUDE.md):
- BB length ∈ {15, 20, 30}, std ∈ {1.8, 2.0, 2.2}
- RSI length ∈ {10, 14, 21}, thresholds ∈ {65/35, 70/30, 75/25}
- ADX threshold ∈ {18, 20, 22}
- Pairs: EUR/USD, USD/JPY
- Walk-forward: train 2023, test 2024–2025
- Candidate criteria: OOS post-cost Sharpe > 0.5, max DD < 15%

## trend_momentum

_Day-5 stub. Notes go here when implementation starts._

## session_breakout

_Day-6 stub. Notes go here when implementation starts._
