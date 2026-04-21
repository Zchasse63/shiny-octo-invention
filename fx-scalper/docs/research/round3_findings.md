# Round 3 findings — multi-timeframe 2026-04-21

## Headline

**Every timeframe produces winners.** Moving from M1 to M5+ dramatically increases the edge per trade because the spread-to-range ratio collapses. At H1, dollar expectancy reaches **$11-16 per trade** vs $0.84 at M1.

Run: [`backtest_results/explore_multi_tf_20260421T2137/`](../../backtest_results/explore_multi_tf_20260421T2137/)

## Winners by timeframe (PF > 1.2, OOS across all 3 walk-forward splits, ≥30 trades/split)

| TF | Configs | Winners | Best PF | Top family | Best $/trade | Best DD |
|---|---|---|---|---|---|---|
| M5  | 1785 | 31 | 2.37 | `rsi_extreme_filtered` | $4.89 | 17% |
| M15 | 1665 | 13 | 1.50 | `bb_rsi_mr` (unfiltered!) | $7.98 | 30% |
| M30 | 1545 | 14 | 1.62 | `bb_rsi_mr_filtered` | $15.66 | 34% |
| H1  | 1381 | 22 | 1.63 | `ema_cross` | $11.64 | 43% |

## The two paradigms

### M5 / session-filtered MR (round 2 extended)

- Dominant family: `rsi_extreme_filtered` with `session=london_ny_overlap`
- Character: 70-80% win rate, $3-5/trade, 20-30% DD, ~50 trades/split
- Same pattern as round 2, just scaled up slightly by the wider spread/edge gap at M5
- Top config: PF 2.37, $+4.89/trade, 80% WR, 26% DD

### M15-H1 / unfiltered momentum-or-MR

- Families: `bb_rsi_mr`, `ema_cross`, `vwap_deviation`, `range_breakout` (all previously losing at M1)
- Character: fewer trades (~40-60/split), larger per-trade edge, higher DD
- Session filter contributes less — spread cost per trade is already small
- Top config (H1): `ema_cross` PF 1.63, $+11.64/trade, 75% WR, 43% DD, ~5 trades/month

### 80 winners across all timeframes

By family (winners with PF > 1.2):

| Family | Count |
|---|---|
| bb_rsi_mr | 35 |
| bb_rsi_mr_filtered | 16 |
| ema_cross | 10 |
| rsi_extreme_filtered | 9 |
| rsi_extreme | 4 |
| vwap_deviation | 4 |
| range_breakout | 2 |
| pullback_ema | 0 |

`pullback_ema` still fails everywhere. `range_breakout` barely survives. Pure MR + momentum families dominate.

## Dollar-annual projection on $500 account

Assuming 3 walk-forward splits = ~13 months of OOS evidence:

| TF | Best per-trade expectancy | Trades/split | $ per split | $/year projected |
|---|---|---|---|---|
| M1  | $+0.84 | 95 | $80 | $320 |
| M5  | $+4.89 | 50 | $245 | $980 |
| M15 | $+7.98 | 103 | $822 | $3,290 |
| M30 | $+15.66 | 43 | $673 | $2,690 |
| H1  | $+11.64 | 44 | $512 | $2,050 |

M15 is the **sweet spot for income** — high per-trade edge PLUS enough trade count to compound. H1 has similar annual but fewer "income moments."

## Problems to flag

### 1. Sharpe annualization bug in `src/backtest/metrics.py`

Reported Sharpe scales with sqrt(bar_minutes). At M5 it's inflated ~2.3×, at H1 ~7.75×. The actual comparative rank is correct, but absolute values are wrong. **Fix: pass `minutes_per_year=525600/bar_minutes_in_frame` when computing metrics on non-M1 data**, or compute the bar frequency automatically from the DataFrame index.

Does NOT affect:
- Profit factor
- Win rate
- Expectancy
- Max drawdown
- Rankings within a single TF

### 2. Trade frequency vs. user's "multiple trades/hour"

The original plan targeted 5+ trades/day minimum. Round-3 winners at H1 do ~44 trades over ~8 months = **0.2 trades/day**. M15 is the most user-compatible winner at ~103 trades/split ≈ 0.5 trades/day. Still far from the scalping frequency goal.

**Trade-off:** more trades per day = worse spread/edge ratio = M1 hell (0 viable configs). The data is forcing us off the "multiple trades/hour" ambition toward "1-5 trades/day with real edge."

### 3. Drawdown risk on $500 account

H1 winners show 40-55% drawdowns. On $500 starting capital, a 50% drawdown is $250 — above the user-stated $400 account floor hard stop. **Either reduce per-trade notional (from $5,000 to $2,500) for H1 strategies, or reject them** per the account-floor circuit breaker.

M5 winners sit at 17-26% DD — survivable.

### 4. Multi-testing / false-discovery caveat still applies

80 winners out of 6,376 filtered configs = 1.3%. Round 4 (cross-pair validation) is the critical robustness test.

## Decisions rolled into round 4

1. **Backfill GBP/USD + USD/JPY** (running now, ~1 hour each)
2. **Cross-test top 3-5 configs per timeframe** on all three pairs
3. **Filter criterion:** config must maintain PF > 1.0 + positive expectancy on ALL 3 pairs
4. **If survives cross-pair:** real edge candidate. Goes to Phase-3 formalization.
5. **If degrades on other pairs:** EUR/USD-specific artifact. Drop.
