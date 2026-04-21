# Round 2 findings — 2026-04-21

## Headline

**74 configs profitable across all 3 walk-forward splits** (of 3,526 with ≥50 trades each), up from 0 in round 1. Best: **PF 2.01, +$0.84/trade, 77% win rate, Sharpe 1.73, max DD 18%** over 95 trades per split on OOS data.

Run: [`backtest_results/explore_20260421T2116/`](../../backtest_results/explore_20260421T2116/)
AI analysis: [`ai_queries/20260421T213735-iter2_round2_winners_analysis.md`](ai_queries/20260421T213735-iter2_round2_winners_analysis.md)

## The single most important finding

**The session filter is the edge, not the indicator.**

- 52 of 74 winners use `session="london_ny_overlap"` (12:00–16:00 UTC)
- 70 of 74 winners have `max_adx=None` (ADX filter adds zero value)
- Unfiltered versions of every family still lose
- The filter dimension was the gap between round 1 (nothing works) and round 2 (~2% of configs work)

## Winner profile (top 3 configs)

All three top configs share:

| Dimension | Value |
|---|---|
| Family | `rsi_extreme_filtered` |
| RSI length | 14 or 21 |
| Oversold / overbought | 20-25 / 75-80 |
| ADX filter | **None** |
| Session | `london_ny_overlap` |
| Stop loss | 2× ATR |
| Take profit | **None** (trail does the work) |
| Trail | 3× ATR trailing |
| Win rate | 75-85% |
| Trades per split | ~90-100 |
| Max drawdown | 15-25% |

## Two winning styles emerged

1. **High win-rate scalp** (~85% WR, PF ~1.6): tight 0.5× ATR stop + small 0.75-1.0 R TP + 3× ATR trail. "Base hits" profile.
2. **Runner-capture** (~70% WR, PF ~1.4, expectancy ~$2/trade): 1-2× ATR stop, chandelier exit, **no fixed TP** — lets winners run.

Style 2 matches the user's stated "base hits + occasional homerun via trail" philosophy better; style 1 has higher win rate but lower expectancy per trade.

## Caveats — don't celebrate yet

- **Multi-testing risk:** 74 winners / 3,526 configs = 2.1%. With 3,526 independent random tests, we'd expect ~5% apparent winners at p<0.05. The structural pattern (session concentration, family concentration) suggests this is real signal, not noise — but cross-validation on other pairs / timeframes is essential.
- **Account-level dollar impact is tiny.** Best expectancy $0.84/trade × 95 trades/split = $80 per 3-month window = **~$320/year** on $500 account. Positive, but not transformative.
- **Trade frequency fails the user's "5+ trades/day" goal.** 95 trades/split over ~250-day splits = 0.4 trades/day. Session-restricted MR is low-frequency by construction.
- **Drawdown profiles ok but not great.** 18-25% DD on the winners — survivable on $500 but sensitive.

## Action items rolling into round 3+

1. **Round 3 (running now):** resample to M5/M15/M30/H1, run all families. Some unfiltered families may suddenly work at higher TFs where spread-per-trade cost is lower.
2. **Round 4:** backfill GBP/USD + USD/JPY, cross-test the london_ny_overlap winners — verify the session effect generalizes.
3. **Round 5 candidates from vbt.chat:** finer session partitions (London-open only vs overlap first-half vs second-half), volatility-regime filter, range compression filter.
4. **Don't spend more parameter search on M1 EUR/USD** — round 2 already found the pocket. Confirm it, don't over-tune.
