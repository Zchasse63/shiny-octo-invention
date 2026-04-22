# fx-scalper — Final rollup (Phase 2 close)

**Recommendation: NOT READY FOR PAPER TRADING YET.**

## Executive summary

Six formal rounds + 2 follow-ons (4b, 5.5) + 2 infrastructure rounds
(7 trade-records, 8 portfolio) of walk-forward exploratory backtesting
on EUR/USD / GBP/USD / USD/JPY, using Dukascopy institutional tick
data 2023-01 → 2026-04 (EUR) and 2025 onward (GBP, JPY), in
vectorbtpro v2026.4.7. Consulted vbt.chat three times during the
session (rounds 5, 6, final red-flag review; total cost $0.18).

### Headline numbers (HONEST — post sizing-fix)

| Metric | Round-5 OOS (WFA mean) | Full-sample (3.3y) | Verdict |
|---|---|---|---|
| Top-1 config PF (EUR/USD) | 2.24 | **0.89** | WFA was regime-lucky |
| Top-1 config WR (EUR/USD) | 78% | 74% | WR holds |
| Top-1 config expectancy | +$0.47/trade | **−$0.29/trade** | Full-sample loses |
| Round-5 top-1 on GBP/USD | PF 2.18 | untested | Insufficient data |
| Round-5 top-1 on USD/JPY | PF 0.00 | 0.00 | Family-agnostic fail |

The round-5 top-1 config (`bb_rsi_mr_filtered` M15 BB(40,2.5) RSI(21,20/70)
`session=active`, `weekday=tue_fri`, SL 0.5× ATR / TP 0.75R / chandelier
2× ATR) passed 3-window walk-forward with OOS mean PF 2.24. Running the
same config over the full 3.3-year sample gave PF 0.89 — one of the
three WFA splits happened to be a PF 4.03 regime-slice that pulled the
mean up. Removing that split the honest OOS PF is ~1.34 over the other
two.

**In plain English: there's no reliable edge yet.**

### Two major bugs found (both fixed)

1. **Position-sizing.** vbt defaulted to sizing each trade at
   full-equity × leverage (~$25K notional on a $500 account). Fixed
   by adding `size=5000, size_type='value', leverage=50` in
   `_run_single_backtest` and `capture_trade_records` per vbt.chat
   round-6 pattern. This ~5×ed every prior $-expectancy number
   downward.

2. **Selection-artifact in 3-window WFA.** 3 splits × 50/50 train/test
   is too few to rule out regime cherry-picking. One golden 6-month
   OOS window gave PF 4.03 on a config that is PF 0.89 on the full
   sample. The walk-forward framework is doing exactly what it's
   supposed to (honest OOS measurement per split) — there just aren't
   enough splits for the mean to be reliable with 30K configs tested.

## What's actually true after honest accounting

### Confirmed real findings

1. **Session filter matters more than the indicator choice.** Every
   profitable OOS config uses an explicit session window (7-16 UTC
   `active` or 12-15 UTC `london_ny_overlap`). This survives
   sizing-fix and regime-splitting — consistent across rounds 1-6.

2. **Weekday filter is structural, not noise.** Round 5.5 lone-spike
   test confirmed `tue_fri`'s neighbors (`tue_wed_fri`, `wed_fri`)
   cluster near its performance. The rule is "skip Monday, keep
   Friday", not any specific day combo.

3. **USD/JPY is hostile to our signal families.** Round 4b tested
   MR + momentum families separately; all 53 configs fail
   catastrophically on JPY 2025-2026. Likely session/regime mismatch
   (JPY trades Asian session primarily; BoJ cycle is trending).

4. **EUR/USD and GBP/USD strategy returns are uncorrelated.** Round 8
   shared-cash portfolio showed per-bar return correlation of **0.001**
   between the two pairs' strategy outputs. If and when we do find a
   robust edge, running it on both pairs gives near-additive
   diversification (joint DD fraction 28.7% of bars — not stacking).

5. **SL at 0.5× ATR is too tight.** MAE analysis showed p25 of winning
   trades was worse than the SL distance. Round-6 ablation on
   unfiltered top-5 configs showed PF peak at SL 2.5× ATR — but this
   may not generalize to the filtered M15 config selected for Phase-3.

### Things I thought were true but aren't

1. ~~"M15 bb_rsi_mr_filtered PF 2.07 / $9.29/trade / $1,500/year on
   $500 account."~~ This was the headline from round 5. With sizing
   fixed it's actually −$0.29/trade expectancy on full sample.

2. ~~"Strategy generalizes to GBP/USD with PF 2.04."~~ Round 4 result
   was partly a sizing-bug artifact. Corrected cross-pair gives 5/20
   configs PF > 1.2 on both EUR and GBP, but full-sample validation
   on those hasn't been done.

3. ~~"Ready for Phase 5 paper trading."~~ We are not.

## vbt.chat red-flag review takeaways

Final vbt.chat consultation (cost $0.06, artifact
`ai_queries/20260422T215614-final_pre_paper_review.md`) corroborated
the concerns:

> Your biggest red flags are:
> 1. GBP/USD evidence is too thin to strongly support the 2-pair
>    portfolio thesis.
> 2. USD/JPY total failure is a regime/generalization warning, not
>    just a "different pair" footnote.
> 3. Execution realism is likely the biggest live-degradation source
>    for a high-win-rate M15 scalper.
> 4. 30K tested configs means you should treat headline PF as
>    optimistic unless you do explicit resampling / multiple-testing
>    adjustment.

vbt.chat called out the key tooling that's missing from our pipeline:
- **Bootstrap / Monte Carlo CI on PF** — not done; recommended to
  compute explicitly using vbt trade records
- **Friction stress ladder** — run the strategy with progressively
  worse slippage / spread assumptions and see whether PF survives
- **Regime monitors** — realized-vol percentile, ATR regime,
  autocorrelation, MAE/MFE drift. `vbt.IF().with_apply_func()` is the
  recommended pattern for custom regime indicators.

## What needs to happen before $500 real capital

In rough priority order:

1. **Bootstrap PF confidence interval** on the top config. If 95% CI
   straddles 1.0, kill the candidate. (round 9 work)
2. **More WFA splits.** Move from 3 to 10-12 rolling windows so
   regime cherry-pick probability is much lower. (infrastructure)
3. **Friction stress ladder.** Test PF with 2×, 5×, 10× the current
   spread/slippage assumptions. If PF collapses at 2×, the live
   system won't work.
4. **Backfill GBP + JPY** to 2023-01. Currently GBP has 12 months
   of data; a proper cross-pair validation needs matched length
   on all pairs.
5. **Completely-held-out test period.** Pick a period NEVER used in
   any round (e.g. 2026-Q2 as it fills in). Run the top candidate
   exactly once. If PF < 1.0, kill.
6. **Execution realism audit via NautilusTrader** (Phase 4 per
   CLAUDE.md). L1 FillModel with realistic latency. If Sharpe
   degrades > 30% vs vbt number, kill.
7. **USD/JPY deep dive.** Separate Asian-session backtest. Probably
   needs different signal family (momentum / breakout).

## What's landed on `main` this session

Commits (time-ordered):
```
98a381a  round 4 + 4b: MR survives on GBP, everything fails on JPY
b504069  round 7: per-trade MAE/MFE on round-5 top-10 — SL is too tight
33a352b  round 5.5: weekday-neighbor lone-spike test — NOT a lone spike
d6e846d  round 5 follow-up: vbt.chat integration + round-7 MAE/MFE primitive
56c0ded  explore round 5: weekday + finer session filters lift top PF to 2.07
[round 6 + 8 commit]
[this final rollup commit]
```

Infrastructure added:
- `ROUND_CHECKLIST.md` — per-round compliance checklist
- `src/strategies/filters.py` — adx / session / weekday / spread / vol filters
- `src/backtest/metrics.py` — coverage_years + auto-detect annualization
- `src/backtest/explorer.py` — `capture_trade_records` with MAE/MFE
- `src/strategies/families/filtered_mr.py` — session + weekday preset grids
- `src/strategies/families/__init__.py` — `get_family_by_name`
- `scripts/run_exploration_multi_tf.py` — rounds 3/5
- `scripts/run_round5_5_weekday_neighbors.py` — round 5.5
- `scripts/run_cross_pair_validation.py` — round 4
- `scripts/run_cross_pair_momentum.py` — round 4b
- `scripts/run_round6_sl_ablation.py` — round 6 (`vbt.Param` sweep)
- `scripts/run_round8_cash_sharing_portfolio.py` — round 8 (`cash_sharing`)
- `scripts/capture_top_config_trades.py` — round 7 MAE/MFE
- `src/utils/ai_research.py` — vbt.chat budget-capped wrapper

## Summary for user

We built a rigorous research pipeline (walk-forward, cross-pair,
trade-record diagnostics, vbt.Param sweeps, shared-cash portfolio),
integrated vbt.chat three times for methodology validation, and found
what looked like a real edge. Then, in the final verification pass,
discovered:

(a) a major sizing bug that inflated every prior $-expectancy ~5×, and
(b) walk-forward PF 2.24 shrinks to full-sample PF 0.89 because 3 WFA
    splits is too few when 30K configs were tested.

The honest conclusion is the edge is **not confirmed**. The strategy
**cannot be paper-traded with real money** without the additional
validation steps listed above. The good news: we now have all the
infrastructure to do those steps efficiently, and the methodology
is defensible.

The discovery that the original numbers were inflated, and that we
caught it *before* touching real money, is itself the most valuable
outcome of this session.
