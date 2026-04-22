# Round 9 — validation gauntlet

## 1. Objective

Apply three concurrent stress tests to the 5 configs that passed round-4
cross-pair validation (PF > 1.2 on both EUR/USD and GBP/USD with
corrected sizing). A config must pass ALL THREE gauntlets to be
eligible for Phase-5 paper trading:

  1. **Full-sample truth.** Plain single-run backtest on 3.3y EUR/USD
     + 1y GBP/USD. `size=5000, size_type='value', leverage=50`. PF
     must exceed 1.0 on both pairs.
  2. **Bootstrap PF 95% CI.** 1,000 resamples-with-replacement of the
     trade PnL series. Lower bound of 95% CI must exceed 1.0 on both
     pairs.
  3. **Friction stress at 2× slippage.** Re-run at 2× the estimated
     half-spread. PF must exceed 1.0 on both pairs.

Rationale: these are the checks vbt.chat's final red-flag review
(`ai_queries/20260422T215614-final_pre_paper_review.md`) flagged as
missing from rounds 1-8.

## 2. Results

```
config_rank family              tf     eur_fs  gbp_fs  eur_bs_lo  gbp_bs_lo  eur_fr2  gbp_fr2  FS  BS  FR  ALL
0           bb_rsi_mr_filtered  15min   1.07    1.19    0.84       0.74       1.00     1.03   ✓   ✗   ✗   ✗
1           bb_rsi_mr_filtered  15min   1.06    1.10    0.84       0.70       1.01     0.96   ✓   ✗   ✗   ✗
2           bb_rsi_mr_filtered  15min   0.89    0.88    0.65       0.51       0.83     0.73   ✗   ✗   ✗   ✗
3           bb_rsi_mr_filtered  1min    0.98    1.58    0.79       0.77       0.70     0.84   ✗   ✗   ✗   ✗
4           bb_rsi_mr_filtered  1min    0.98    1.91    0.81       0.95       0.72     1.13   ✗   ✗   ✗   ✗
```

**0 of 5 configs pass all three gauntlets.**

### Why each gauntlet mattered

- **Full-sample** caught configs 2, 3, 4 (PF 0.88-0.98 on EUR) that
  looked good in the 3-window WFA aggregate but fall apart on the
  full range.
- **Bootstrap 95% CI** caught configs 0 and 1 (full-sample PF >1 but
  their observed trade-PnL distribution is noisy enough that a
  resample could easily give PF < 1.0). The lower bound tells us the
  "worst-case-within-sampling-noise" PF; 0.84 EUR / 0.74 GBP is not
  a gate-clearing number.
- **Friction 2×** caught every config. At 2× our current half-spread
  slippage assumption, PF drops to ~1.0 at best. Live OANDA
  execution can easily be 2× worse than Dukascopy tick spreads
  (slippage, requotes, news spikes).

## 3. Interpretation

The best config (rank 0: bb_rsi_mr_filtered M15, BB(40, 2.0) RSI(21,
35/75), session=london_ny_overlap, weekday=tue_fri, SL 0.5× ATR /
TP 0.75R / chandelier 2× ATR) is the only one with full-sample PF > 1.0
on both pairs. That's not nothing — it's a weakly positive edge. But:

- PF 1.07 EUR ± CI to 0.84 means a plausible true PF anywhere from
  "not profitable after fees" to "modestly profitable"
- PF 1.00 at 2× friction means a 10-20 bps worsening in spread wipes
  the edge
- 12 months of GBP data is a fundamentally thin measurement

**Verdict:** There's SOMETHING here (full-sample WR 74%, direction is
right) but the edge is below the "statistically significant after
multi-testing + friction stress" bar.

## 4. What's needed to unlock paper trading

In rough priority order:

1. **Backfill GBP/USD to 2023-01.** Resume Dukascopy ingestion.
   With 3+ years of GBP data the bootstrap CI tightens substantially.
2. **Move WFA to 10-12 rolling windows.** 3 splits × 30K configs
   is a statistical power problem; 12 splits approximately quadruples
   the separation between signal and noise. Implement via
   `vbt.cv_split` per round-5 vbt.chat pattern.
3. **Re-run ALL survivors of #2 through the gauntlet.** Use stricter
   gates: full-sample PF > 1.1, bootstrap lo95 > 1.05, friction 2× PF > 1.05.
4. **If still zero survivors** — the signal family (BB+RSI MR) is the
   problem, not the validation. Explore alternative families
   (Keltner breakout, stat-arb pairs, volatility-adjusted RSI, etc.).
5. **Held-out period** — once a candidate passes the above, take the
   most recent 3 months of EUR/USD (data NEVER seen by any round) and
   run the candidate exactly once. PF < 1.0 = kill.
6. **NautilusTrader L1 FillModel validation** (Phase-4 in CLAUDE.md)
   before paper trading.

## 5. Is the session a success or a failure?

The edge we originally reported (round-5 "PF 2.07, $9/trade,
$1500/year") turned out to be an artifact of two bugs:

  - Full-equity-leverage sizing inflated $-expectancy ~5×
  - 3-window WFA was regime-cherry-picking with 30K configs tested

Both found, both documented, both fixed. Infrastructure for rigorous
re-validation (sizing, `vbt.Param` sweeps, bootstrap CI, friction
ladder, `cash_sharing` portfolio) is all in place.

**This is exactly what Phase 2 is supposed to produce — either a
validated edge or a defensible "no".** We have the defensible "no",
with a clear gate list for advancing.

## 6. COMPLIANCE

- [x] Objective (§1)
- [x] Method: three-gauntlet design (§1)
- [x] Raw CSVs at `backtest_results/round9_gauntlet_*`
- [x] Full-sample validation applied
- [x] Bootstrap PF CI (1,000 resamples, 95% 2-sided)
- [x] Friction stress ladder (1×, 2×, 5×, 10×)
- [x] Clear pass/fail gates with "0 of 5" survivor count (§2)
- [x] Interpretation + "what's needed to unlock paper" path (§3-4)
- [ ] vbt.chat consulted — covered by final red-flag review in round-8
      closeout (artifact `20260422T215614-...`)
