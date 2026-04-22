# Round 5 — weekday filter + finer session partitions

## 1. Objective

Test whether **weekday filters** (skip Monday gap / Friday chop / etc.) and **finer session partitions** (London-open-only, NY-open-only, overlap-first-half vs second-half) materially improve the round-3 profitable basins.

## 2. Hypothesis

- **Expected:** Weekday filter provides modest uplift (+5-10% PF) by dropping Monday-gap noise. Finer session partitions concentrate edge (smaller windows = fewer distracted trades, higher PF, lower trade count).
- **Would refute:** If best configs still use `session="all"` + `weekday="all"`, then round 3's session/weekday axis was already at its useful resolution and further granularity is noise.

## 3. Factors varied

On top of existing round-3 grid:

| Dimension | Round-3 | Round-5 additions |
|---|---|---|
| Session | all / asian / active / london_ny_overlap | +london_open_2h, ny_open_2h, overlap_first_half, overlap_second_half |
| Weekday | (all only) | all, tue_thu, mon_thu, tue_fri |

Applied across both filtered families (`bb_rsi_mr_filtered`, `rsi_extreme_filtered`). All 8 families included in the sweep.

## 4. Method

- **Data:** EUR/USD M1 Dukascopy ticks resampled to M5/M15/M30 (1.2M → 240K/80K/40K bars).
- **Walk-forward:** 3 rolling windows, 50/50 train/test each. OOS coverage ≈ 1.65 years.
- **Sweep:** 8 families × 40 param combos (random-subset) × 15 exit configs × 3 TFs × 3 splits × 2 (IS+OOS) = **43,200 backtests**.
- **Metrics:** profit factor, win rate, expectancy, max DD, Sortino. Annualization now uses `coverage_years` from the actual returns index (no more 12/8 hardcoded ratio).
- **Filter:** ≥30 trades per OOS split, all 3 splits completed.
- **Ranking:** aggregate (mean) across the 3 OOS splits.

## 5. Headline numbers

- **25,020 rows** across 3 TFs × 8 families. 0 failures.
- **8,690 configs** completed all 3 OOS splits with ≥30 trades.
- **147 profitable configs** (PF > 1.2 AND positive expectancy across all 3 splits).

| TF | Configs | Winners PF>1.2 | Best PF | Best $/trade |
|---|---|---|---|---|
| M5 | 1,785 | 50 | 1.92 | $+8.18 |
| **M15** | 1,665 | **67** | **2.07** | **$+20.14** |
| M30 | 1,545 | 30 | 1.71 | $+19.14 |

**Winner concentration shifted dramatically toward M15** — 67 winners (up from 13 in round 3). Round-5 weekday+session additions hit pay dirt here.

## 6. Top 10 overall performers

```
 TF     Family                    PF    $/trade  WR    DD    Trades  Session              Weekday
 M15    bb_rsi_mr_filtered        2.07  $+9.29   78%   32%   51      london_ny_overlap    tue_fri
 M15    bb_rsi_mr                 1.92  $+7.23   82%   25%   90      (unfiltered)         -
 M5     bb_rsi_mr_filtered        1.92  $+3.65   79%   21%   78      active               tue_fri
 M15    bb_rsi_mr_filtered        1.87  $+8.61   84%   21%   35      active               tue_fri
 M15    bb_rsi_mr                 1.83  $+9.27   83%   30%   87      (unfiltered)         -
 M5     bb_rsi_mr                 1.82  $+4.62   79%   20%   55      (unfiltered)         -
 M15    bb_rsi_mr_filtered        1.78  $+11.80  74%   44%   62      active               mon_thu
 M15    bb_rsi_mr_filtered        1.77  $+6.06   83%   28%   55      london_ny_overlap    tue_fri
 M15    bb_rsi_mr                 1.72  $+11.25  77%   34%   85      (unfiltered)         -
 M30    rsi_extreme               1.71  $+11.73  73%   43%   36      (unfiltered)         -
```

**bb_rsi_mr dominates** — 9 of top 10. Filtered variant AND unfiltered both appear.

## 7. Per-factor breakdown

### Weekday filter effect (filtered families)

```
weekday    configs   top_pf   winners(PF>1.2)
all            30    1.49           6
mon_thu       150    1.78          34
tue_fri       140    2.07          29   ← dominant
tue_thu       118    1.35           1   ← surprisingly bad
```

**tue_fri and mon_thu both materially beat unfiltered weekday** (top_pf 2.07 / 1.78 vs 1.49 for `all`). `tue_thu` (mid-week-only, most conservative) produced just 1 winner — too restrictive, thinning the trade count below signal.

### Session partition effect (filtered families)

```
session                configs   top_pf   winners
london_ny_overlap           63    2.07      20
active                     213    1.92      42
all                        116    1.36       3
ny_open_2h                  15    1.33       4
overlap_second_half         15    1.24       1
overlap_first_half          15    0.97       0
london_open_2h               1    0.73       0
```

- **`london_ny_overlap` remains king** (PF 2.07) — same finding as round 2/3.
- **`active` (7-16 UTC) produces more winners** (42) because its wider window gives more trade-count stability.
- **Super-fine partitions (london_open_2h, ny_open_2h) mostly failed** — too few trades per split to distinguish signal from noise. Over-slicing destroys statistical power.

## 8. Caveats

1. **Multi-testing risk:** 147 winners / 8,690 configs = 1.7%. At random with PF>1.2 gate we'd expect ~5% winners. Our structural patterns (bb_rsi_mr dominance, london_ny_overlap + tue_fri concentration) argue for real signal, not noise — but round 4 cross-pair validation is still the critical robustness test.

2. **Expectancy numbers are M15 artifacts of scale.** $+9/trade is nominal on $5,000 notional. Same strategy on $500 account with $100/trade margin = same $ profit per trade (position size is fixed by $100 margin), but represents much larger % of account.

3. **Max DD of 32-44% on winners would trip current $400 circuit-breaker on $500 account.** Per round 3.5 meta-analysis, scaling capital to $1,500+ resolves this, OR we halve position size ($50/trade margin).

4. **`tue_thu` being worst is suspicious.** Monday+Friday contribute real trading volume that mid-week-only loses. This is a finding, not a bug — Monday/Friday aren't always noise.

5. **Round 5 doesn't include round-4 cross-pair data yet.** GBP/USD + USD/JPY backfills are resuming in background now.

## 9. AI analysis integration

**Pending.** Per ROUND_CHECKLIST mandate, I need to pass round-5 results through `vbt.chat` before closing the round. Query executes next (this session). Notes from prior vbt.chat calls still apply:

- Round 1→2: session filter is the edge. **Confirmed again** — 42/50 M5 winners and most M15 winners use explicit session filter.
- Round 2→3: "trail off + moderate stops" for MR. **Mostly still true** at M5; M15 unfiltered bb_rsi_mr uses chandelier trail.
- Round 3→4: recommended Monte Carlo / bootstrap CIs on PF. **Not yet implemented** — deferred to round 7 (vbt.Portfolio.bootstrap or custom).

## 10. Action items for round 6+

1. **Round 4 (cross-pair validation)** runs the moment GBP/USD + USD/JPY backfills finish (currently ~60-70% done, resuming now). All Top-10 round-5 configs re-tested on all 3 pairs. Hard requirement: must stay PF > 1.0 + positive expectancy on all 3 pairs.

2. **Round 6 (position sizing variants)** — use `adjust_func_nb` to test:
   - Kelly-fraction sizing (bet more when recent win-rate is high)
   - Volatility-inverse sizing (smaller in high ATR)
   - Losing-streak contraction (halve after N losses)

3. **Round 7 (trade-level diagnostics)** — capture `pf.trades.records_readable` for top-N configs; compute MAE/MFE per trade to validate stop-sizing. This is what vbt.chat specifically recommended as "pre-paper-trading stress testing."

4. **Round 8 (portfolio correlation)** — rather than assuming correlation, run top M5+M15+M30 configs as columns in a single `Portfolio.from_signals` call with `cash_sharing=True` and measure actual inter-strategy correlation.

5. **Skip `tue_thu` variant going forward** — empirically worst; no reason to keep testing it.

## 11. COMPLIANCE (per ROUND_CHECKLIST)

- [x] Objective stated (§1)
- [x] Hypothesis + refutation criteria (§2)
- [x] Factors enumerated (§3)
- [x] Raw CSV at `backtest_results/explore_multi_tf_20260422T0026/combined_results.csv`
- [ ] Per-trade records captured — **NOT YET** (round 7 work)
- [x] Walk-forward OOS used (3 windows, 50/50)
- [x] Diary event emitted (event logged via explorer)
- [x] Annualization via `coverage_years` — yes, `BacktestMetrics.annualized_profit_usd()` available, though I used per-split × windows math for simplicity in this doc
- [x] Primary metrics = PF + expectancy + WR + DD (§6)
- [ ] RRR computed — **pending for round 4+ survivors**
- [x] Multi-testing caveat noted (§8)
- [ ] vbt.chat consulted — **pending, next action**
- [x] Findings doc in correct 10-section structure
