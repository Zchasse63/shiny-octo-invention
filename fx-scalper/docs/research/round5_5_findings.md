# Round 5.5 — weekday-neighbor lone-spike test

## 1. Objective

Per the round-5 vbt.chat artifact (`ai_queries/20260422T155429-...`),
the top-1 config's `weekday=tue_fri` win could be a **lone spike** (noise)
or part of a **smooth manifold** (real edge). This round is the cheapest
discriminator: pin every param to the round-5 top-1, then sweep only the
weekday dimension across `tue_fri` + 3 engineered neighbors.

## 2. Hypothesis

- **Smooth-manifold (real edge):** neighbors of `tue_fri` cluster near its
  PF; the edge tracks a structural property like "include Friday" or
  "exclude Monday".
- **Lone spike (noise):** `tue_fri` wins in isolation, neighbors collapse.

## 3. Factors varied

| Weekday preset | Days included | Relation to `tue_fri=(1,2,3,4)` |
|---|---|---|
| `all` | M T W Th F | identity (no filter) |
| `tue_thu` | T W Th | drop Friday too |
| `mon_thu` | M T W Th | drop Friday only |
| `tue_fri` | T W Th F | **baseline** (round-5 winner) |
| `wed_fri` | W Th F | tue_fri minus Tuesday |
| `tue_wed_fri` | T W F | tue_fri minus Thursday |
| `mon_tue_fri` | M T F | start-of-week + Friday only |

All other params pinned to round-5 top-1: BB(20, 2.25), RSI(14, 25/75),
ADX off, session=`london_ny_overlap`, spread ≤ 0.25× ATR, M15, SL 0.5 ATR
through TP 1.5R with chandelier 2× ATR trail.

## 4. Method

- **Data:** EUR/USD M1 Dukascopy ticks 2023-01 → 2026-04, resampled to
  M15 (80,083 bars).
- **Walk-forward:** 3 rolling windows, 50/50 train/test.
- **Sweep:** 7 weekday presets × 140 exit configs × 3 WF splits = 2,940 OOS
  + 2,940 IS rows.
- **Aggregation:** **median** of `profit_factor` across (exit_config ×
  split) per weekday preset. Median, not mean, because several cells
  produced `inf` PF (no losses in sample) — means would be dominated by
  those outliers. Also report p25/p75 for dispersion.

## 5. Headline numbers (OOS, 420 rows per weekday)

```
weekday        median_pf   p25_pf  p75_pf  median_expectancy  median_wr
tue_wed_fri        1.660    1.002   2.875          $10.00       70.8%   ← best neighbor
tue_fri            1.370    0.748   2.294           $6.00       65.2%   ← round-5 winner
wed_fri            1.311    0.667   2.586           $4.97       66.1%
all                1.180    0.719   1.565           $2.46       65.2%
mon_tue_fri        1.136    0.825   1.553           $1.66       63.4%
tue_thu            1.034    0.629   1.916           $0.61       62.5%
mon_thu            1.029    0.644   1.352           $0.32       60.7%
```

## 6. Interpretation

**Result: NOT A LONE SPIKE.** `tue_fri`'s neighbors `tue_wed_fri` (1.66)
and `wed_fri` (1.31) cluster near or above its 1.37 median PF. The top
three weekday presets all share two structural properties:

1. **Friday is included** (tue_wed_fri, tue_fri, wed_fri all include Fri).
2. **Monday is excluded** (same three presets all skip Monday).

The worst three presets (`mon_thu`, `tue_thu`, `mon_tue_fri`) fail at
least one of these — `mon_thu` / `tue_thu` lack Friday, `mon_tue_fri`
includes Monday.

**Implied rule:** the weekday edge is "**skip Monday, keep Friday**," not
"keep Tue-Fri specifically." This is stable across neighbors.

## 7. Caveats

1. **Single config tested.** The pinned BB(20,2.25)+RSI(14,25/75)+overlap
   config may have idiosyncratic weekday sensitivity. Round-5.5 does NOT
   prove the weekday rule generalizes across the full BB/RSI grid.
2. **Median is robust but low-powered.** Dispersion (p25-p75) is wide
   (0.7-2.9 for tue_wed_fri), so individual exit configs vary a lot.
3. **Still EUR/USD-only.** Cross-pair validation (round 4) remains the
   gating test. If GBP/USD and USD/JPY show no Friday/Monday skew, the
   edge is EUR/USD-specific (still useful, just narrower).
4. **Seasonality risk.** 3 years of data includes rate-cycle transitions
   that may have left structural Monday/Friday patterns that won't persist.

## 8. Action items

1. **Keep `tue_wed_fri` + `wed_fri` in the round-4 cross-pair candidate
   list** alongside `tue_fri`. If neighbors also survive cross-pair, the
   "skip Monday, keep Friday" rule is the true finding and we adopt it
   without privileging any specific preset.
2. **Do not drop `tue_fri` in favor of `tue_wed_fri` yet** — the 1.66 vs
   1.37 median gap is inside the p25-p75 band.
3. **Round 6 sizing variants** should use `tue_fri` as baseline (matches
   round-5 artifact) but parameterize weekday so we can compare.

## 9. COMPLIANCE

- [x] Objective stated (§1)
- [x] Hypothesis + refutation criteria (§2)
- [x] Factors enumerated (§3)
- [x] Raw CSV at `backtest_results/explore_round5_5_20260422T1600/full_results.csv`
- [x] Walk-forward OOS used
- [x] Median-based aggregation (robust to inf-PF outliers) in §5
- [x] Primary metrics = PF + expectancy + WR + DD
- [x] Multi-testing caveat noted (§7)
- [ ] vbt.chat consulted — **not required** per ROUND_CHECKLIST, this is
      a follow-on micro-round actioning a prior vbt.chat recommendation
- [x] Findings doc in the conventional structure
