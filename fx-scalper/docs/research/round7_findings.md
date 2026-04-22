# Round 7 — per-trade MAE/MFE diagnostics on round-5 top-10 configs

## 1. Objective

Capture `pf.trades.records_readable` for round-5's top-10 OOS configs,
compute per-trade MAE / MFE (maximum adverse / favorable excursion), and
use the distributions to verify whether our stop-loss and take-profit
sizing is calibrated to the actual behaviour of winning signals. This is
the round-7 deliverable mandated by `ROUND_CHECKLIST` ("per-trade records
captured for top configs") and the highest-priority action item from the
round-5 vbt.chat artifact.

## 2. Hypothesis

- **If SL is well-sized:** the MAE distribution's p10 (worst 10%) should
  cluster near the initial stop distance. If MAE p50 is already near the
  stop, the stop is too tight — we're killing winners.
- **If TP / trail is well-sized:** MFE p90 should be comfortably above
  the fixed TP level; trail should be catching the upside beyond TP.

## 3. Method

- **Data:** EUR/USD M1 Dukascopy 2023-01 → 2026-04 (3.3 years, full
  sample — NOT walk-forward. This round is diagnostic, not validation).
- **Configs:** top-10 OOS winners from round-5's combined_results.csv,
  ranked by mean PF across 3 walk-forward splits with min 30 trades per
  split and PF ≥ 1.2 on all splits. Re-run on full-range bars with their
  matching timeframe (M5 / M15 / M30).
- **MAE / MFE formula** (per trade): walk every bar from entry to exit,
  take the worst (long: lowest mid-low; short: highest mid-high) to get
  the adverse excursion; take the best for favorable excursion; report as
  fraction of entry price.
- **Outputs:** `backtest_results/trade_records_20260422T1604/` — one
  parquet per config plus `top10_summary.csv`.

## 4. Headline numbers

### 4.1 Top-1 (M15 `bb_rsi_mr` unfiltered, 541 trades full-sample)

| Percentile | MAE | MFE |
|---|---|---|
| p01 (worst/best 1%) | −0.433% | +0.025% |
| p10 | −0.199% | +0.047% |
| p25 | −0.101% | +0.071% |
| p50 | −0.047% | +0.113% |
| p75 | −0.024% | +0.181% |
| p90 | −0.010% | +0.311% |
| p95 | — | — |

Fraction of trades with MAE worse than −0.10%: **25.3%**
Fraction of trades with MAE worse than −0.15%: **16.3%**

### 4.2 Top-10 rollup (from `top10_summary.csv`)

All configs show consistent MAE-median in the range **−0.04% to −0.07%**
(M5-M30), with MFE-p90 in **+0.22% to +0.32%**. Full-sample win rates
track OOS means within a few percentage points (68-78% vs 73-85% OOS).

## 5. Interpretation — the big finding

Our current exit framework uses **SL 0.5× ATR initial stop**. ATR-14 on
M15 EUR/USD typically runs ~0.12-0.18% of price (depending on regime),
so 0.5× ATR ≈ **−0.06% to −0.09%** stop distance. That places the stop
**between the MAE p50 (−0.05%) and p25 (−0.10%)** — meaning ~35-40% of
trades touch the stop even when the signal ultimately would have worked.

**The stop is too tight.** Widening SL to 1.0× ATR (~−0.12% to −0.18%)
would push the stop past the MAE p10 (−0.20%) and dramatically reduce
stop-outs on eventual winners. Cost: larger per-trade loss when the
stop actually triggers. Benefit: more of the 25-40% currently-stopped
trades get to run.

**TP / trail looks reasonably sized.** TP at 1.5R ≈ 1.5 × 0.5 × ATR ≈
0.09-0.14%. That captures trades between MFE p50 (0.11%) and p75 (0.18%).
The chandelier 2× ATR trail picks up the upper tail (MFE p90 0.31%).
This looks right — don't change TP/trail without re-running.

## 6. Action items for round 6

1. **Add SL-width ablation** as a round-6 axis: test {0.5×, 0.75×, 1.0×,
   1.5×} ATR initial stops across the top-5 round-5 configs. Hypothesis:
   PF peak shifts to 0.75-1.0× ATR; expectancy-per-trade rises; max DD
   rises modestly (larger losers) but more-than-offset by fewer
   stop-outs.

2. **Use MAE percentiles as the stop-sizing heuristic going forward**:
   set SL at the MAE p15 of a prior-window simulation so ~15% of trades
   touch stop. This is an empirically-grounded alternative to ATR
   multipliers.

3. **Round-6 position sizing** should be run AFTER re-optimizing SL width,
   not before — the dynamic sizing interactions with a too-tight stop
   would be misleading.

## 7. Caveats

1. **Full-sample MAE, not walk-forward.** The percentiles include the
   2024-2026 regime; if regime shifts, MAE distribution shifts too.
   Round 6 should use walk-forward MAE percentiles from the train split.
2. **Entry-price basis.** MAE is measured from `Avg Entry Price` using
   mid-high / mid-low. Actual fills on OANDA may include slippage that
   our Dukascopy backtest doesn't see — paper trading is the honest test.
3. **Top-configs bias.** MAE distribution of TOP-10 configs may be
   more favorable than the median config. A broader MAE study across
   the profitable-but-not-top tier would be more honest for the general
   stop heuristic.

## 8. COMPLIANCE

- [x] Objective stated (§1)
- [x] Hypothesis (§2)
- [x] Method / data enumerated (§3)
- [x] Per-trade records captured — ✓ this is the checklist item itself.
      Parquet files at `backtest_results/trade_records_20260422T1604/`
- [x] Primary metrics + MAE/MFE percentiles (§4)
- [x] Action items tied to round 6 (§6)
- [x] Caveats (§7)
- [ ] vbt.chat consulted — N/A for this diagnostic; the approach itself
      came from the round-5 vbt.chat artifact
