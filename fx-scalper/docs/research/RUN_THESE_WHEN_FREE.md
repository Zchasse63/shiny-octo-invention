# Ready-to-run experiments (after machine frees up)

**Status as of 2026-04-22 evening:** hit a hard system resource wall —
Claude Code + Claude Helper renderer were using 75% CPU / 7 GB RAM, and
the G10 backfills (each holding 3+ GB in compressed memory before final
Parquet write) tipped the machine into swap. Every heavy Python process
now takes 9+ minutes just to import pandas/numpy/vbt because of disk
I/O contention. The experiments are all coded correctly — they just
can't execute cleanly right now.

## Run order

All scripts below are already on `main` and work correctly (validated
via smoke tests or earlier full runs). Run them in order when the
machine has memory + CPU headroom (overnight, or when Claude Code is
closed).

### 1. Backfill the missing G10 pairs (run ONE at a time)

The backfill script buffers all ticks in memory before writing
Parquet at the end, so running 2+ concurrently will blow out RAM.
Recommend running each pair serially:

```bash
cd ~/Desktop/Forex/fx-scalper
for pair in AUD_USD NZD_USD USD_CHF USD_CAD EUR_GBP EUR_JPY; do
  echo "=== Pulling $pair ==="
  .venv/bin/python scripts/pull_dukascopy.py \
    --start 2023-01-01 --end 2026-04-20 --instruments "$pair" \
    2>&1 | tee -a "logs/dukascopy_$(echo $pair | tr '[:upper:]' '[:lower:]').log"
done
```

Estimated wall time: ~20-25 min per pair × 6 pairs = ~2-2.5 hours total.
Run overnight.

### 2. Round 14 — meta-labeling with sklearn HGB

Tests whether a machine-learning secondary classifier can lift the
marginal primary BB+RSI signal past the retail-cost hurdle.

```bash
.venv/bin/python scripts/run_round14_meta_labeling.py \
  2>&1 | tee logs/round14.log
```

Expected runtime: 1-2 minutes on a quiet machine. Produces:
- `backtest_results/round14_meta_<ts>/cv_metrics.csv` — per-fold AUC
- `backtest_results/round14_meta_<ts>/threshold_sweep.csv` — PF vs
  probability-threshold table
- `backtest_results/round14_meta_<ts>/labels_with_meta.csv` — full
  labeled dataset with classifier probabilities

**Decision criterion:** does the classifier-filtered PF beat the
primary's unfiltered PF by ≥ 0.2? If yes, meta-labeling is useful.

### 3. Round 16 — stat-arb Kalman EUR/GBP

```bash
.venv/bin/python scripts/run_round16_stat_arb.py \
  2>&1 | tee logs/round16.log
```

Expected runtime: 30-60 seconds. Output:
- `backtest_results/round16_statarb_<ts>/summary.csv`

**Expected result:** Koronidis (2013) found EUR/USD vs GBP/USD do NOT
cointegrate at 5% on daily data, so we likely get 0 trades or very
few. That's informative.

### 4. Round 12/13 — full G10 basket trend-following

Prerequisite: step 1 (backfills) must be done first. Runs one MA/
Donchian rule across all 9 G10 pairs simultaneously via
`Portfolio.from_signals(..., cash_sharing=True, group_by=True)`.

```bash
.venv/bin/python scripts/run_round12_13_basket.py \
  2>&1 | tee logs/round12_13.log
```

Expected runtime: 1-2 minutes. Output:
- `backtest_results/round12_13_basket_<ts>/basket_summary.csv`
- `backtest_results/round12_13_basket_<ts>/per_pair_detail.csv`

**This is the highest-EV open experiment** per our research review.
The edge for retail FX algos historically comes from diversified
basket trend-following, not single-pair scalping. If any
(family, timeframe, params) combo produces combined PF > 1.2 on
the G10 basket with BCa lower bound > 1.0, that's a genuine
candidate for Phase-5 paper trading.

### 5. If #4 shows promise — Round 17 NautilusTrader validation

Not yet scripted. When we have a candidate, port it to
NautilusTrader with `FillModel(prob_fill_on_limit=0.95,
prob_slippage=0.5)`. If the Sharpe doesn't degrade more than 30%
vs the vbt number, proceed to Phase 5 (OANDA practice paper
trade).

## What you'll get after running these

- Clean pass/fail on the meta-labeling hypothesis (Round 14)
- Clean pass/fail on the stat-arb hypothesis (Round 16)
- The G10 basket trend-following test (Round 12/13) — the actual
  high-confidence answer on whether diversified trend-following
  works at retail costs
- If Round 12/13 passes, a concrete ready-to-paper-trade config

## What's already confirmed

- **The M15 BB+RSI MR edge doesn't clear retail costs** (rounds 1-9 + 11)
- **Position-sizing bug fixed** (rounds 6+ all use correct
  `size=5000, size_type='value'` = $100 margin per trade)
- **Statistical rigor in place**: purged k-fold CV, BCa bootstrap,
  Deflated Sharpe Ratio (Round 11)
- **Peer-reviewed fixing-reversal** validated pre-cost but killed at
  retail spreads (Round 15 — matches Krohn 2024 paper exactly)
- **Trend following families** (Donchian, MA crossover) in the
  registry, ready for basket testing

## What's uncertain / still to test

- Whether a diversified G10 trend-following basket clears retail
  costs — **Round 12/13 answers this, and it's the biggest
  single experiment left**
- Whether meta-labeling on the M15 primary lifts PF past 1.2 with
  statistical significance — Round 14
- Whether stat-arb on FX produces any tradeable signal — Round 16
- Whether NautilusTrader-simulated execution realism preserves the
  edge — Round 17

Everything above is bounded: the next ~4 hours of compute time
(backfills + 3 experiments + Nautilus port) either confirms or kills
each remaining hypothesis. After that we either have a
paper-trade-ready system or we know what doesn't work well enough to
stop trying retail FX algos in general.
