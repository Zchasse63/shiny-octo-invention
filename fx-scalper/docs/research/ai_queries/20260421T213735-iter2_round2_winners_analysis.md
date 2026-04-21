# iter2_round2_winners_analysis

**When:** 2026-04-21T21:37:35+00:00 UTC
**Provider:** anthropic / `anthropic/default`
**Tokens:** input=3256 output=5074
**Estimated cost:** $0.0859

## Question

I just ran an exploratory backtest sweep using vectorbt Pro on EUR/USD M1 bars,
2023-01-01 to 2026-04-20, with a common exit framework (SL at N×ATR, optional
fixed TP at N×R, optional ATR/chandelier trail) across six signal families.

**Goal profile:** high-frequency scalping (5+ trades/day), base-hit win rate
plus occasional trailing-stop runners for asymmetric upside. Small account
($500) with 50:1 leverage, $100 margin per trade ($5,000 notional).

**Top 10 configs by profit factor (OOS, walk-forward):**
```
| family               | family_params                                                                                                                                                              | exit_config                                                                                                  |   profit_factor |   win_rate |   sharpe |   sortino |   total_trades |   max_drawdown_pct |   expectancy_usd |
|:---------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------|----------------:|-----------:|---------:|----------:|---------------:|-------------------:|-----------------:|
| rsi_extreme_filtered | {"max_adx": null, "overbought": 75, "oversold": 25, "rsi_length": 21, "session": "london_ny_overlap"}                                                                      | {"atr_length": 14, "sl_atr_mult": 2.0, "tp_r_mult": null, "trail_atr_mult": 3.0, "trail_kind": "atr_trail"}  |          2.0100 |     0.7669 |   1.7319 |    0.1312 |        95.3333 |             0.1831 |           0.8426 |
| rsi_extreme_filtered | {"max_adx": null, "overbought": 80, "oversold": 20, "rsi_length": 14, "session": "london_ny_overlap"}                                                                      | {"atr_length": 14, "sl_atr_mult": 2.0, "tp_r_mult": null, "trail_atr_mult": 3.0, "trail_kind": "atr_trail"}  |          1.6705 |     0.7534 |   1.5129 |    0.1061 |       104.0000 |             0.1864 |           0.7874 |
| rsi_extreme_filtered | {"max_adx": null, "overbought": 80, "oversold": 20, "rsi_length": 14, "session": "london_ny_overlap"}                                                                      | {"atr_length": 14, "sl_atr_mult": 0.5, "tp_r_mult": 1.0, "trail_atr_mult": 3.0, "trail_kind": "atr_trail"}   |          1.6650 |     0.8353 |   1.9023 |    0.1508 |       102.0000 |             0.2200 |           1.4955 |
| rsi_extreme_filtered | {"max_adx": null, "overbought": 75, "oversold": 25, "rsi_length": 21, "session": "london_ny_overlap"}                                                                      | {"atr_length": 14, "sl_atr_mult": 0.5, "tp_r_mult": 0.75, "trail_atr_mult": 3.0, "trail_kind": "atr_trail"}  |          1.6164 |     0.8738 |   1.8252 |    0.1262 |        91.3333 |             0.1568 |           1.2167 |
| rsi_extreme_filtered | {"max_adx": null, "overbought": 80, "oversold": 20, "rsi_length": 14, "session": "london_ny_overlap"}                                                                      | {"atr_length": 14, "sl_atr_mult": 0.5, "tp_r_mult": 0.75, "trail_atr_mult": 3.0, "trail_kind": "atr_trail"}  |          1.6012 |     0.8513 |   1.6453 |    0.1196 |       102.0000 |             0.2312 |           1.0610 |
| rsi_extreme_filtered | {"max_adx": null, "overbought": 80, "oversold": 20, "rsi_length": 14, "session": "london_ny_overlap"}                                                                      | {"atr_length": 14, "sl_atr_mult": 1.0, "tp_r_mult": 1.0, "trail_atr_mult": 2.0, "trail_kind": "chandelier"}  |          1.4367 |     0.7029 |   2.4431 |    0.1902 |       106.3333 |             0.2568 |           2.0717 |
| bb_rsi_mr_filtered   | {"bb_length": 20, "bb_std": 2.25, "max_adx": null, "max_spread_atr_frac": 0.25, "rsi_length": 14, "rsi_long_threshold": 25, "rsi_short_threshold": 75, "session": "asian"} | {"atr_length": 14, "sl_atr_mult": 0.5, "tp_r_mult": 0.75, "trail_atr_mult": 3.0, "trail_kind": "atr_trail"}  |          1.3908 |     0.8299 |   1.1595 |    0.0867 |       146.0000 |             0.1605 |           0.4190 |
| rsi_extreme_filtered | {"max_adx": null, "overbought": 80, "oversold": 20, "rsi_length": 14, "session": "london_ny_overlap"}                                                                      | {"atr_length": 14, "sl_atr_mult": 1.5, "tp_r_mult": 0.75, "trail_atr_mult": 3.0, "trail_kind": "atr_trail"}  |          1.3742 |     0.7428 |   1.9240 |    0.1902 |       102.0000 |             0.2359 |           1.9598 |
| rsi_extreme_filtered | {"max_adx": null, "overbought": 75, "oversold": 30, "rsi_length": 30, "session": "london_ny_overlap"}                                                                      | {"atr_length": 14, "sl_atr_mult": 0.5, "tp_r_mult": 0.75, "trail_atr_mult": 2.0, "trail_kind": "chandelier"} |          1.3551 |     0.7854 |   0.6743 |    0.0450 |        78.3333 |             0.2526 |           0.1406 |
| bb_rsi_mr_filtered   | {"bb_length": 20, "bb_std": 2.25, "max_adx": null, "max_spread_atr_frac": 0.25, "rsi_length": 14, "rsi_long_threshold": 25, "rsi_short_threshold": 75, "session": "asian"} | {"atr_length": 14, "sl_atr_mult": 0.5, "tp_r_mult": 1.0, "trail_atr_mult": 3.0, "trail_kind": "atr_trail"}   |          1.3461 |     0.8209 |   1.2840 |    0.1019 |       144.6667 |             0.1699 |           0.4711 |
```

**Top 10 by win rate (OOS):**
```
| family               | family_params                                                                                         | exit_config                                                                                                  |   profit_factor |   win_rate |   sharpe |   sortino |   total_trades |   max_drawdown_pct |   expectancy_usd |
|:---------------------|:------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------|----------------:|-----------:|---------:|----------:|---------------:|-------------------:|-----------------:|
| rsi_extreme_filtered | {"max_adx": null, "overbought": 75, "oversold": 25, "rsi_length": 21, "session": "london_ny_overlap"} | {"atr_length": 14, "sl_atr_mult": 0.5, "tp_r_mult": 0.75, "trail_atr_mult": 3.0, "trail_kind": "atr_trail"}  |          1.6164 |     0.8738 |   1.8252 |    0.1262 |        91.3333 |             0.1568 |           1.2167 |
| rsi_extreme_filtered | {"max_adx": null, "overbought": 80, "oversold": 20, "rsi_length": 21, "session": "active"}            | {"atr_length": 14, "sl_atr_mult": 0.5, "tp_r_mult": 0.75, "trail_atr_mult": 3.0, "trail_kind": "atr_trail"}  |          1.2897 |     0.8734 |   1.0954 |    0.0523 |        51.5000 |             0.2036 |           1.5306 |
| rsi_extreme_filtered | {"max_adx": null, "overbought": 80, "oversold": 20, "rsi_length": 21, "session": "active"}            | {"atr_length": 14, "sl_atr_mult": 0.5, "tp_r_mult": 1.0, "trail_atr_mult": 3.0, "trail_kind": "atr_trail"}   |          1.2869 |     0.8634 |   1.0873 |    0.0531 |        51.5000 |             0.2036 |           1.5603 |
| rsi_extreme_filtered | {"max_adx": null, "overbought": 80, "oversold": 20, "rsi_length": 14, "session": "london_ny_overlap"} | {"atr_length": 14, "sl_atr_mult": 0.5, "tp_r_mult": 0.75, "trail_atr_mult": 3.0, "trail_kind": "atr_trail"}  |          1.6012 |     0.8513 |   1.6453 |    0.1196 |       102.0000 |             0.2312 |           1.0610 |
| rsi_extreme_filtered | {"max_adx": null, "overbought": 80, "oversold": 30, "rsi_length": 21, "session": "london_ny_overlap"} | {"atr_length": 14, "sl_atr_mult": 0.5, "tp_r_mult": 0.75, "trail_atr_mult": 3.0, "trail_kind": "atr_trail"}  |          1.1847 |     0.8474 |   0.9142 |    0.0911 |       168.6667 |             0.2941 |           0.5311 |
| rsi_extreme_filtered | {"max_adx": null, "overbought": 75, "oversold": 30, "rsi_length": 30, "session": "london_ny_overlap"} | {"atr_length": 14, "sl_atr_mult": 0.5, "tp_r_mult": 0.75, "trail_atr_mult": 3.0, "trail_kind": "atr_trail"}  |          1.1830 |     0.8460 |   0.4248 |    0.0266 |        74.0000 |             0.2419 |          -0.0463 |
| rsi_extreme_filtered | {"max_adx": null, "overbought": 75, "oversold": 25, "rsi_length": 21, "session": "london_ny_overlap"} | {"atr_length": 14, "sl_atr_mult": 0.5, "tp_r_mult": 1.0, "trail_atr_mult": 3.0, "trail_kind": "atr_trail"}   |          1.2997 |     0.8404 |   1.1611 |    0.0848 |        91.0000 |             0.1688 |           0.7969 |
| rsi_extreme_filtered | {"max_adx": null, "overbought": 80, "oversold": 25, "rsi_length": 14, "session": "active"}            | {"atr_length": 14, "sl_atr_mult": 0.5, "tp_r_mult": 0.75, "trail_atr_mult": 3.0, "trail_kind": "atr_trail"}  |          1.0361 |     0.8375 |   0.5806 |    0.0907 |       504.3333 |             0.4053 |           0.0787 |
| rsi_extreme_filtered | {"max_adx": null, "overbought": 80, "oversold": 20, "rsi_length": 21, "session": "active"}            | {"atr_length": 14, "sl_atr_mult": 0.5, "tp_r_mult": 0.75, "trail_atr_mult": 2.0, "trail_kind": "chandelier"} |          1.2407 |     0.8359 |   0.9984 |    0.0445 |        52.0000 |             0.2139 |           1.3012 |
| rsi_extreme_filtered | {"max_adx": null, "overbought": 80, "oversold": 20, "rsi_length": 14, "session": "london_ny_overlap"} | {"atr_length": 14, "sl_atr_mult": 0.5, "tp_r_mult": 1.0, "trail_atr_mult": 3.0, "trail_kind": "atr_trail"}   |          1.6650 |     0.8353 |   1.9023 |    0.1508 |       102.0000 |             0.2200 |           1.4955 |
```

**Per-family averages across all runs:**
```
| family               |   profit_factor |   win_rate |   sharpe |   sortino |   total_trades |   max_drawdown_pct |   expectancy_usd |
|:---------------------|----------------:|-----------:|---------:|----------:|---------------:|-------------------:|-----------------:|
| bb_rsi_mr            |          0.7278 |     0.5093 |  -8.7700 |   -4.0185 |      3606.1861 |           inf      |          -0.2482 |
| bb_rsi_mr_filtered   |          0.9310 |     0.5567 |  -0.5883 |   -0.0541 |       321.1611 |             0.3770 |          -0.2806 |
| ema_cross            |          0.6315 |     0.4234 | -20.6797 |  -11.2596 |      6022.6711 |           inf      |          -0.0950 |
| pullback_ema         |          0.6365 |     0.4580 | -19.4325 |  -14.8072 |      9872.3322 |           inf      |          -0.0579 |
| range_breakout       |          0.4767 |     0.4115 | -13.6555 |   -4.6936 |      2059.6625 |             0.8272 |          -0.8105 |
| rsi_extreme          |          0.7246 |     0.5076 | -11.0319 |   -6.7790 |      5179.6183 |             0.9312 |          -0.1712 |
| rsi_extreme_filtered |          0.8852 |     0.5351 |  -0.8215 |   -0.0363 |       126.6974 |             0.2731 |          -0.1850 |
| vwap_deviation       |          0.7385 |     0.5086 | -11.8263 |   -7.8832 |      7999.1372 |           inf      |          -0.1022 |
```

**Notes from the operator:** Massive turn from round 1. 74 configs (of 3,526) have PF>1.1 + positive expectancy across ALL 3 walk-forward splits. Top: PF 2.01, +$0.84/trade, 77% WR, Sharpe 1.73, DD 18%. STRIKING pattern: 52 of 74 winners use session=london_ny_overlap (12-16 UTC), 70 of 74 have max_adx=None (ADX filter not adding value; session filter IS the edge). Top 3 winners all rsi_extreme_filtered. Unfiltered versions of every family still lose (pf < 1.0). Questions: (1) Is 74/3526=2.1% winners at PF>1.1 statistically significant or multi-testing noise? (2) The top rsi_extreme_filtered config has max_adx=None (meaning the ADX filter had no effect) — does that tell us something specific? (3) What should round 3 test: higher timeframes (M5/M15/M30), different pairs, or more session-filter variants? (4) Anything we should be suspicious of in these results — look-ahead bias, survivorship, walk-forward leakage?

Given this data, draw on vectorbt Pro's documentation and examples to answer:

1. What PATTERNS do you see in the top performers? Any common parameter
   regions (short EMAs? wide stops? trail-off configs?) that correlate
   with strong results?
2. What is MISSING from this sweep? Which signal families, indicator
   combinations, or exit structures should we add for the next iteration?
3. Which vbt Pro features would materially improve the quality of this
   exploration? (Splitter variants? parameter optimization with Optuna?
   vbt.ranges / vbt.trades accessors I should be using?)
4. Any red flags in these numbers that suggest overfitting, data leaks,
   or miscalibration (e.g. win rates or profit factors that look too good
   to be real at this frequency)?

Please be specific — reference vbt modules / function names / example
notebook patterns where applicable.


## Answer

You’ve run a broad walk-forward sweep on EUR/USD M1 and want to know what the results are actually saying, what to test next, and which VectorBT PRO tools are most useful for improving the research quality.

## Short answer

Yes — there **are** patterns in your winners, and the strongest one is not the indicator itself but the **context filter**: `session="london_ny_overlap"` dominates, while `max_adx=None` appearing in most winners suggests the ADX filter is **not contributing meaningful edge** in this setup.

The next iteration should focus less on adding random indicators and more on:

- **stress-testing the session effect**
- **cross-validating across more splits / instruments**
- **testing neighboring parameter stability**
- **checking whether the edge survives with realistic execution assumptions**

And in VectorBT PRO, the most relevant tools from the provided context are:

- `@vbt.parameterized(...)`
- `vbt.Param(...)`
- `vbt.split(...)`
- chunking / mono chunks such as `mono_n_chunks="auto"`
- train/test cross-validation workflows from the cross-validation examples
- `pf.trades` / `pf.positions` metrics like `expectancy`

Relevant references:
- Optimization via parameterized decorator: https://vectorbt.pro/features/optimization/#parameterized-decorator
- Cross-validation article example: https://www.pyquantnews.com/easily-cross-validate-parameters-boost-strategy/
- CV tutorial pattern on column stacking / train-test application: https://vectorbt.pro/pvt_16ebf9ef/tutorials/cross-validation/applications/#column-stacking
- Stop/exit exploration tutorial: https://vectorbt.pro/pvt_16ebf9ef/tutorials/stop-signals/#parameters
- Indicator docs / snippets mentioned in Discord:
  - https://vectorbt.pro/pvt_16ebf9ef/documentation/indicators/
  - https://vectorbt.pro/pvt_16ebf9ef/cookbook/indicators/
- Pairs-trading tutorial showing broad parameter grids and grouping:
  - https://vectorbt.pro/pvt_16ebf9ef/tutorials/pairs-trading/#level-researcher

---

## 1. Patterns in the top performers

Based on your tables, I see these recurring regions:

### A. The strongest edge appears to be the session filter
This is the clearest pattern.

You already noted:
- `52 of 74` winners use `session=london_ny_overlap`
- `70 of 74` have `max_adx=None`

That implies:

- **time-of-day regime matters a lot**
- **ADX gating does not seem to improve this mean-reversion scalp**
- the signal family alone is not enough; the edge likely depends on **when** it trades

This fits your winners:
- top PF and top WR rows are overwhelmingly `rsi_extreme_filtered`
- most of those use `london_ny_overlap`
- a few use `active`, but overlap is dominant

### B. `rsi_extreme_filtered` is the only family consistently appearing at the top
That suggests the profitable region is currently:
- **short-horizon RSI extremes**
- **with session restriction**
- **with tight ATR-defined risk**
- **plus a trailing component**

Importantly, your per-family averages are still weak overall:
- `rsi_extreme_filtered` average PF is only `0.8852`
- but its top configs are your best OOS performers

So the family has **localized profitable pockets**, not broad robustness yet.

### C. Stops are generally tight, not wide
Your best rows repeatedly use:
- `sl_atr_mult = 0.5` or `2.0`
- often `tp_r_mult = 0.75` or `1.0`
- `trail_atr_mult = 3.0`
- mostly `atr_trail`, occasionally `chandelier`

But there are two different winning styles:

#### Style 1: high WR / scalp-biased
Examples:
- `sl_atr_mult=0.5`
- `tp_r_mult=0.75` or `1.0`
- `trail_atr_mult=3.0`
- `atr_trail`

These produce:
- WR ~84–87%
- PF ~1.6
- expectancy positive but not huge

This is your “base hit” profile.

#### Style 2: lower WR but stronger expectancy
Example:
- `sl_atr_mult=1.0`
- `tp_r_mult=1.0`
- `trail_kind="chandelier"`
- WR ~70%
- PF ~1.44
- expectancy ~$2.07, which is among the better values

That suggests some configs are moving away from pure win-rate harvesting and allowing better runner capture.

### D. The top PF config has no fixed TP
Your #1 PF row is:

- `sl_atr_mult=2.0`
- `tp_r_mult=null`
- `trail_atr_mult=3.0`
- `trail_kind="atr_trail"`

That’s interesting because it implies:
- letting the trail do the exit work may outperform fixed take-profit in at least one robust OOS pocket
- for this setup, **capping winners too early may hurt PF**

So yes, there is a visible pattern:
- **high win-rate configs often use small TP**
- **best PF config removes TP and lets trail manage upside**

### E. BB+RSI mean-reversion only works when filtered hard
`bb_rsi_mr_filtered` appears in the top 10, while `bb_rsi_mr` is terrible on average.

That supports the same conclusion:
- raw signal families are mostly weak
- **context/filtering is doing the heavy lifting**

### F. No evidence here that short EMA families are helping
Your top lists are not showing `ema_cross` or `pullback_ema` winners. Their family averages are very poor. So from your current sweep, I would **not** conclude “short EMAs correlate with strong performance.” The opposite: in this dataset, those families appear unpromising relative to filtered RSI extremes.

---

## 2. What’s missing from the sweep

From the provided context, the main recommendation is brute-force parameterization and broad indicator experimentation. The maintainer explicitly said:

- use the `parameterized` decorator
- check the cookbook for indicators
- you can call any indicator by name, even use an indicator as a parameter

That suggests your next round should expand in a structured way.

## Missing dimensions

### A. More explicit indicator combinations as filters
Since session filtering appears to matter more than ADX, test additional **context filters** rather than only entry generators.

Good next category:
- volatility regime filter
- spread/ATR fraction filter variants
- range compression / expansion regime
- trend-vs-mean-reversion state filter

The Discord guidance explicitly says you can use indicator names and brute-force combinations through the indicators cookbook/docs:
- https://vectorbt.pro/pvt_16ebf9ef/cookbook/indicators/
- https://vectorbt.pro/pvt_16ebf9ef/documentation/indicators/

### B. More session variants
Since session is the strongest visible effect, add finer partitions, for example:
- London open only
- NY open only
- overlap first half vs second half
- pre-overlap ramp
- exclude known dead zones

Your current results strongly justify spending search budget here.

### C. Neighboring exit structures
You have enough evidence that exits matter a lot. The stop-signals tutorial is directly relevant because it explores stop values/types as a research axis:
- https://vectorbt.pro/pvt_16ebf9ef/tutorials/stop-signals/#parameters

Based on your winners, add:

- **trail only**
- **fixed TP only**
- **fixed TP + trail activation**
- **time-based exit / holding exit**
- **random/benchmark-like exits** as sanity baselines, if only for comparison

The stop-signals tutorial explicitly analyzes:
- SL exits
- TS exits
- TP exits
- random exits
- holding exits

That’s exactly the framework you should borrow.

### D. Additional instruments
Your current result is one pair only. From the provided cross-validation discussions, overfitting concerns are a major theme. The strongest next test is not just more parameters, but:
- more pairs
- more market regimes
- more splits

### E. Higher timeframes
Yes — M5/M15/M30 should be tested. Not because they are “better” by default, but because:
- if the edge only exists on M1, it may be microstructure-sensitive
- if it survives on M5, that can improve robustness
- if it disappears entirely, the M1 effect may be execution/noise dependent

### F. A trade-labeling workflow
You mentioned ML-like downstream use. One Discord example discusses labeling trades and using indicators at entry, then training a classifier. That fits your use case well. Even if you do not use ML yet, you should save:
- entry-time indicator snapshot
- regime labels
- trade outcome labels
- holding time
- MAE/MFE if available in your workflow

This will let you reverse-engineer what your winners actually have in common.

---

## 3. Which VectorBT PRO features would materially improve this exploration?

## A. `@vbt.parameterized(...)` with chunking
This is the first thing the maintainer recommended for this exact use case.

Use the parameterized decorator rather than hand-written loops:
- optimization feature page: https://vectorbt.pro/features/optimization/#parameterized-decorator

And if slow, use chunking:
- maintainer: “if it's slow, try using mono chunks”
- example from the PyQuantNews article:
  - `mono_n_chunks="auto"`
  - `execute_kwargs=dict(warmup=True, engine="pathos")`

That’s directly relevant to your grid sweeps.

Example pattern from the provided article:
```python
import vectorbtpro as vbt

param_objective = vbt.parameterized(
    objective,
    merge_func="concat",
    mono_n_chunks="auto",
    execute_kwargs=dict(warmup=True, engine="pathos")
)
```

## B. `vbt.split(...)` for cross-validation / walk-forward
This is probably the most important quality upgrade.

The provided article shows:
- parameterize the strategy
- then wrap with `vbt.split(...)`
- evaluate train/test across many ranges

That is exactly how to reduce false discoveries from a large sweep.

Reference:
- https://www.pyquantnews.com/easily-cross-validate-parameters-boost-strategy/

The pattern:
```python
cv_objective = vbt.split(
    param_objective,
    splitter=splitter,
    takeable_args=["data"],
    merge_func="concat",
)
```

You’re already doing walk-forward, but the documentation example suggests pushing this further into a systematic split workflow where train and test outputs are stored together and compared directly.

## C. Column stacking / apply best train params to matching test columns
The cross-validation tutorial’s “column stacking” section is highly relevant:
- https://vectorbt.pro/pvt_16ebf9ef/tutorials/cross-validation/applications/#column-stacking

Why it matters:
- you can compute train metrics for all combinations
- identify best or most stable params per split
- then apply the selected param combo to the corresponding test segment only

This is much cleaner than manually carrying around best params.

## D. Analyze parameter stability, not just maxima
The CV tutorial explicitly discusses smoothing / neighboring performance and looking for robust areas instead of isolated peaks.

That matters a lot for your question “2.1% winners — signal or multiple-testing noise?”

What you want to know is:
- are the good configs isolated spikes?
- or do neighboring parameters also work?

If profitable configs cluster around:
- RSI length 14–21
- OB/OS around 75/25 to 80/20
- trail around 3 ATR
- session around overlap

that’s much more convincing than one best tuple.

## E. `pf.trades.expectancy`
This metric is explicitly recommended by the maintainer in the provided Discord context as more useful than focusing only on win rate. Your tables already include expectancy, which is good.

The maintainer said:
- don’t focus only on win rate
- profit factor is limited
- consider `pf.trades.expectancy`
- Sharpe above 1–2 is solid, but context matters

So yes, keep ranking by:
- expectancy
- PF
- drawdown
- OOS consistency

not win rate alone.

## F. `pf.positions.readable` / trade-level inspection
The de Prado-related Discord message shows `pf.positions.readable` and discusses inspecting how trades are represented. This is useful for validating:
- are you getting the intended number of trades?
- are entries/exits occurring when expected?
- are reversals and same-bar effects creating artifacts?

Trade-table inspection is essential when a strategy looks surprisingly good.

## G. Indicator factory / custom indicators
If you want to add more sophisticated filters or exit logic, the provided indicator example shows using `vbt.IF(...).with_apply_func(...)` to build custom indicators, such as ATR or Keltner-derived TP/SL logic.

Reference:
- verified code in indicators channel showing `ATRVec = vbt.IF(...).with_apply_func(...)`

That’s a good path if your next iteration needs custom regime filters or dynamic stop logic.

---

## 4. Red flags / overfitting / leakage concerns

## A. 74 winners out of 3,526 is not convincing by itself
`74 / 3526 = 2.1%` profitable configs with PF > 1.1 and positive expectancy across all 3 splits is **not enough on its own** to claim significance.

Why:
- you ran many tests
- some winners will appear by chance
- the key question is whether winners are **clustered and stable**, not how many exist

What would make it more convincing:
- winners cluster in a small interpretable region
- neighboring params also work
- same region survives on other pairs / timeframes / additional splits

From your data, there **is** one encouraging sign:
- winners are not scattered randomly across all families
- they cluster in `rsi_extreme_filtered`
- mostly during `london_ny_overlap`
- with similar exit structures

That’s more promising than random isolated winners.

## B. Average trade counts are lower than your stated target
You said goal is `5+ trades/day`. Your top configs show:
- ~78 to ~146 trades aggregated across OOS splits on average
- not obviously consistent with a true high-frequency scalp profile over long periods unless split accounting is averaging in a specific way

So I’d verify:
- what exactly `95.3333` trades means
- per split? average across splits? normalized?
- actual trades/day on each test segment

This is not necessarily a flaw, but it’s something to reconcile with the strategy objective.

## C. M1 FX scalping is highly sensitive to execution realism
This is the biggest practical red flag.

At M1 frequency, small positive expectancy can be wiped out by:
- spread variation
- slippage
- fill latency
- stop execution assumptions
- bar-resolution ambiguity

Since your edge is around roughly `$0.84` to `$2.07` expectancy per trade on `$5,000` notional, execution modeling matters a lot.

The support thread about discrepancies with MT5 vs VBT shows the maintainer’s key advice:
- compare the signals first
- trade count differences often explain mismatched results

That applies here too:
- inspect actual signals and stop behavior carefully
- especially because trailing exits on OHLC bars can be very assumption-sensitive

## D. Sortino values look oddly small relative to Sharpe in the top rows
Some rows have:
- Sharpe ~1.7 to 2.4
- Sortino ~0.1 to 0.19

That asymmetry is worth auditing. It may be fine depending on annualization / return sampling / downside definition, but it’s unusual enough that I would re-check:
- return frequency
- annualization
- whether per-trade or per-bar returns are used
- whether many tiny negative bars distort downside stats

## E. `inf` drawdowns in per-family averages
Several family averages show `max_drawdown_pct = inf`.

That is a red flag for:
- portfolio/value path issues
- blow-ups or invalid divisions
- possibly pathologies in bad configurations

Even if top configs look fine, this suggests the sweep includes unstable runs and the metric pipeline should be checked.

## F. ADX “not helping” probably means regime filter mismatch, not universal irrelevance
The fact that `max_adx=None` wins does **not** mean ADX is useless in general.

It likely means one of:
- the strategy’s edge is strongest in overlap-session liquidity conditions, and ADX is redundant there
- the chosen ADX threshold region is wrong
- ADX is filtering out too many good reversions
- trend-strength filtering is mismatched to a mean-reversion entry family

So the specific conclusion is:
- **in this tested space, ADX did not improve the profitable pocket**
not:
- ADX never helps

## G. Only 3 walk-forward splits is still fairly light
It’s better than in-sample optimization only, but still not that much evidence for a noisy M1 strategy. The provided CV material strongly points toward richer split-based testing.

## H. Watch for session-definition leakage
If session boundaries were defined using only timestamps, fine. But if “active” or overlap labels were derived using information from the bar stream in a way that indirectly references future bars, that would be a problem. Same for ATR/trailing logic if exits are evaluated unrealistically within the same bar.

---

## Direct answers to your four embedded questions

## (1) Is 74/3526 statistically significant or multi-testing noise?
From the provided context alone: **not enough to claim significance**.

More convincing evidence would be:
- cross-validation via more splits using `vbt.split(...)`
- stability across neighboring params
- replication across instruments
- parameter-region clustering rather than isolated winners

## (2) Does `max_adx=None` tell us something specific?
Yes:
- ADX filter is probably **not adding incremental value** to this setup
- the session filter seems to be the dominant contextual edge
- this may indicate your profitable setup is more about **time-of-day mean reversion** than trend-strength filtering

## (3) What should round 3 test?
Priority order from your current evidence:

1. **More session variants**
2. **More instruments/pairs**
3. **More CV splits**
4. **Neighboring exit variants**
5. **Higher timeframes M5/M15/M30**

If budget is limited, I would not spend the next round primarily on more random indicator families.

## (4) Anything suspicious?
Yes, mainly:
- M1 execution realism
- low number of robust winners relative to search space
- only 3 OOS splits
- bar-based trailing stop assumptions
- odd Sharpe/Sortino relationship
- `inf` drawdowns in family summaries

---

## Practical next-step plan in VBT PRO

## Step 1: Re-run with structured parameterization
Use:
- `@vbt.parameterized(...)`
- `vbt.Param([...])`
- chunking via `mono_n_chunks="auto"`

As recommended in the optimization docs and Discord.

## Step 2: Wrap with split-based CV
Use `vbt.split(...)` exactly like the provided article example to store all split/param results in one object.

## Step 3: Rank by robustness, not best PF
For each config, compute:
- mean test PF
- minimum test PF across splits
- mean expectancy
- maximum DD
- count of profitable test splits

Then prefer:
- high minimum quality
- clustered neighboring params
- acceptable DD

## Step 4: Inspect trade tables for top configs
Use trade/position-level inspection to verify:
- entries/exits are sensible
- trailing stops are behaving as expected
- no accidental signal duplication

## Step 5: Expand only the dimensions your results say matter
Focus next on:
- session granularity
- exit regimes
- more pairs
- more splits

---

## Final interpretation

Your sweep does show a real **pattern candidate**:

- profitable pocket is concentrated in **`rsi_extreme_filtered`**
- mostly during **London/NY overlap**
- with **tight ATR risk and trailing exits**
- and **ADX filtering appears unnecessary in this tested region**

That is much more informative than “RSI works.” The stronger conclusion is:

> the current edge appears to be a **time-of-day-conditioned mean-reversion effect**, and exits/session filters matter more than adding generic trend filters.

But it is **not yet robust enough** to trust without:
- more split-based CV
- more instruments
- parameter-neighborhood stability analysis
- more realistic execution assumptions

## Relevant links

- Optimization / parameterized decorator: https://vectorbt.pro/features/optimization/#parameterized-decorator
- Cross-validation example: https://www.pyquantnews.com/easily-cross-validate-parameters-boost-strategy/
- CV tutorial, column stacking: https://vectorbt.pro/pvt_16ebf9ef/tutorials/cross-validation/applications/#column-stacking
- Stop-signal parameter research: https://vectorbt.pro/pvt_16ebf9ef/tutorials/stop-signals/#parameters
- Indicators docs: https://vectorbt.pro/pvt_16ebf9ef/documentation/indicators/
- Indicator cookbook/snippets: https://vectorbt.pro/pvt_16ebf9ef/cookbook/indicators/
- Parameter-grid / grouping example: https://vectorbt.pro/pvt_16ebf9ef/tutorials/pairs-trading/#level-researcher

If you want, I can next turn this into a **VectorBT PRO round-3 experiment design**, including a `@vbt.parameterized` + `vbt.split` template for:
- session variants
- pair expansion
- robustness ranking across splits.
