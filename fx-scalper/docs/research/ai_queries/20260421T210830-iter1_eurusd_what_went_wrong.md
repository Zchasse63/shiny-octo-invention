# iter1_eurusd_what_went_wrong

**When:** 2026-04-21T21:08:30+00:00 UTC
**Provider:** anthropic / `anthropic/default`
**Tokens:** input=2856 output=4128
**Estimated cost:** $0.0705

## Question

I just ran an exploratory backtest sweep using vectorbt Pro on EUR/USD M1 bars,
2023-01-01 to 2026-04-20, with a common exit framework (SL at N×ATR, optional
fixed TP at N×R, optional ATR/chandelier trail) across six signal families.

**Goal profile:** high-frequency scalping (5+ trades/day), base-hit win rate
plus occasional trailing-stop runners for asymmetric upside. Small account
($500) with 50:1 leverage, $100 margin per trade ($5,000 notional).

**Top 10 configs by profit factor (OOS, walk-forward):**
```
| family      | family_params                                                                                           | exit_config                                                                                                 |   profit_factor |   win_rate |   sharpe |   sortino |   total_trades |   max_drawdown_pct |   expectancy_usd |
|:------------|:--------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------|----------------:|-----------:|---------:|----------:|---------------:|-------------------:|-----------------:|
| bb_rsi_mr   | {"bb_length": 30, "bb_std": 2.5, "rsi_length": 21, "rsi_long_threshold": 25, "rsi_short_threshold": 75} | {"atr_length": 14, "sl_atr_mult": 2.0, "tp_r_mult": 2.0, "trail_atr_mult": 2.0, "trail_kind": "off"}        |          0.9914 |     0.4095 |   0.2010 |    0.0536 |       307.6667 |             0.5340 |           0.3189 |
| bb_rsi_mr   | {"bb_length": 30, "bb_std": 2.0, "rsi_length": 21, "rsi_long_threshold": 20, "rsi_short_threshold": 70} | {"atr_length": 14, "sl_atr_mult": 2.0, "tp_r_mult": 2.0, "trail_atr_mult": 2.0, "trail_kind": "off"}        |          0.9813 |     0.3940 |   0.1402 |    0.1168 |       804.6667 |             0.6685 |           0.2264 |
| bb_rsi_mr   | {"bb_length": 30, "bb_std": 2.0, "rsi_length": 21, "rsi_long_threshold": 20, "rsi_short_threshold": 70} | {"atr_length": 14, "sl_atr_mult": 1.5, "tp_r_mult": 2.0, "trail_atr_mult": 3.0, "trail_kind": "atr_trail"}  |          0.9684 |     0.4536 |   0.0388 |    0.0301 |       776.3333 |             0.6280 |          -0.0204 |
| bb_rsi_mr   | {"bb_length": 30, "bb_std": 2.0, "rsi_length": 21, "rsi_long_threshold": 20, "rsi_short_threshold": 70} | {"atr_length": 14, "sl_atr_mult": 2.0, "tp_r_mult": 1.5, "trail_atr_mult": 3.0, "trail_kind": "atr_trail"}  |          0.9684 |     0.4536 |   0.0388 |    0.0301 |       776.3333 |             0.6280 |          -0.0204 |
| rsi_extreme | {"overbought": 80, "oversold": 25, "rsi_length": 21}                                                    | {"atr_length": 14, "sl_atr_mult": 2.0, "tp_r_mult": 1.5, "trail_atr_mult": 3.0, "trail_kind": "atr_trail"}  |          0.9681 |     0.4348 |   0.1020 |    0.0319 |       336.3333 |             0.5370 |          -0.1355 |
| rsi_extreme | {"overbought": 80, "oversold": 25, "rsi_length": 21}                                                    | {"atr_length": 14, "sl_atr_mult": 1.5, "tp_r_mult": 2.0, "trail_atr_mult": 3.0, "trail_kind": "atr_trail"}  |          0.9681 |     0.4348 |   0.1020 |    0.0319 |       336.3333 |             0.5370 |          -0.1355 |
| bb_rsi_mr   | {"bb_length": 30, "bb_std": 2.0, "rsi_length": 21, "rsi_long_threshold": 20, "rsi_short_threshold": 70} | {"atr_length": 14, "sl_atr_mult": 1.5, "tp_r_mult": 2.0, "trail_atr_mult": 2.0, "trail_kind": "off"}        |          0.9625 |     0.4104 |  -0.1175 |    0.0096 |       871.3333 |             0.6414 |           0.2864 |
| bb_rsi_mr   | {"bb_length": 30, "bb_std": 2.0, "rsi_length": 21, "rsi_long_threshold": 20, "rsi_short_threshold": 70} | {"atr_length": 14, "sl_atr_mult": 2.0, "tp_r_mult": 0.75, "trail_atr_mult": 2.0, "trail_kind": "off"}       |          0.9568 |     0.5937 |  -0.5726 |   -0.1050 |       833.0000 |             0.5686 |          -0.0226 |
| bb_rsi_mr   | {"bb_length": 30, "bb_std": 2.0, "rsi_length": 21, "rsi_long_threshold": 20, "rsi_short_threshold": 70} | {"atr_length": 14, "sl_atr_mult": 2.0, "tp_r_mult": 1.5, "trail_atr_mult": 2.0, "trail_kind": "chandelier"} |          0.9464 |     0.3894 |  -0.3463 |   -0.0695 |       841.6667 |             0.6309 |           0.0283 |
| bb_rsi_mr   | {"bb_length": 30, "bb_std": 2.0, "rsi_length": 21, "rsi_long_threshold": 20, "rsi_short_threshold": 70} | {"atr_length": 14, "sl_atr_mult": 1.5, "tp_r_mult": 2.0, "trail_atr_mult": 2.0, "trail_kind": "atr_trail"}  |          0.9464 |     0.3894 |  -0.3463 |   -0.0695 |       841.6667 |             0.6309 |           0.0283 |
```

**Top 10 by win rate (OOS):**
```
| family         | family_params                                                                                           | exit_config                                                                                                  |   profit_factor |   win_rate |   sharpe |   sortino |   total_trades |   max_drawdown_pct |   expectancy_usd |
|:---------------|:--------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------|----------------:|-----------:|---------:|----------:|---------------:|-------------------:|-----------------:|
| vwap_deviation | {"entry_sigma": 3.0, "sigma_window": 120, "vwap_window": 120}                                           | {"atr_length": 14, "sl_atr_mult": 0.5, "tp_r_mult": 0.75, "trail_atr_mult": 2.0, "trail_kind": "chandelier"} |          0.7115 |     0.7081 |  -7.9747 |   -2.4681 |      3073.0000 |             0.9907 |          -0.1609 |
| vwap_deviation | {"entry_sigma": 3.0, "sigma_window": 120, "vwap_window": 60}                                            | {"atr_length": 14, "sl_atr_mult": 0.5, "tp_r_mult": 0.75, "trail_atr_mult": 2.0, "trail_kind": "chandelier"} |          0.7039 |     0.7070 |  -5.7591 |   -1.2860 |      1641.3333 |             0.9569 |          -0.2875 |
| rsi_extreme    | {"overbought": 80, "oversold": 30, "rsi_length": 21}                                                    | {"atr_length": 14, "sl_atr_mult": 0.5, "tp_r_mult": 0.75, "trail_atr_mult": 2.0, "trail_kind": "chandelier"} |          0.6983 |     0.6997 |  -4.8277 |   -0.8895 |       992.0000 |             0.8052 |          -0.3984 |
| vwap_deviation | {"entry_sigma": 2.5, "sigma_window": 120, "vwap_window": 30}                                            | {"atr_length": 14, "sl_atr_mult": 0.5, "tp_r_mult": 0.75, "trail_atr_mult": 2.0, "trail_kind": "chandelier"} |          0.7111 |     0.6994 |  -7.8726 |   -2.3228 |      2723.6667 |             0.9919 |          -0.1818 |
| bb_rsi_mr      | {"bb_length": 30, "bb_std": 2.0, "rsi_length": 21, "rsi_long_threshold": 20, "rsi_short_threshold": 70} | {"atr_length": 14, "sl_atr_mult": 0.5, "tp_r_mult": 0.75, "trail_atr_mult": 2.0, "trail_kind": "chandelier"} |          0.8799 |     0.6970 |  -1.8482 |   -0.3059 |       918.3333 |             0.6216 |          -0.2424 |
| vwap_deviation | {"entry_sigma": 1.5, "sigma_window": 120, "vwap_window": 60}                                            | {"atr_length": 14, "sl_atr_mult": 0.5, "tp_r_mult": 0.75, "trail_atr_mult": 2.0, "trail_kind": "chandelier"} |          0.6543 |     0.6933 | -18.7930 |  -12.3970 |     11596.0000 |             1.0000 |          -0.0431 |
| vwap_deviation | {"entry_sigma": 1.5, "sigma_window": 120, "vwap_window": 120}                                           | {"atr_length": 14, "sl_atr_mult": 0.5, "tp_r_mult": 0.75, "trail_atr_mult": 2.0, "trail_kind": "chandelier"} |          0.6681 |     0.6919 | -21.8012 |  -16.6178 |     14804.6667 |             1.0000 |          -0.0338 |
| vwap_deviation | {"entry_sigma": 3.0, "sigma_window": 120, "vwap_window": 30}                                            | {"atr_length": 14, "sl_atr_mult": 0.5, "tp_r_mult": 0.75, "trail_atr_mult": 2.0, "trail_kind": "chandelier"} |          0.7063 |     0.6908 |  -5.4531 |   -1.0751 |      1272.3333 |             0.9336 |          -0.3651 |
| rsi_extreme    | {"overbought": 80, "oversold": 25, "rsi_length": 14}                                                    | {"atr_length": 14, "sl_atr_mult": 0.5, "tp_r_mult": 0.75, "trail_atr_mult": 2.0, "trail_kind": "chandelier"} |          0.7020 |     0.6899 |  -5.6839 |   -1.1893 |      1319.3333 |             0.8781 |          -0.3287 |
| rsi_extreme    | {"overbought": 80, "oversold": 25, "rsi_length": 21}                                                    | {"atr_length": 14, "sl_atr_mult": 0.5, "tp_r_mult": 0.75, "trail_atr_mult": 2.0, "trail_kind": "chandelier"} |          0.7302 |     0.6881 |  -2.8435 |   -0.3232 |       357.0000 |             0.5638 |          -0.6697 |
```

**Per-family averages across all runs:**
```
| family         |   profit_factor |   win_rate |   sharpe |   sortino |   total_trades |   max_drawdown_pct |   expectancy_usd |
|:---------------|----------------:|-----------:|---------:|----------:|---------------:|-------------------:|-----------------:|
| bb_rsi_mr      |          0.7354 |     0.4795 |  -9.8735 |   -5.0533 |      4528.9274 |           inf      |          -0.2085 |
| ema_cross      |          0.6337 |     0.3985 | -20.3364 |  -11.3704 |      6054.9651 |           inf      |          -0.0948 |
| pullback_ema   |          0.6494 |     0.4371 | -19.4812 |  -14.1175 |      8786.4800 |           inf      |          -0.0657 |
| range_breakout |          0.4816 |     0.3862 | -12.9813 |   -4.1444 |      1798.0696 |             0.8209 |          -0.8525 |
| rsi_extreme    |          0.7164 |     0.4773 | -10.3636 |   -5.8649 |      4383.0222 |             0.9402 |          -0.2061 |
| vwap_deviation |          0.7458 |     0.4782 | -11.4830 |   -7.5171 |      7560.2533 |           inf      |          -0.1190 |
```

**Notes from the operator:** 0 of 1,292 configs with real trade counts across all 3 walk-forward splits produced positive expectancy OOS. Best profit factor was ~0.99 (still losing). Highest win-rate configs (VWAP deviation, 70%) had catastrophic Sharpe (−7 to −20) — tight-stop trap. Data is real Dukascopy EUR/USD, 1.2M M1 bars, 2023-01-02 → 2026-04-20, with bid/ask spread modeled as per-bar half-spread in slippage. Concern: did we just demonstrate that these simple patterns dont work on minute-bar FX? Or is there a structural issue with my sweep (grid too narrow, wrong timeframe, missing a key signal family)?

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

You ran a fairly broad walk-forward sweep on EUR/USD M1 and the main result is: **nothing is truly profitable OOS yet**, with the best configs clustering around break-even and the highest win-rate setups still losing badly. Based on the provided VBT PRO guidance, here’s how I’d interpret it.

## 1. Patterns in the top performers

### A. The only near-survivors are mean-reversion families
Your OOS top-10 by profit factor is dominated by:

- `bb_rsi_mr`
- `rsi_extreme`

That suggests the strongest signal region in your current search is **short-horizon mean reversion**, not breakout/trend-following.

Your per-family averages reinforce this:

- `bb_rsi_mr`: highest-ish survivability among simple families
- `rsi_extreme`: similar
- `range_breakout`: clearly weakest on average
- `ema_cross` / `pullback_ema`: poor on M1 FX in this setup

So within *your tested design*, **fading extremes works better than chasing momentum**.

### B. Better configs tend to use moderate/wider stops, not ultra-tight stops
The best PF configs mostly use:

- `sl_atr_mult` around `1.5` to `2.0`
- `tp_r_mult` around `1.5` to `2.0`

The worst “high win-rate” configs use:

- `sl_atr_mult = 0.5`
- `tp_r_mult = 0.75`

That’s a classic trap: high win rate, negative expectancy. This is consistent with the maintainer’s warning not to optimize on win rate alone and to prefer metrics like **expectancy** and also Sharpe with caution. In your table, that warning is confirmed directly.

### C. Trail-off or very restrained exits seem better than aggressive trailing
Your best PF rows are mostly:

- `trail_kind: "off"`

The ATR/chandelier trail variants appear in top-10 PF, but often with:
- lower PF,
- weaker Sharpe,
- sometimes negative expectancy.

So for these M1 mean-reversion entries, **trailing may be prematurely cutting what little edge exists**. That fits the observation that the more stable near-break-even rows are often fixed SL/TP structures.

### D. Parameter regions in the “least bad” set are pretty concentrated
For `bb_rsi_mr`, common regions are:

- `bb_length = 30`
- `bb_std = 2.0–2.5`
- `rsi_length = 21`
- long threshold around `20–25`
- short threshold around `70–75`

That’s useful. Even though the edge is not positive yet, your search seems to be identifying a **stable parameter basin** rather than random isolated winners. That’s exactly the kind of thing to inspect further with grouped analysis.

### E. Trade frequency and profitability are fighting each other
Your stated goal is **5+ trades/day**, but the best PF configs have trade counts like:

- ~300
- ~800

across OOS split averages, which may be too low for your desired scalping profile. The high-frequency configs are mostly the worst ones, especially in `vwap_deviation`. So at least in this sweep, **higher turnover is degrading edge after spread/slippage**.

---

## 2. What’s missing from the sweep?

From the provided VBT PRO context, the most important missing pieces are not just “more parameters” but **broader strategy and validation structure**.

### A. More exit designs
You already tested fixed TP, ATR SL, and trailing. But the provided material suggests you should expand exit exploration much more systematically.

Relevant clues:

- The stop-signals tutorial emphasizes parameterizing and analyzing **different exit types and stop values**
- The Keltner/ATR example shows building custom TP/SL logic as an indicator or signal generator
- Dynamic stop adjustment via `adjust_func_nb` is possible in portfolio simulations

So next iteration should include:

- **time-based exits**
  - e.g. max bars in trade
  - session cutoff
  - “if not resolved in N bars, flatten”
- **volatility-adaptive exits**
  - ATR regime-aware stop multipliers
- **one-sided exits**
  - SL only, no TP
  - TP only, no trail
- **hybrid exits**
  - partial fixed TP + runner trail concept, if your simulation supports your intended trade model
- **channel-based exits**
  - Keltner / ATR-band / MA-envelope exits like in the custom `tp_sl_kc` example

Your current results suggest the simple fixed frameworks may be too blunt.

### B. Regime filters
The Discord history repeatedly warns that strategies are not simply “good or bad”; they depend on market regime. That’s likely crucial here.

You should add filters such as:

- volatility regime
- session / time-of-day
- trend filter
- spread filter
- news-proxy exclusion windows if available
- directional bias from higher timeframe context

In your current sweep, the same signal family likely trades:
- dead hours,
- transition hours,
- trend hours,
- mean-reverting hours

That can destroy a weak intraday edge.

### C. Indicator-as-parameter design
A maintainer explicitly said you can use the cookbook for indicators and **call any indicator by name, making it possible to use an indicator as a parameter**.

That means next sweep shouldn’t just vary numeric settings inside a fixed family — it should parameterize the **signal family itself** or major building blocks, such as:

- entry trigger type
- filter type
- confirmation type
- exit trigger type

For example:
- RSI extreme + Bollinger
- RSI extreme + VWAP deviation
- RSI extreme + volume/volatility regime proxy
- breakout + retest filter
- trend filter + mean-reversion entry

### D. More robust signal families
Given your current families, missing common intraday FX structures include:

- opening/session range variants
- higher-timeframe bias + lower-timeframe pullback
- volatility compression/expansion
- channel reversion around adaptive bands
- retest logic after break
- bar-pattern/state-machine entries rather than single-indicator thresholds

The provided support threads also reference opening range breakout and de Prado-style labeling workflows. That suggests a good next step is:
- generate many candidate trades from simpler primary signals,
- label outcomes,
- attach context features at entry,
- then analyze what separates winners from losers.

Even without full ML, that labeling workflow can tell you which entry contexts matter.

### E. Better parameter search shape
This may simply be too rigid a grid. The docs/examples strongly point to:
- `@vbt.parameterized`
- chunking
- parallel execution
- and optionally Optuna

If your parameter ranges are too coarse, you may be missing viable basins; if they are too naive, you may be wasting search budget on obviously bad regions.

---

## 3. Which VBT PRO features would materially improve this exploration?

## A. `@vbt.parameterized`
This is the first feature to lean on. It was explicitly recommended by the maintainer as the easiest and best-practice way to do this kind of search.

Use it to parameterize:

- family parameters
- exit parameters
- even indicator choice / signal family choice
- regime filters

Useful options mentioned in context:

- `merge_func="concat"` for scalar outputs like stats/objective
- `merge_func="column_stack"` for stacked signal arrays
- `mono_n_chunks="auto"` for chunking large parameter grids
- `execute_kwargs=dict(warmup=True, engine="pathos")` for parallel chunk execution

That should make your sweep cleaner and more scalable.

Relevant links:
- https://vectorbt.pro/features/optimization/#parameterized-decorator

## B. Splitters via `vbt.split`
The cross-validation article is very relevant here. It shows combining:

- `vbt.parameterized(...)`
- with `vbt.split(...)`

to run the same objective across many train/test ranges and then compare train vs test behavior.

That matters because your current result already smells like:
- weak edge,
- unstable edge,
- or parameter drift.

The example specifically analyzes train/test correlation after parameterized walk-forward testing. You should do the same:
- measure train vs test rank correlation
- inspect parameter stability across splits
- group results by parameter combinations and compare OOS consistency

Relevant link:
- https://www.pyquantnews.com/easily-cross-validate-parameters-boost-strategy/

## C. Indicator factory / custom indicators
You likely need richer exits and feature engineering. The provided examples show using:

- `vbt.IF(...).with_apply_func(...)`
- custom ATR
- custom TP/SL logic
- indicator-generated exit triggers

This is especially useful if your next iteration includes:
- custom trailing logic
- session-aware features
- composite state filters
- derived channels / adaptive bands

Relevant example:
- custom ATR + Keltner/TP/SL indicator snippet from the indicators channel

## D. Portfolio stop customization with `adjust_func_nb`
If you want truly dynamic stop logic, the `adjust_func_nb` pattern is relevant. The provided thread shows that dynamic stop/take-profit management can be injected into portfolio simulation.

That’s useful for:
- activating trailing only after unrealized profit exceeds threshold
- switching from fixed stop to breakeven
- widening/narrowing stops based on ATR regime

Just note the version-sensitive function issue in that thread:
- `vbt.pf_nb.is_stop_info_active_nb` required a newer version in that case

So if you use this path, make sure you’re on a version where the backend helpers you need are available.

## E. Index records for sparse event-based simulations
If many of your entries/exits are sparse and event-driven, one maintainer suggested using:

- **index records**

instead of full boolean arrays.

That can help if your next iteration includes:
- sparse session entries
- entry at specific event times
- event-based labeling / ORB / regime transitions

Relevant link:
- https://vectorbt.pro/features/portfolio/#index-records

## F. Use trades-based diagnostics, not just portfolio headline stats
From the context, two things stand out:

- maintainer recommended `pf.trades.expectancy`
- there are examples using portfolio outputs like stop-hit timing via orders

For your case, you should inspect:
- trade expectancy
- distribution of holding times
- stop-type hit breakdown
- time-to-hit TP vs SL
- trade outcome by hour/session/regime

The stop-signals discussion and order inspection example suggest using order/trade-level analysis, not only aggregate PF/Sharpe.

---

## 4. Red flags in your numbers

### A. Profit factor below 1 almost everywhere
This is the clearest signal. You already summarized the key point:

- `0 / 1292` configs with real trade counts had positive OOS expectancy
- best PF ≈ `0.99`

That strongly suggests either:
1. there is no tradable edge in the tested setup after costs, or
2. your implementation/cost assumptions are too punitive or misaligned.

Either way, as currently tested, it’s not production-worthy.

### B. High win-rate / strongly negative Sharpe combinations
These are a major red flag for **misleading objective selection**, not necessarily leakage.

Examples:
- win rate around `0.69–0.71`
- PF around `0.65–0.71`
- Sharpe from `-5` to `-21`

This is textbook “base-hit but negative expectancy” behavior. The maintainer explicitly warned against focusing on win rate in isolation. Your sweep confirms that warning exactly.

### C. `max_drawdown_pct = inf` in family averages
That is worth auditing immediately.

Seeing `inf` in grouped family averages suggests one of:
- numerical pathology in aggregation
- some portfolios blowing up to zero/negative value
- bad denominator handling in drawdown calculation
- grouping/averaging issue across invalid results

This is not something to ignore. It could indicate:
- leverage/margin settings interacting badly,
- account depletion in some runs,
- or a stats post-processing issue.

I would inspect raw portfolio outputs for those configs before trusting any grouped family summary.

### D. Minute-bar FX with spread modeled: near-break-even may actually be meaningful
This is not a red flag, but an important calibration point.

You modeled spread via per-bar half-spread slippage. On M1 FX, that can wipe out most naive indicator edges. So the fact that several mean-reversion configs got to PF `0.95–0.99` OOS may actually indicate:
- the signal is not nonsense,
- but it is too weak relative to costs.

That suggests the next edge may come from:
- better entry timing,
- regime filtering,
- lower-turnover subset,
- or improved exit conditioning.

### E. Watch for implementation mismatches around signals and trade counts
One support answer pointed out that if results differ from other platforms, the first thing to compare is **signals and trade count**. For your own framework, I’d verify:

- are entries generated at bar close or next bar?
- are stops/TPs evaluated using your intended intrabar logic?
- are long/short collisions handled as intended?
- is spread/slippage applied once or effectively twice?
- are trade counts consistent with what you expect visually?

If something structural is wrong, it often shows up first as unexpectedly high trade counts or implausible exit behavior.

---

## What I would do next in VBT PRO

### 1. Rebuild the sweep around `@vbt.parameterized`
Parameterize:
- signal family
- family params
- exit structure
- regime filter
- session filter

Use:
- `merge_func="concat"` for objective/stat outputs
- `mono_n_chunks="auto"`
- `execute_kwargs=dict(warmup=True, engine="pathos")`

Reference:
- https://vectorbt.pro/features/optimization/#parameterized-decorator

### 2. Add split-based validation using `vbt.split`
Instead of just ranking OOS outcomes, compute:
- train vs test correlation
- parameter stability across splits
- median OOS by parameter region
- degradation from train to test

Reference:
- https://www.pyquantnews.com/easily-cross-validate-parameters-boost-strategy/

### 3. Optimize for expectancy first, not win rate
From the Discord guidance:
- use `pf.trades.expectancy`
- Sharpe can still be useful
- win rate alone is not enough

I would rank by:
- expectancy
- PF
- drawdown-aware metrics
- and then inspect trade count / stability

### 4. Add trade-level diagnostics
Use trade/order outputs to analyze:
- stop-hit timing
- holding duration
- outcome by hour/session
- outcome by volatility regime
- TP-hit vs SL-hit mix

The stop-signal/order examples in the provided context are the pattern to follow.

### 5. Add custom exit modules with `vbt.IF`
Build:
- time stop
- volatility stop
- Keltner/channel exit
- conditional trail activation

The ATR/Keltner example is a good pattern for this.

### 6. Consider dynamic stop logic with `adjust_func_nb`
Particularly for your goal of:
- many small wins
- occasional runners

You may want:
- fixed SL initially
- TP1 or breakeven activation
- then trail only after favorable excursion

The provided `adjust_func_nb` example is the relevant route, subject to version compatibility.

---

## Bottom line

Based on the numbers you showed, I would **not** conclude “simple patterns never work on minute-bar FX.” I would conclude:

1. **Your currently tested simple families are not profitable OOS after costs.**
2. The only promising region is **mean reversion**, especially:
   - Bollinger + RSI style entries,
   - with `rsi_length ~ 21`,
   - `bb_length ~ 30`,
   - `bb_std ~ 2.0–2.5`,
   - and **moderate stops** (`1.5–2 ATR`),
   - with **trailing often off**.
3. The sweep is missing:
   - regime filters,
   - richer exits,
   - signal-family-as-parameter design,
   - and deeper trade-level diagnostics.
4. The biggest VBT PRO upgrades for your next iteration are:
   - `@vbt.parameterized`
   - `vbt.split`
   - custom indicators via `vbt.IF`
   - trade/portfolio diagnostics such as `pf.trades.expectancy`
   - possibly dynamic stop logic with `adjust_func_nb`
   - index-record workflows if your entries are sparse/event-based

## Relevant links

- Parameterized decorator:  
  https://vectorbt.pro/features/optimization/#parameterized-decorator

- Cross-validation / split + parameterized workflow example:  
  https://www.pyquantnews.com/easily-cross-validate-parameters-boost-strategy/

- Portfolio index records:  
  https://vectorbt.pro/features/portfolio/#index-records

- Indicator snippets:  
  https://vectorbt.pro/pvt_16ebf9ef/cookbook/indicators/

- Indicator docs:  
  https://vectorbt.pro/pvt_16ebf9ef/documentation/indicators/

- First steps/tutorials:  
  https://vectorbt.pro/pvt_16ebf9ef/#first-steps

- Stop-signal tutorial parameters section:  
  https://vectorbt.pro/pvt_16ebf9ef/tutorials/stop-signals/#parameters

If you want, I can help you sketch a **next-iteration VBT PRO search design** using `@vbt.parameterized` + `vbt.split` that tests:
- family choice,
- regime filters,
- exit module choice,
- and expectancy-based ranking.
