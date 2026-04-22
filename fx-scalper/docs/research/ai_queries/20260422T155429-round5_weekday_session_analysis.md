# round5_weekday_session_analysis

**When:** 2026-04-22T15:54:29+00:00 UTC
**Provider:** anthropic / `anthropic/default`
**Tokens:** input=607 output=3662
**Estimated cost:** $0.0568

## Question

Round 5 of fx-scalper exploratory backtesting on EUR/USD. Tested weekday filter + finer session partitions on top of round 3 sweep. 3 walk-forward splits on ~3 years of Dukascopy M1 data resampled to M5/M15/M30.

Results: 147 configs with PF > 1.2 + positive expectancy OOS across ALL 3 walk-forward splits (of 8690 total). Top 3:

1. bb_rsi_mr_filtered on M15, session=london_ny_overlap, weekday=tue_fri: PF 2.07, $9.29/trade, 78% WR, 32% DD, 51 trades/split
2. bb_rsi_mr (unfiltered) on M15: PF 1.92, $7.23/trade, 82% WR, 25% DD, 90 trades/split
3. bb_rsi_mr_filtered on M5, session=active, weekday=tue_fri: PF 1.92, $3.65/trade, 79% WR, 21% DD, 78 trades/split

Key pattern findings:
- Weekday: tue_fri (Tue-Fri) top PF 2.07 and 29 winners; mon_thu top 1.78; tue_thu (mid-week only) just 1 winner; all-weekdays PF 1.49
- Session: london_ny_overlap top PF 2.07 (20 winners); active (L+NY combined) most winners at 42; super-fine partitions like london_open_2h mostly failed — too few trades

Questions for vbt.chat:

1. Does dropping Tuesday-through-Thursday (keeping Tue-Fri) vs dropping only Friday (Mon-Thu) tell us something structural about EUR/USD intra-week dynamics, or is this multi-testing noise? What would vbt Pro's cv_split diagnostic pattern add here?

2. 'bb_rsi_mr' family continues to dominate across M1→M30. Is this a sign of genuine MR edge on EUR/USD, or does the literature suggest I should be extra cautious about BB+RSI specifically (popular indicator, crowded)? What vbt tools detect 'crowded trade' decay?

3. For round 6 I want to test position sizing variants via adjust_func_nb. Specifically: (a) Kelly-fraction sizing based on trailing win rate, (b) ATR-inverse sizing (smaller in high-vol), (c) losing-streak size reduction. What vbt pattern/function should I use for each — specific adjust_func_nb recipes would help.

4. vbt.Portfolio.from_signals can accept multiple columns simultaneously with cash_sharing=True. If I run M5+M15+M30 top configs as 3 columns in one portfolio, does vbt give me actual realized correlation between their trade streams, or do I need to compute it from trade records myself?

5. Red flags in these numbers? Anything that suggests overfitting — win rate clusters at 78-84%, DD clusters at 20-45%, all on EUR/USD only. What would YOU test next before trusting this in paper trading?

Be specific with vbt function names and reference existing vbt notebook patterns.

## Answer

You’re asking how to interpret these WFA findings and, specifically, which **VectorBT PRO** patterns are relevant for taking this to the next round without overfitting.

## Short answer

Based on the provided context:

- Your workflow is feasible in VBT PRO.
- For **walk-forward / CV diagnostics**, the key pattern is to use `@vbt.cv_split(...)` with a **custom `selection`** function/template.
- For **multiple simultaneous strategies/configs**, VBT PRO can simulate multiple columns together, and `cash_sharing=True` lets them share capital, but the context does **not** mention a built-in “realized trade-stream correlation” metric. If you need that specifically, safest is to compute it yourself from portfolio/trade outputs.
- For **dynamic sizing**, the documented pattern in the context is to use `adjust_func_nb` together with `adjust_args`, `broadcast_named_args`, and placeholders like `vbt.Rep(...)` / `vbt.RepFunc(...)`.
- For **overfitting concerns**, the context strongly points toward WFA / CV, proper split handling, custom metric selection, and robustness checks rather than trusting a single headline metric.

## 1) Tue–Fri vs Mon–Thu: structure or multi-testing noise?

From the available context alone, VBT PRO does **not** provide a direct “structural market explanation” for why `tue_fri` beats `mon_thu`. So I can’t claim this is an EUR/USD regime fact.

What VBT PRO *can* add is better **cross-validation / walk-forward diagnostics**:

- Use `@vbt.cv_split(...)`
- Return either:
  - metrics, or
  - full `pf` objects
- Provide a custom `selection` function to choose the best parameter combination on each train split

Relevant maintainer example:

```python
def selection(grid_results):
    return pd.Series(grid_results).map(lambda pf: pf.sharpe_ratio).idxmax()

@vbt.cv_split(
    splitter="from_rolling", 
    splitter_kwargs=dict(length=365, split=0.5, set_labels=["train", "test"]),
    takeable_args=["data"],
    parameterized_kwargs=dict(random_subset=100),
    selection=vbt.RepFunc(selection)
)
def sma_crossover_cv(data, fast_period, slow_period, metric):
    fast_sma = data.run("sma", fast_period, hide_params=True)
    slow_sma = data.run("sma", slow_period, hide_params=True)
    entries = fast_sma.real_crossed_above(slow_sma)
    exits = fast_sma.real_crossed_below(slow_sma)
    pf = vbt.PF.from_signals(data, entries, exits, direction="both")
    return pf
```

Also, if your function returns multiple metrics, you can still optimize on one metric by passing `selection` as a template / `vbt.RepFunc(...)` and returning the index of the best parameter combination. The maintainer explicitly suggested this pattern.

So, for your weekday question, what `cv_split` adds is:

- whether `tue_fri` keeps winning **consistently across splits**
- whether it wins by a **stable margin**
- whether neighboring variants like `mon_fri`, `tue_thu`, `mon_thu`, etc. are also decent, or whether the winner is isolated

That last point is important. In the context, users discussing WFA emphasized that once split output is properly merged, it becomes straightforward to compare best train params to test results. If `tue_fri` is a lone spike while nearby weekday filters collapse, that’s a red flag.

## 2) BB+RSI dominance across M1→M30: genuine MR edge or caution?

The provided context does **not** include literature conclusions about BB+RSI being crowded or decaying. So I can’t support a claim either way from the supplied material.

What the context *does* suggest:

- brute-force indicator/filter exploration is a valid use case
- the maintainer recommended using:
  - indicator snippets
  - indicator documentation
  - tutorials
- several users explicitly warned about **overfitting**
- one user noted that holdout / unseen data testing is essential

So the supported answer is: **be cautious** and validate it via WFA / CV and robustness rather than assuming persistence.

As for VBT tools for “crowded trade decay”: nothing in the provided context names a built-in crowdedness/decay detector. What you *can* do with VBT PRO from the context:

- perform walk-forward analysis with `cv_split`
- split data into rolling periods
- compare train vs test behavior
- inspect trade data via:
  - `pf.trades.records_readable`
  - `pf.trade_history`

The maintainer explicitly said:

> `pf.trades.records_readable` or `pf.trade_history` contains all trade information, including entry/exit time and P&L.

So if you want to detect decay, the supported route is to analyze trade records over time rather than rely on a specific built-in “crowded trade” diagnostic.

## 3) Round 6 sizing via `adjust_func_nb`: what pattern should you use?

Yes — the exact pattern mentioned in the context is:

- `vbt.Portfolio.from_signals(...)`
- `adjust_func_nb=...`
- `adjust_args=(...)`
- `broadcast_named_args=...`
- `size=vbt.RepFunc(lambda wrapper: np.full(wrapper.shape_2d, np.nan))`
- placeholders like `vbt.Rep("size")`, `vbt.Rep("sl_stop")`, etc.

Example from the context:

```python
pf = vbt.Portfolio.from_signals(
    close=close,
    adjust_func_nb="adjust_func_nb_6.py",
    adjust_args=(
        vbt.Rep("size"), 
        vbt.Rep("sl_stop"), 
        vbt.Rep("delta_format"), 
        vbt.Rep("risk_amount"),
        vbt.Rep("atr"),
        multipliers,
        exit_sizes
    ),
    size=vbt.RepFunc(lambda wrapper: np.full(wrapper.shape_2d, np.nan)),
    sl_stop=final_sl,
    delta_format=0,
    broadcast_named_args=dict(
        risk_amount=risk_amount,
        atr=atr,
        multipliers=multipliers,
        exit_sizes=exit_sizes
    ),
    ...
)
```

And another working pattern:

```python
pf = vbt.PF.from_signals(
    data,
    entries=filtered_entries,
    exits=exits,
    adjust_func_nb=adjust_func_nb,
    adjust_args=(
        vbt.Rep("size"), 
        vbt.Rep("sl_stop"), 
        vbt.Rep("delta_format"), 
        vbt.Rep("risk_amount"),
        vbt.Rep("atr"),
        vbt.Rep("multipliers"),
        vbt.Rep("exit_sizes"),
    ),
    size=vbt.RepFunc(lambda wrapper: np.full(wrapper.shape_2d, np.nan)),
    sl_stop=final_sl,
    freq="1day",
    delta_format=0,
    broadcast_named_args=dict(
        risk_amount=risk_amount,
        atr=atr,
        multipliers=multipliers,
        exit_sizes=exit_sizes
    )
)
```

### How this maps to your three sizing ideas

Using only the context, the right general recipe for all three is:

#### (a) Kelly-fraction sizing based on trailing win rate
Use `adjust_func_nb` and pass in precomputed arrays/stat inputs through `broadcast_named_args`.

Pattern:
- compute your trailing win-rate input outside the simulator
- broadcast it as an argument
- in `adjust_func_nb`, set `size` dynamically from that value

#### (b) ATR-inverse sizing
This is the most directly supported by the context because ATR is already shown being passed into `adjust_func_nb` via `broadcast_named_args=dict(atr=atr, ...)`.

Pattern:
- compute `atr`
- pass it through `broadcast_named_args`
- scale `size` inversely with ATR inside `adjust_func_nb`

#### (c) Losing-streak size reduction
Same general pattern:
- precompute or maintain the necessary input
- pass it into `adjust_func_nb`
- reduce `size` based on the streak logic

### Important caveat from the context
The context shows that users can run into typing/broadcasting issues with `adjust_func_nb`, and the maintainer’s guidance was basically:

- ensure the arrays such as `multipliers` and `exit_sizes` are generated correctly
- isolate failing parameter combinations outside of `split` if needed

So for round 6, I’d keep each sizing rule as a **separate minimal prototype** first before combining them.

## 4) Multi-column portfolio with `cash_sharing=True`: realized correlation available?

The provided context does not mention a built-in portfolio method that returns realized correlation between strategy trade streams.

What is supported in the context:

- `Portfolio.from_signals(...)` can work with multiple columns
- portfolio/trade objects expose rich records and analytics
- trade details are available through:
  - `pf.trades.records_readable`
  - `pf.trade_history`

So if by “actual realized correlation between their trade streams” you mean:
- correlation of trade PnL by timestamp,
- overlap in exposure,
- simultaneous drawdowns,
- return-series correlation across strategy columns,

then based on the context, you should assume you’ll need to compute that from outputs/records yourself unless a specific metric in your own docs turns up.

Also, if later you want a single walk-forward portfolio object, the maintainer said:

- stacking portfolios is possible, but **has side effects**
- another option is to return arrays, stack them, and simulate a portfolio from those arrays

So for multi-timeframe top configs, that “simulate once from arrays / signals” route may be cleaner than stitching many portfolio objects after the fact.

## 5) Red flags and what to test next

From the context, these are the biggest red flags / next checks that are actually supported:

### Red flags
1. **Single asset only**
   - A user in research explicitly noted that similar systems across different assets are more effective than one excellent system on one asset.

2. **Potential overfitting**
   - Multiple users warned about overfitting.
   - WFA/CV is repeatedly emphasized as necessary.

3. **Fine partitions reducing trades**
   - Your own finding that super-fine sessions fail due to too few trades matches the general warning sign: a tiny sample can create fragile winners.

4. **High win rate alone is not enough**
   - The maintainer explicitly said not to focus only on win rate.
   - Expectancy and risk-adjusted behavior matter too.

Relevant metrics called out in the context:
- `pf.trades.expectancy`
- Sharpe ratio
- Profit factor
- Max drawdown
- Win rate
- Total trades

### What I would test next in VBT PRO terms

#### A. Custom `cv_split` selection with one optimization metric, but return multiple metrics
This is directly supported.

If your function returns a DataFrame of metrics, use `selection=vbt.RepFunc(...)` and choose the best param set from your preferred metric, while still keeping the full metrics output.

Maintainer guidance:
- default selection is `RepEval("[np.nanargmax(grid_results)]")`
- you can provide your own:

```python
def selection_func(grid_results):
    print(grid_results)
    best_idx = ...
    return best_idx

vbt.RepFunc(selection_func)
```

This is probably the cleanest next step if you want to optimize on one criterion and inspect others.

#### B. Merge split outputs properly
If you use `@split` / `@cv_split` and want well-formatted DataFrames, the context shows that merge behavior matters:
- `merge_func="concat"` may fail for DataFrames in some cases
- using row stacking solved it for one user
- the cookbook has a section on merging functions

So if your round 6 output becomes messy, this is worth checking.

#### C. Inspect raw trades, not just summary stats
Use:
- `pf.trades.records_readable`
- `pf.trade_history`

The maintainer explicitly recommended these for entry/exit time, price, and P&L.

That’s especially relevant for:
- weekday edge validation
- session overlap analysis
- clustering of losses / regime dependence

#### D. Use rolling splits rather than a single WFA path
The context repeatedly recommends rolling `cv_split` / splitter-based workflows. That won’t solve all overfitting, but it’s better than one in-sample / one out-of-sample comparison.

#### E. Consider returning `pf` objects in CV when needed
The maintainer explicitly showed `cv_split` with functions that return `pf`. That lets you optimize based on portfolio-level properties, e.g. `pf.sharpe_ratio`.

#### F. Be careful when trying to stack WFA portfolio objects
Maintainer note:
- portfolio stacking is the only option if you want concatenation, but it has side effects
- another option is to return arrays and simulate afterward

So for production-style WFA reconstruction, the array-first route may be more robust.

---

## Suggested VBT PRO patterns for your round 6

### 1. Walk-forward optimization with custom selection
Use `@vbt.cv_split(...)` plus `selection=vbt.RepFunc(...)`.

### 2. Dynamic sizing
Use `vbt.Portfolio.from_signals(...)` with:
- `adjust_func_nb`
- `adjust_args`
- `broadcast_named_args`
- `vbt.Rep(...)`
- `vbt.RepFunc(...)`

### 3. Trade-level diagnostics
Use:
- `pf.trades.records_readable`
- `pf.trade_history`

### 4. Daily / session partition workflows
Use:
- `splitter`
- cross-validation tutorial approach

The maintainer explicitly recommended `splitter` for splitting a month into days and running a pipeline on each day.

### 5. If combining multiple configs into one simulation
Use multiple columns in one `Portfolio.from_signals(...)` run. If you later need one WFA portfolio across splits:
- stacking portfolios is possible but has side effects
- simulating once from stacked arrays is the alternative suggested by the maintainer

## Bottom line

From the provided context, the most defensible next move is:

1. Keep your current WFA setup.
2. Add a **custom `selection`** function in `@vbt.cv_split(...)`.
3. Test your sizing variants through **`adjust_func_nb` + `broadcast_named_args`**.
4. Inspect **`pf.trades.records_readable` / `pf.trade_history`** for trade clustering and session/weekday dependence.
5. Don’t rely on win rate alone; include expectancy, PF, drawdown, and trade count.
6. Treat the weekday/session result as **possibly real but not yet proven** until you confirm it remains stable across rolling splits and nearby parameter variants.

## Relevant links

- Accumulation docs: https://vectorbt.pro/pvt_16ebf9ef/documentation/portfolio/from-signals/#accumulation
- Accumulation modes: https://vectorbt.pro/pvt_16ebf9ef/api/portfolio/enums/#vectorbtpro.portfolio.enums.AccumulationMode
- Indicator snippets: https://vectorbt.pro/pvt_16ebf9ef/cookbook/indicators/
- Indicator documentation: https://vectorbt.pro/pvt_16ebf9ef/documentation/indicators/
- First steps: https://vectorbt.pro/pvt_16ebf9ef/#first-steps
- Portfolio stats: https://vectorbt.pro/api/portfolio/base/#stats
- Portfolio Numba analysis: https://vectorbt.pro/pvt_16ebf9ef/api/portfolio/nb/analysis/
- Portfolio Numba records: https://vectorbt.pro/pvt_16ebf9ef/api/portfolio/nb/records/
- CV tutorial reference mentioned by users: https://vectorbt.pro/pvt_16ebf9ef/tutorials/cross-validation/applications/#iteration-schemes

If you want, I can next turn this into a **round-6 implementation template** using:
- `@vbt.cv_split`
- a custom `selection`
- `adjust_func_nb`
- and a multi-column `from_signals` layout for M5/M15/M30.
