# round3_5_portfolio_and_stress_tests

**When:** 2026-04-21T23:50:26+00:00 UTC
**Provider:** anthropic / `anthropic/default`
**Tokens:** input=448 output=3092
**Estimated cost:** $0.0477

## Question

I have round-3 exploratory backtest results on EUR/USD M1-H1 with 6 signal families tested. The best config per timeframe after walk-forward:

| TF | Family | Best PF | Per-trade $ | Trades/OOS split | Max DD% | Best exit config |
|---|---|---|---|---|---|---|
| M5 | rsi_extreme_filtered | 2.37 | $+4.89 | 50 | 26% | SL 0.5xATR, TP 0.75R, chandelier 2xATR, session=active |
| M15 | bb_rsi_mr (unfiltered) | 1.50 | $+7.98 | 103 | 39% | SL 0.5xATR, TP 1.5R, chandelier 2xATR, BB(30,2.5), RSI(21) 35/80 |
| M30 | bb_rsi_mr_filtered | 1.62 | $+6.09 | 45 | 34% | SL 0.5xATR, TP 0.75R, chandelier 1xATR, BB(20,2.25), RSI(14) 25/75 |
| H1 | ema_cross | 1.63 | $+11.64 | 44 | 43% | EMA 21/89, SL 0.5xATR, TP 0.75R, chandelier 2xATR |

RRR on $500: M5=2.80x, M15=6.41x, M30=2.39x, H1=3.61x.

User account: $500 starting, $100 margin/trade, 50x leverage. Considering scale to $1500-2000.

Questions:

1. M15 bb_rsi_mr PF 1.50 DD 39% vs M5 rsi_extreme_filtered PF 2.37 DD 26% — which is the safer paper-trade candidate?

2. What vbt Pro features should I use BEFORE paper trading to:
   - Compute MAE/MFE per trade (are stops sized right?)
   - Detect regime dependency
   - Monte Carlo / bootstrap OOS PF confidence intervals

3. To run 3 strategies simultaneously (M5+M15+M30), what vbt pattern supports sub-portfolios sharing combined equity with ACTUAL correlation measured (not assumed)?

4. 1.3% winner rate (80/6376). What vbt features address multi-testing / false-discovery — White reality check, bootstrap CI, or Superior Predictive Ability?

5. What would YOU test next beyond indicators? Specifically non-indicator factors like position sizing variants (Kelly), dynamic strategy switching, post-news blackouts, weekday filters.

Reference vbt function names / notebook patterns specifically.

## Answer

You’re asking the right pre-deployment questions: not just “which config is best,” but how to stress-test it in VBT PRO before paper trading.

## 1. Safer paper-trade candidate: M5 vs M15

Based on the table alone, **M5 `rsi_extreme_filtered` looks like the safer candidate**:

- **Higher PF**: `2.37` vs `1.50`
- **Lower max DD**: `26%` vs `39%`
- Decent OOS sample per split: `50` trades vs `103` for M15

Caveat: M15 has **more trades per OOS split**, which can make its estimate somewhat more statistically stable, but its drawdown is materially worse and PF materially lower. So if the goal is **safer paper trading**, the M5 setup is the more conservative starting point from the metrics shown.

Also, one Discord point is relevant here: maintainers noted that **expectancy** and Sharpe are often more informative than win rate alone, and that a strategy should not be judged on one metric only. So before choosing finally, I’d compare at least:

- `pf.trades.expectancy`
- `profit_factor`
- `max_dd`
- `total_return`
- trade count / exposure

Relevant note:
- Expectancy was explicitly recommended over focusing only on win rate, and Sharpe > 1–2 was mentioned as solid in general context.

## 2. What VBT PRO features to use before paper trading

### A. MAE / MFE per trade
From the provided context, the main trade-level inspection tools mentioned are:

- `pf.trades.records_readable`
- `pf.trade_history`
- `pf.positions.readable`
- `pf.positions`

These are the first places to extract per-trade information.

A maintainer also said that after running the portfolio, you can **extract and enrich positions data as needed**, which fits your MAE/MFE workflow well. So the practical VBT PRO pattern from the available context is:

1. Run `vbt.Portfolio.from_signals(...)`
2. Pull `pf.positions` or `pf.trades`
3. Enrich each trade/position with excursion calculations and entry-context features

This is also exactly the pattern discussed in the de Prado / trade-labeling use case.

If your strategy involves multiple entries before a common exit, the relevant feature is:

- `accumulate`
- with proper `size` and `size_type`

Maintainer guidance:
- Use **accumulation** to execute multiple trades rather than only the initial one.
- See `AccumulationMode` options.
- You can then extract and enrich positions data.

### B. Detect regime dependency
From the context, the most relevant VBT PRO pattern is:

- `@vbt.cv_split(...)`

This is the recommended pattern for:
- rolling splits
- walk-forward evaluation
- checking whether performance is stable across time slices

The maintainer also recommended:
- Use `splitter` to split data into days / windows and run a pipeline on each split
- See the cross-validation tutorial

So for regime dependency, the most direct pre-paper-trading pattern in VBT PRO is:

- run the same strategy under `@vbt.cv_split`
- return multiple metrics per split
- inspect how PF / return / DD vary by split
- define custom `selection` if you want to optimize on one metric while still storing others

The relevant `cv_split` details from Discord:

- Your function can return **multiple metrics**
- To optimize on one metric but still keep others, pass **`selection` as a template**
- Default selection was noted as:
  - `RepEval("[np.nanargmax(grid_results)]")`
- You can pass a custom selector via:
  - `vbt.RepFunc(selection_func)`

So for regime dependency, I’d use rolling CV and inspect metric dispersion across windows, not just the aggregate best result.

### C. Monte Carlo / bootstrap OOS PF confidence intervals
There is no explicit VBT PRO function name in the provided context for White’s Reality Check, SPA, or a built-in Monte Carlo bootstrap utility.

What **is** in context:

- Discord users explicitly recommended **Monte Carlo methods** for robustness / overfitting investigation
- They also mentioned:
  - **Combinatorial Purged Cross Validation**
  - **System Parameter Permutation**

So, based only on the context, the VBT PRO-aligned way to do this is:

1. Use `@vbt.cv_split` for repeated OOS evaluation
2. Export trade / split-level results
3. Apply your Monte Carlo / bootstrap logic on top of those outputs

In short: **CV is the native pattern explicitly shown; Monte Carlo robustness testing is recommended in discussion, but no specific built-in VBT PRO function is named in the provided material.**

## 3. Running 3 strategies simultaneously with shared equity and measured correlation

The closest supported pattern in the provided context is to run multiple columns / signals together in one portfolio and use portfolio-level cash sharing / grouping mechanics.

What the context explicitly supports:

- Process multiple symbols / columns at once rather than looping symbol-by-symbol
- Use `Portfolio.from_signals(...)`
- Use `accumulate` when multiple trades/signals must coexist
- Extract portfolio / trade records afterward for analysis

Also, for shared capital logic, we have an example from low-level portfolio adjustment code that references:

- `c.cash_sharing`

So shared-equity handling is definitely part of the portfolio engine.

Given your case, the most likely VBT PRO pattern from the context is:

- represent each strategy (`M5`, `M15`, `M30`) as separate columns
- run them together in one `vbt.Portfolio.from_signals(...)`
- use common capital / cash-sharing setup
- then analyze realized interaction through:
  - combined portfolio stats
  - trade histories
  - per-column results

For “actual correlation measured, not assumed,” the context doesn’t name a specific helper. But it does support the workflow of:

- running the strategies together in one portfolio
- then comparing the resulting combined vs standalone trades / positions / stats

If you need multiple entries and staggered behavior, use:

- `accumulate`
- `size`
- `size_type`

And if you need custom dynamic allocation or risk budgeting, the context shows a pattern using:

- `adjust_func_nb`
- `adjust_args`
- `broadcast_named_args`

That pattern is already being used for dynamic risk, stop, and exit ladder logic, so it’s the relevant VBT PRO mechanism if you want sub-strategy sizing to depend on shared portfolio state.

## 4. Multi-testing / false discovery with 6,376 tests and 1.3% winners

From the provided context, the strongest direct guidance is:

- use **walk-forward / `cv_split`**
- use **Monte Carlo methods**
- use **Combinatorial Purged Cross Validation**
- use **System Parameter Permutation**
- be careful about overfitting

There is **no explicit mention** in the provided material of built-in VBT PRO support for:

- White’s Reality Check
- Superior Predictive Ability

So I can’t claim specific VBT PRO functions for those from this context.

What I *can* say from the available material:

- VBT PRO gives you the tooling to run large parameter grids efficiently through:
  - `@vbt.parameterized(...)`
  - indicator parameterization with `vbt.Param(...)`
  - cookbook indicator snippets
- For robustness beyond raw optimization, the documented / discussed pattern is:
  - `@vbt.cv_split(...)`
  - custom metric selection
  - repeated split evaluation
  - Monte Carlo / permutation-style post analysis

So among the choices you listed, **bootstrap CI and repeated OOS/CV evaluation are directly supported by the workflows shown**, while White/SPA are not explicitly documented in the provided context.

## 5. What I would test next beyond indicators

Yes — I would move **away from adding more indicators** and toward **execution / filtering / allocation structure**.

From the context, these are the most VBT-PRO-relevant next steps:

### A. Position sizing variants
Very strong candidate.

The provided examples show dynamic sizing through:

- `adjust_func_nb`
- `size`
- `size_type='Amount'`
- stop-based sizing from ATR / risk amount
- `broadcast_named_args`

This is exactly the right area to test next.

Examples in context include:
- ATR-based stop sizing
- risk-percent sizing
- dynamic ladder exits
- leverage and stop logic

So if you want to compare fixed size vs risk-based vs more adaptive sizing, `adjust_func_nb` is the mechanism explicitly shown.

### B. Dynamic strategy switching / regime routing
Also a strong next step.

There’s no explicit “regime switcher” helper in the provided context, but you can build it with:
- signal arrays you control directly
- split-based evaluation with `cv_split`
- indicator pipelines / parameterization
- multi-column portfolio testing

Since maintainers emphasized that strategies behave differently in uptrends/downtrends and that one metric is not enough, regime-aware switching is a natural next experiment.

### C. Session / time blackouts
Definitely worth testing.

Your own best configs already include:
- `session=active`

And the maintainer explicitly said:
- signal timing is controlled by where you place `True` in your boolean arrays

So:
- post-news blackouts
- weekday filters
- intraday time windows

all fit naturally by masking entry arrays before `Portfolio.from_signals(...)`.

### D. Weekday / intraday segmentation
Yes.

The intraday workflow guidance from Discord was:
- use `splitter`
- split into days
- run per-day pipelines
- control entry/exit timing through boolean arrays

So weekday and session filters are very aligned with recommended workflow.

### E. Exit structure stress-testing
Absolutely.

Your results differ substantially by exit config already. The provided context includes examples of:
- `sl_stop`
- `tp_stop`
- `tsl_stop`
- `tsl_th`
- `stop_ladder="dynamic"`
- `adjust_func_nb`
- partial exits via dynamic TP ladder logic / `exit_sizes`

So before paper trading, I would test:
- fixed TP/SL vs chandelier-only
- partial take-profit ladders
- break-even move logic
- trailing threshold logic

These are better next steps than adding more entry indicators.

## Specific VBT PRO patterns/function names from the provided context

Here are the concrete names/patterns that are actually supported by the context:

### Optimization / brute force
- `@vbt.parameterized(merge_func="concat")`
- `vbt.Param(...)`
- indicator cookbook / snippets
- call indicators by name:
  - `data.run('talib:RSI', ...)`
  - `data.run('talib:MACD', ...)`
  - `data.run('talib:BBANDS', ...)`

Link:
- https://vectorbt.pro/features/optimization/#parameterized-decorator

### Cross-validation / walk-forward
- `@vbt.cv_split(...)`
- `splitter="from_n_rolling"`
- `selection`
- `vbt.RepFunc(selection_func)`

Relevant code pattern was explicitly shown in context.

### Trade / position analysis
- `pf.trades.records_readable`
- `pf.trade_history`
- `pf.positions`
- `pf.positions.readable`

### Multi-entry / multiple trades
- `accumulate`
- `AccumulationMode`
- proper `size` and `size_type`

### Dynamic sizing / stop logic
- `adjust_func_nb`
- `adjust_args`
- `broadcast_named_args`
- `sl_stop`
- `tp_stop`
- `tsl_stop`
- `tsl_th`
- `stop_ladder="dynamic"`

## Practical recommendation from your table

If I had to choose one immediate paper-trade candidate from your summary:

1. **Start with M5 `rsi_extreme_filtered`**
   - best PF
   - materially lower DD
   - cleaner first candidate

2. Before paper trading, run:
   - rolling `@vbt.cv_split`
   - trade-level extraction via `pf.trades.records_readable` / `pf.positions`
   - excursion analysis by enriching positions
   - dynamic sizing / stop sensitivity with `adjust_func_nb`
   - session / weekday blackouts as signal masks

3. Then combine:
   - M5 + M15 + M30 as separate columns in one portfolio
   - shared equity logic
   - compare standalone vs combined DD / PF / expectancy

## Relevant links

- Parameterized decorator:  
  https://vectorbt.pro/features/optimization/#parameterized-decorator

- Indicator cookbook/snippets:  
  https://vectorbt.pro/pvt_16ebf9ef/cookbook/indicators/

- Indicators documentation:  
  https://vectorbt.pro/pvt_16ebf9ef/documentation/indicators/

- First steps:  
  https://vectorbt.pro/pvt_16ebf9ef/#first-steps

- Accumulation enum/options:  
  https://vectorbt.pro/pvt_16ebf9ef/api/portfolio/enums/#vectorbtpro.portfolio.enums.AccumulationMode

If you want, I can turn this into a **concrete VBT PRO notebook plan** with:
- one `@vbt.parameterized` research function,
- one `@vbt.cv_split` walk-forward function,
- one combined multi-strategy `Portfolio.from_signals` template,
- and a trade-enrichment step for MAE/MFE.
