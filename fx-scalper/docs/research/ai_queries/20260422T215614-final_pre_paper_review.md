# final_pre_paper_review

**When:** 2026-04-22T21:56:14+00:00 UTC
**Provider:** anthropic / `anthropic/default`
**Tokens:** input=796 output=4134
**Estimated cost:** $0.0644

## Question

fx-scalper FINAL pre-paper-trade review. Seeking your RED-FLAG analysis.

Six rounds of walk-forward EUR/USD exploration + cross-pair validation on
GBP/USD + USD/JPY, running in vbt.pro. Honest post-sizing-fix numbers:

TOP CANDIDATE (for Phase-5 paper trading):
  Family: bb_rsi_mr_filtered (BB + RSI mean reversion with filters)
  Timeframe: M15
  BB(40, 2.5)  RSI(21, 20/70)  ADX=off
  Session: active (7-16 UTC)  Weekday: tue_fri
  Spread filter: <= 0.5 x ATR
  Exit: SL 2.5 x ATR  TP 0.75R  chandelier trail 2 x ATR
  Sizing: size=5000 size_type=value leverage=50 init_cash=500
        (fixed $100 margin per trade, $5K notional)

Cross-pair results (corrected sizing, full 3.3y EUR / 1.0y GBP / 1.25y JPY):
  EUR_USD: walk-forward OOS PF 2.24 (mean across 3 splits)
  GBP_USD: walk-forward OOS PF 2.18
  USD_JPY: walk-forward OOS PF 0.00  (strategy completely fails)

Portfolio layer: EUR/USD + GBP/USD run together with cash_sharing=True,
group_by=True in one vbt.Portfolio.from_signals call. Per-bar return
correlation between the two: 0.001 (zero). Joint-DD fraction 28.7%.

Round 7 diagnostics: per-trade MAE / MFE captured via
pf.trades.records_readable. MAE p25 at -0.10%, p10 at -0.20%. MFE p50
+0.11%, p90 +0.31%. SL at 2.5 x ATR (~0.18% on M15 EUR/USD) sits at
MAE p5 — stops fire rarely on real adverse excursions.

FIVE RED-FLAG QUESTIONS before we touch real capital with \$500 live:

1. THIN GBP DATA. GBP has only 12 months of Dukascopy data loaded
   (2025-04 -> 2026-04). The PF 2.18 on GBP walk-forwards to ~4-month
   OOS splits (thin). How much weight should I put on that number?
   If the cross-pair GBP result is fragile, what happens to the
   two-pair portfolio thesis?

2. TRADE DISTRIBUTION. 480 trades over 3.3 years on EUR/USD M15 =
   ~145 trades/year = ~12/month. Is that enough sample size for a
   retail trader to judge live-vs-backtest tracking within 1 sigma in
   the first 30 days of paper? What are the vbt-supported tools to
   compute the minimum-observation-window needed?

3. STATIONARY-REGIME ASSUMPTION. Strategy was trained on 2023-01 ->
   2026-04. Mean-reversion on EUR/USD worked in that window. What
   PRICE-ONLY regime indicators should I monitor (realized-vol
   percentile, Hurst, ATR regime, autocorrelation) to know when the
   edge is degrading?  Any vbt.pro indicators you would specifically
   call out?

4. OVERFITTING vs 'TRUE EDGE'. 6 rounds + ~30K OOS configs tested +
   WFA used throughout. By my read of Harvey / Liu multiple-testing
   the PF 2.24 number should be haircut to maybe PF 1.4-1.6 for
   multi-testing. Is there a vbt.pro / vectorbt Pro pattern for
   bootstrap CI on PF or computing deflated Sharpe?

5. EXECUTION GAP. Backtest uses Dukascopy bid/ask tick data resampled
   to M15 mid-close. OANDA real fills will have a few ms latency,
   occasional requotes, potentially wider spreads on news spikes, and
   US FIFO / no-hedging rules. What is the sharpest back-of-envelope
   for execution degradation I should expect on a 78% WR M15 scalper
   entering with market orders?

Be specific on vbt.pro tools. No news/event-calendar recommendations —
that layer is explicitly out of scope for this pass.


## Answer

You want a **red-flag review** of this FX scalper before paper trading, with emphasis on what is actually supportable in `vectorbtpro`.

## Short answer

Your biggest red flags are:

1. **GBP/USD evidence is too thin to strongly support the 2-pair portfolio thesis.**
2. **USD/JPY total failure is a regime/generalization warning**, not just a “different pair” footnote.
3. **Your stop looks very loose relative to observed MAE**, which can inflate win rate while hiding unfavorable tail behavior.
4. **Execution realism is likely the biggest live-degradation source** for a high-win-rate M15 scalper.
5. **30K tested configs means you should treat headline PF as optimistic** unless you do explicit resampling / multiple-testing adjustment.

Below is the more concrete breakdown.

---

## 1. Thin GBP data: how much weight should you put on PF 2.18?

**Not much by itself.**

Based on the context, people repeatedly stress that:
- one OOS path is not enough,
- walk-forward helps,
- robustness comes from **multiple OOS observations** and **statistical properties**, not one attractive point estimate,
- cross-validation and walk-forward are the main recommended tools.

So the red flag is not that GBP/USD is profitable, but that **12 months of data with ~4-month OOS splits is too little to anchor the diversification thesis**. A high PF from a short sample can be fragile.

### Practical interpretation
- Treat the GBP result as **supportive but weak** evidence.
- Treat the portfolio thesis as **primarily EUR/USD-led** until GBP has more OOS history.
- If GBP is removed and the idea still looks good, that is a better sign than needing GBP to “rescue” portfolio stats.

### What happens to the 2-pair thesis?
Your portfolio claim should be framed as:

- **Strong claim:** “EUR/USD edge exists.”
- **Weak claim:** “GBP/USD may share the same edge.”
- **Unproven claim:** “A robust 2-pair production portfolio exists.”

The fact that `USD_JPY` completely fails is an important warning that this is **not a broad FX-invariant edge**. It may be a **specific EUR/GBP structural effect** rather than a general mean-reversion pattern.

### VBT-relevant angle
Use the same portfolio setup you already mentioned:
- one `vbt.Portfolio.from_signals` call
- columns = pairs
- `group_by=True`
- `cash_sharing=True`

But evaluate:
- EUR alone
- GBP alone
- EUR+GBP together

and compare whether the combined profile is meaningfully better or just cosmetically better because of one short sample.

Relevant example pattern from Discord:
- multi-asset portfolio via columns with `group_by=True` and `cash_sharing=True`

---

## 2. Trade distribution: is ~12 trades/month enough for first-month live-vs-backtest tracking?

**No, not for a confident 1-sigma judgment in 30 days.**

With ~12 trades/month, your first 30 days of paper trading will be **too noisy** to meaningfully confirm or reject the strategy unless the divergence is very large.

From the context:
- users recommend **walk-forward**, **Monte Carlo**, and **bootstrap simulations** for robustness,
- one should analyze statistical properties rather than over-read a short live segment.

### Red flag
A 78% win-rate strategy with relatively small MFE numbers and execution-sensitive entries can look fine over 10–15 trades even if live execution is already degrading it. The first month is likely a **sanity-check window**, not a statistically stable validation window.

### Better framing
Use first 30 days to validate:
- signal parity,
- trade count parity,
- spread filter behavior,
- fill direction correctness,
- stop/exit logic parity.

Do **not** use first 30 days to estimate stable PF.

### What VBT-supported tools are implied by the context?
The context explicitly mentions:
- **Monte Carlo (1000 bootstrap simulations)**
- **walk-forward validation**
- extracting trade records from `pf.trades.records_readable`
- using cross-validation tutorial concepts for OOS statistical analysis

So the practical VBT-style approach is:

1. Extract trade returns from `pf.trades`.
2. Bootstrap sequences of trades.
3. Ask: after `N` trades, what is the distribution of PF / win rate / avg trade / drawdown?
4. Find the smallest `N` where the confidence interval is narrow enough for your decision threshold.

You can do this from `pf.trades.records_readable` since that is already used in your workflow.

Conceptually:

```python
import vectorbtpro as vbt
import numpy as np
import pandas as pd

trades = pf.trades.records_readable
trade_pnl = trades["PnL"].to_numpy()

# Then bootstrap custom metrics over prefixes of length N
```

There isn’t anything in the provided context stating a built-in one-liner for “minimum observation window”, so I would not claim one. But the **pattern** supported by the context is:
- use trade records,
- bootstrap / Monte Carlo,
- evaluate stability of metrics over sample size.

---

## 3. Stationary-regime assumption: what price-only regime indicators should you monitor?

This is a very good red-flag question, and your instinct is right.

From the context, regime awareness is encouraged via:
- **expanding windows**,
- **walk-forward**,
- robustness over changing regimes,
- using indicators and custom metrics.

### The main red flag
Your edge appears pair-specific and likely regime-sensitive. Since `USD_JPY` fails entirely, I would monitor **whether EUR/USD and GBP/USD are still in the type of market micro-regime where this mean-reversion logic works**.

### Price-only regime measures you named that make sense
From your list, the most relevant are:

- **Realized volatility percentile**
- **ATR regime**
- **Autocorrelation**
- **Hurst-like persistence/anti-persistence proxy**

These are all reasonable price-only diagnostics for a mean-reversion strategy.

### What to monitor operationally
For an M15 mean-reversion system, I would watch whether:
- short-horizon volatility is moving into an extreme percentile,
- ATR relative to recent baseline is expanding,
- return autocorrelation becomes less mean-reverting / more trend-like,
- trade MAE/MFE profile shifts unfavorably.

That last one is especially important because you already compute MAE/MFE from `pf.trades.records_readable`.

### VBT-specific indicators from context
The context explicitly mentions or shows use of:
- `vbt.RSI.run(...)`
- SMA / MA usage
- ATR-based stop logic
- indicator cookbook / indicators callable by name
- custom indicators with `vbt.IF().with_apply_func()`

So the safest answer from context is:

- use built-in indicators such as **RSI**, **ATR**, and moving averages as state filters;
- if you want custom regime metrics like rolling autocorrelation or Hurst proxy, implement them via `vbt.IF().with_apply_func()`.

Example pattern:

```python
import vectorbtpro as vbt
import numpy as np

# Built-ins shown in context
rsi = vbt.RSI.run(close, window=21)
# ATR is implied by ATR-based stops in the context; use it as your vol regime input

# For custom rolling regime metrics:
# create an indicator with vbt.IF().with_apply_func()
```

I would especially monitor:
- ATR percentile over a rolling lookback,
- rolling realized range / realized vol percentile,
- rolling lag-1 return autocorrelation,
- rolling MAE/MFE drift.

Because the context does not explicitly document a built-in Hurst indicator, I wouldn’t claim one.

---

## 4. Overfitting vs true edge: bootstrap CI on PF / deflated Sharpe

Your skepticism is justified.

From the context:
- one user explicitly mentions **Deflated Sharpe** to adjust for multiple testing over **5000+ param combos**
- Monte Carlo / bootstrap simulations are explicitly part of a robust pipeline
- the maintainer points users to **cross-validation tutorial** for assessing OOS statistical properties
- there is no indication in the supplied context of a built-in “deflated Sharpe” API call

### So the supported answer is:
- **Yes**, the VBT PRO pattern is to use cross-validation / walk-forward and custom post-analysis.
- **Yes**, bootstrap CI on PF is consistent with the workflows discussed.
- **Deflated Sharpe is mentioned in the community context**, but no specific VBT PRO built-in function is provided in the supplied material.

### Red flag
With **6 rounds and ~30K OOS configs tested**, your top-line PF absolutely deserves a haircut. That is exactly the kind of setting where people in the context recommend:
- robustness testing,
- walk-forward,
- Monte Carlo,
- and not relying on a single metric like Sharpe.

Also note from the discussion:
- some users consider **Sharpe unreliable** and prefer other metrics like **total return** depending on context,
- so don’t center your decision on one adjusted metric alone.

### VBT-style implementation pattern
Use:
- `pf.trades.records_readable`
- OOS split outputs from your walk-forward runs
- bootstrap over:
  - trade sequence
  - split-level metrics
  - parameter neighborhood stability

The context also mentions:
- sanity-checking **neighbor stability**
- stressing with **fees/slippage bumps**
- evaluating forward windows **without re-sim**

That is exactly the right direction for you.

---

## 5. Execution gap: back-of-envelope degradation for a 78% WR M15 market-order scalper

This is probably your **largest practical red flag**.

From the context, repeated warnings are:

- high-frequency / faster trading is heavily impacted by
  - commissions,
  - spreads/slippage,
  - execution quality,
  - fill assumptions
- with OHLC-type simulation, touched price assumptions can be optimistic
- realistic execution modeling should include
  - slippage,
  - maker/taker or fee modeling,
  - latency,
  - spread blowout / stress
- users report that cost mis-modeling can collapse PF dramatically

### Why this matters especially for your setup
Your system has:
- **high win rate (78%)**
- **small-ish favorable excursions**
- market-order entry
- spread filter based on ATR
- a stop that rarely gets hit

That combination often means the strategy is vulnerable to:
- a few bps worse entry,
- occasional wider spread,
- slightly worse exit,
- FIFO/no-hedging implementation differences.

Because if average edge per trade is modest, small execution drag can flatten it quickly.

### Your own numbers already imply fragility
You reported:
- MAE p25: `-0.10%`
- MAE p10: `-0.20%`
- MFE p50: `+0.11%`
- MFE p90: `+0.31%`

That says a typical trade’s favorable excursion is not huge. So execution loss of even a few hundredths of a percent can matter materially.

### Sharp back-of-envelope
The cleanest stress test, based on the context, is not a single magic estimate but a **friction ladder**:

Re-run with progressively worse assumptions:
- current baseline
- +1 spread unit equivalent
- +2 spread units equivalent
- extra slippage on entry only
- extra slippage on entry and exit
- `price="nextopen"` and `price="nextclose"` style pessimism where applicable
- wider spread during stress

This matches the context’s suggested “Stress & Friction” approach:
- **bump fees/slip**
- rerun with `price="nextopen"` / `price="nextclose"`

### What to expect qualitatively
For this kind of setup, I would expect live PF to degrade the most from:
1. spread widening around entry,
2. exit price worsening on small-profit trades,
3. occasional outlier fills.

Since the context does not provide a numeric formula for expected live PF degradation, the most defensible answer is:
- assume your backtest PF is **optimistic**,
- build a pessimistic envelope with slippage/spread stress,
- and judge whether the strategy still survives.

If it only works under very tight fill assumptions, that is a major red flag.

---

# Specific red-flag conclusions on your setup

## Red flag A: USD/JPY failure is more important than the zero-correlation portfolio result
Per-bar return correlation near zero between EUR/USD and GBP/USD is nice, but it does **not** prove robustness. The complete failure on `USD_JPY` tells you the edge is probably **structurally conditional**. That is a stronger warning than the correlation stat is a comfort.

## Red flag B: your stop may be too wide relative to observed path behavior
You wrote:
- SL at ~0.18%
- MAE p25 `-0.10%`, p10 `-0.20%`
- stop sits near MAE p5

If stops almost never fire, the apparent robustness may partly come from a design that converts many trades into “eventually mean-reverted winners” while leaving tail risk under-measured. That can support high PF in calm conditions, but may fail abruptly when regime shifts.

## Red flag C: 30K tested configs means headline PF is not decision-grade alone
You already know this. The proper response is not to discard the strategy but to require:
- bootstrap confidence intervals,
- parameter-neighbor stability,
- stronger friction testing,
- live-paper parity checks.

## Red flag D: first 30 paper days won’t validate edge, only implementation
At ~12 trades/month, first month is mostly about:
- signal parity,
- position sizing parity,
- spread filter parity,
- stop/exit parity,
- broker rule handling.

---

# Concrete VBT PRO-oriented next steps

## 1. Re-run a friction grid
The context explicitly supports stress/friction reruns.

Test:
- baseline
- spread/slippage +25%
- +50%
- +100%
- pessimistic execution timing where possible

And compare:
- PF
- trade expectancy
- win rate
- max drawdown

## 2. Bootstrap trade-level PF intervals
Supported by the context’s Monte Carlo / bootstrap workflow.

Use:
- `pf.trades.records_readable`
- resample trade sequences many times
- compute PF distribution
- report median / 5th percentile / 95th percentile

## 3. Bootstrap sample-size-to-confidence
For `N = 10, 20, 30, ...` trades:
- resample `N` trades repeatedly
- compute distribution of PF / win rate / expectancy
- find when the interval narrows enough for your purposes

This gives your minimum paper-trade observation window.

## 4. Monitor regime drift with indicators + custom IF
Use built-ins where available and custom indicators via `vbt.IF().with_apply_func()` for:
- ATR percentile
- rolling realized vol percentile
- rolling autocorrelation
- MAE/MFE drift

## 5. Treat GBP as provisional, not confirmatory
Keep the portfolio test, but make EUR/USD the main decision anchor until GBP has deeper OOS history.

---

# Bottom-line answer to your five questions

## 1. Thin GBP data
Put **low-to-moderate weight** on PF 2.18. It is **supportive**, not **confirmatory**. If GBP is fragile, your 2-pair portfolio thesis weakens a lot, because it then becomes mostly a one-pair thesis with cosmetic diversification.

## 2. Trade distribution
~12 trades/month is **not enough** to judge live-vs-backtest tracking within 1 sigma after 30 days in any stable way. Use trade-level bootstrap / Monte Carlo from `pf.trades.records_readable` to estimate the minimum observation window.

## 3. Stationary regime assumption
Monitor price-only regime drift via:
- ATR / realized vol percentile,
- rolling autocorrelation,
- Hurst-like custom proxy,
- MAE/MFE profile drift.
Use built-ins like RSI/ATR and custom indicators through `vbt.IF().with_apply_func()`.

## 4. Overfitting vs true edge
Yes, your top PF should be discounted. The context supports:
- walk-forward / CV,
- Monte Carlo / bootstrap,
- Deflated Sharpe as a concept mentioned by users,
but does **not** provide a confirmed built-in one-liner for deflated Sharpe in the supplied material. So do it as custom post-analysis around VBT outputs.

## 5. Execution gap
For a high-win-rate M15 scalper with modest per-trade excursions, execution drag is likely the **main live risk**. The best back-of-envelope is a **friction stress ladder**, not a single scalar estimate:
- worsen spread/slippage,
- stress execution timing,
- check whether PF still survives.
If not, do not trust the live thesis.

---

## Relevant links from the provided context

- Portfolio callbacks / risk sizing example:  
  `https://vectorbt.pro/pvt_16ebf9ef/cookbook/portfolio/#callbacks`

- Index records suggestion for sparse event construction:  
  `https://vectorbt.pro/features/portfolio/#index-records`

- Accumulation for multiple trades:  
  `https://vectorbt.pro/pvt_16ebf9ef/documentation/portfolio/from-signals/#accumulation`

- Accumulation modes:  
  `https://vectorbt.pro/pvt_16ebf9ef/api/portfolio/enums/#vectorbtpro.portfolio.enums.AccumulationMode`

- First steps docs referenced by maintainer:  
  `https://vectorbt.pro/pvt_16ebf9ef/#first-steps`

If you want, I can turn this into a **go / no-go checklist** for your Phase-5 paper trade, using only metrics and tooling patterns supported by the context.
