# round6_implementation_plan

**When:** 2026-04-22T21:45:36+00:00 UTC
**Provider:** anthropic / `anthropic/default`
**Tokens:** input=569 output=2450
**Estimated cost:** $0.0385

## Question

fx-scalper round 6 — three concrete vbt.pro API questions for implementation.

CONTEXT (short): EUR/USD walk-forward rounds 1-5 landed on a mean-reversion family (bb_rsi_mr_filtered) at M15 with session filter (london_ny_overlap, 12-15 UTC) and weekday filter (tue_fri), using vbt.Portfolio.from_signals with sl_stop / tsl_stop / tp_stop as fraction-of-price arrays. Top OOS PF 2.07, WR 78%, 51 trades per split. Cross-pair validation showed MR generalizes to GBP/USD (PF 2.04) but NOT USD/JPY (PF 0.00). Round 7 MAE/MFE diagnostics showed our SL at 0.5x ATR is tighter than the MAE p25 of winning trades.

THREE SPECIFIC QUESTIONS:

Q1. POSITION SIZING FIX. Currently init_cash=500, leverage=50 sizes each trade at full-equity x leverage (~dollar25K notional). We want EACH trade to use exactly dollar100 cash margin so dollar5,000 notional at 50:1 leverage. On EUR/USD at 1.08 this is 4,629 EUR units; on USD/JPY at 150 it is 33 USD units. What is the canonical vbt.pro pattern for fixed cash committed per trade regardless of equity? Options: (a) size=5000 with size_type=value (dollar-notional target), (b) adjust_func_nb that computes units = 5000 / entry_price, (c) pre-computed size array broadcast to entry bars. Which is cleanest and interoperates with sl_stop / tsl_stop / tp_stop fraction-of-price arrays? Code snippet please.

Q2. SL-WIDTH ABLATION via vbt.Param. We want to sweep sl_atr_mult across {0.5, 0.75, 1.0, 1.25, 1.5} on top-5 configs in ONE Portfolio.from_signals call (broadcast across columns). Our sl_stop is a Series of sl_atr_mult * atr / close masked to entry bars. What is the vbt.Param / broadcast_named_args pattern that passes a LIST of sl_atr_mult values and returns a multi-column portfolio keyed by (config_id, sl_atr_mult)?

Q3. MULTI-PAIR PORTFOLIO WITH cash_sharing=True (round 8 prep). Plan is to run EUR/USD + GBP/USD top configs as two columns in one Portfolio.from_signals with cash_sharing=True to measure realized correlation + combined DD. Each pair has its own close and entry/exit arrays. Can I pass a dict or multi-index DataFrame directly, or do I need to timestamp-align manually first? Minimum example.

Be concrete — vbt function names, kwargs, snippets. Reference Portfolio.from_signals cookbook / docs by URL if available.

## Answer

You’re asking for the cleanest `vbt.Portfolio.from_signals` patterns for:

1. fixed cash/margin per trade,
2. parameterizing ATR-based stop width with `vbt.Param`,
3. building a multi-pair shared-cash portfolio.

Based on the provided context, here are the most concrete `vbt.pro`-style answers.

---

## Q1. Fixed cash committed per trade

### Short answer

The cleanest canonical pattern is usually:

- **fixed notional per trade**: `size=5000`, `size_type="value"`
- or, if you truly want **dynamic/custom sizing logic**, use `adjust_func_nb`

For your case, since you want **exactly $100 margin at 50:1 leverage = $5,000 notional**, the simplest implementation is:

```python
import vectorbtpro as vbt

pf = vbt.Portfolio.from_signals(
    close=close,
    entries=entries,
    exits=exits,
    size=5000,
    size_type="value",
    leverage=50,
    sl_stop=sl_stop,
    tsl_stop=tsl_stop,
    tp_stop=tp_stop,
    delta_format="percent"
)
```

This is consistent with the maintainer guidance that `size_type="value"` is the way to specify a fixed cash value per order. There is also a related example from support:

- with `init_cash=100`, `size=100`, `size_type="value"`, and `allow_partial=False`, each order is exactly `$100` in value.

So if your intent is **fixed trade notional**, option **(a)** is the cleanest.

### Important distinction

There are two different goals:

- **Fixed notional exposure per trade** → use `size_type="value"`
- **Fixed risk per trade based on stop distance and current cash** → use `adjust_func_nb`

From your wording, you want **fixed committed margin / fixed notional**, not risk-based sizing. So you likely **do not need** `adjust_func_nb` here.

### If you want exact units instead

Option **(b)** / **(c)** also works:

```python
units = 5000 / close
```

then:

```python
pf = vbt.Portfolio.from_signals(
    close=close,
    entries=entries,
    exits=exits,
    size=units,
    size_type="amount",
    leverage=50,
    sl_stop=sl_stop,
    tsl_stop=tsl_stop,
    tp_stop=tp_stop,
    delta_format="percent"
)
```

This is fine too, but compared with `size=5000, size_type="value"`, it’s less elegant unless you specifically need per-bar unit control.

### Best recommendation for your use case

Use **fixed notional**:

```python
import vectorbtpro as vbt

notional_per_trade = 5000

pf = vbt.Portfolio.from_signals(
    close=close,
    entries=entries,
    exits=exits,
    size=notional_per_trade,
    size_type="value",
    leverage=50,
    sl_stop=sl_stop,
    tsl_stop=tsl_stop,
    tp_stop=tp_stop,
    delta_format="percent",
    init_cash=500
)
```

This should interoperate normally with `sl_stop / tsl_stop / tp_stop` arrays expressed as fractions of price.

### When to use `adjust_func_nb`

Use `adjust_func_nb` only if you want something like:

- risk 1% of free cash,
- size based on stop width,
- dynamic allocation from current portfolio cash/value,
- custom per-bar sizing logic.

That pattern is shown repeatedly in the support history via:

- `vbt.pf_nb.get_free_cash_nb(c)`
- `vbt.pf_nb.resolve_stop_price_nb(...)`
- writing to `size[c.i, c.col]`

Relevant support/doc references:
- Cookbook example mentioned repeatedly: **“Risk only 1% of the cash balance with each trade”**
- Dynamic docs: https://vectorbt.pro/pvt_16ebf9ef/documentation/portfolio/from-signals/#dynamic

---

## Q2. SL-width ablation with `vbt.Param`

### Short answer

Yes — `vbt.Param` is the intended pattern.

From support, the maintainer explicitly suggested parameterizing stop arrays like this:

```python
sl_stop = vbt.Param(
    [array1, array2, ...],
    keys=pd.Index([mult1, mult2, ...], name="multiplier")
)
```

So for your ATR stop-width sweep, build one `sl_stop` array per multiplier:

```python
import pandas as pd
import vectorbtpro as vbt

sl_mults = [0.5, 0.75, 1.0, 1.25, 1.5]

sl_stop_param = vbt.Param(
    [mult * atr / close for mult in sl_mults],
    keys=pd.Index(sl_mults, name="sl_atr_mult")
)

pf = vbt.Portfolio.from_signals(
    close=close,
    entries=entries,
    exits=exits,
    sl_stop=sl_stop_param,
    tp_stop=tp_stop,
    tsl_stop=tsl_stop,
    delta_format="percent",
    size=5000,
    size_type="value",
    leverage=50
)
```

That is the closest direct answer supported by the context.

### If you want `(config_id, sl_atr_mult)` in the output

If your top-5 configs are already represented as columns, then each `mult * atr / close` array should preserve those config columns. `vbt.Param(..., keys=...)` will add another parameter level on top, giving you a multi-column result keyed by the existing column identity plus `sl_atr_mult`.

So the pattern is:

- columns already represent `config_id`
- `vbt.Param([...], keys=Index(..., name="sl_atr_mult"))` adds the stop-width dimension

### Example shape intuition

Suppose:

- `close` has columns = top-5 configs
- each `mult * atr / close` returns the same 5 columns

Then:

```python
sl_stop_param = vbt.Param(
    [0.5 * atr / close, 0.75 * atr / close, 1.0 * atr / close, 1.25 * atr / close, 1.5 * atr / close],
    keys=pd.Index([0.5, 0.75, 1.0, 1.25, 1.5], name="sl_atr_mult")
)
```

and pass that directly into `from_signals`.

### About masking to entry bars

One support thread noted an important stop-order caveat:

- ensure your stop-loss array has values for **every timestamp**, not just entries.

So if your current `sl_stop` is masked to entry bars only, that may be fragile. In the support history, the maintainer said:

> Ensure your stop loss array has entries for every timestamp, not just for entries.

That’s especially relevant if you want stop management to continue after entry.

### Related docs/support

- `vbt.Param` for stop arrays: support thread on ATR stop multipliers
- Stop-order parameterization docs:  
  https://vectorbt.pro/pvt_16ebf9ef/documentation/portfolio/from-signals/#stop-orders

That page also shows parameterizing stop configurations and notes how parameter combinations expand columns.

---

## Q3. Multi-pair portfolio with `cash_sharing=True`

### Short answer

You should pass **aligned columnar data** into one `Portfolio.from_signals` call.

From the provided context, `from_signals` operates on standard array-like inputs and processes each column individually. With `cash_sharing=True`, shared cash acts at the **group** level.

There is no example in the provided context saying you can pass a raw dict of unrelated indexes and let it align automatically. So the safe answer is:

- **Yes**, use a multi-column DataFrame (or aligned arrays)
- **Manually align timestamps first**
- Then pass all pair-specific `close`, `entries`, `exits`, etc. as same-shape objects

### Minimum example

```python
import pandas as pd
import vectorbtpro as vbt

# Align all series/dataframes to the same index first
close = pd.concat(
    {
        "EURUSD": eurusd_close,
        "GBPUSD": gbpusd_close,
    },
    axis=1
)

entries = pd.concat(
    {
        "EURUSD": eurusd_entries,
        "GBPUSD": gbpusd_entries,
    },
    axis=1
)

exits = pd.concat(
    {
        "EURUSD": eurusd_exits,
        "GBPUSD": gbpusd_exits,
    },
    axis=1
)

sl_stop = pd.concat(
    {
        "EURUSD": eurusd_sl_stop,
        "GBPUSD": gbpusd_sl_stop,
    },
    axis=1
)

tp_stop = pd.concat(
    {
        "EURUSD": eurusd_tp_stop,
        "GBPUSD": gbpusd_tp_stop,
    },
    axis=1
)

pf = vbt.Portfolio.from_signals(
    close=close,
    entries=entries,
    exits=exits,
    sl_stop=sl_stop,
    tp_stop=tp_stop,
    size=5000,
    size_type="value",
    leverage=50,
    cash_sharing=True,
    group_by=True,
    delta_format="percent"
)
```

### Why this is the right pattern

From the support history:

- `from_signals` processes columns individually
- with `cash_sharing=True`, value/cash is handled at the group level
- `get_value_nb(c)` returns value for the **full group**
- context already contains row/column info (`c.i`, `c.col`)

So the intended model is: one aligned matrix, multiple columns, optionally grouped together.

### About dict input

You *can* build a multi-column object from dict-like pieces using `pd.concat({...}, axis=1)`, but based on the available context, I would not assume `from_signals` will auto-align separate dict inputs with different indexes for you. Align first.

### If you want pair + config together

You can go one step further and create MultiIndex columns such as:

- level 0 = pair
- level 1 = config_id

Then pass the full matrix into `from_signals`. Since `from_signals` works per column, each `(pair, config_id)` column becomes one simulated stream, while `cash_sharing=True` lets them share portfolio cash if grouped together.

---

## Suggested implementation summary

### Fixed notional / fixed margin trade sizing
Use:

```python
size=5000,
size_type="value",
leverage=50
```

### Parameter sweep of SL ATR multiplier
Use:

```python
sl_stop=vbt.Param(
    [mult * atr / close for mult in sl_mults],
    keys=pd.Index(sl_mults, name="sl_atr_mult")
)
```

### Multi-pair shared-cash portfolio
Build aligned multi-column DataFrames first, then:

```python
cash_sharing=True,
group_by=True
```

---

## Relevant links

- Dynamic `from_signals` docs:  
  https://vectorbt.pro/pvt_16ebf9ef/documentation/portfolio/from-signals/#dynamic

- Stop orders docs / parameterization examples:  
  https://vectorbt.pro/pvt_16ebf9ef/documentation/portfolio/from-signals/#stop-orders

- Cookbook reference repeatedly cited in support:  
  **Portfolio cookbook → callbacks**  
  https://vectorbt.pro/pvt_16ebf9ef/cookbook/portfolio/#callbacks

If you want, I can next turn this into one complete Round-8-ready snippet with:

- EURUSD + GBPUSD
- fixed `$100` margin / `$5000` notional
- `sl_atr_mult` sweep via `vbt.Param`
- `cash_sharing=True`
- MultiIndex columns as `(pair, config_id)`
