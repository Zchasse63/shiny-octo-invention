# STRATEGIES

Auto-generated from `fx-scalper/src/strategies/`. **Do not hand-edit this file** — add narrative to `fx-scalper/docs/strategies_manual.md` and re-run `python fx-scalper/scripts/render_strategies.py`.

Last rendered: `2026-04-21 19:36:20` UTC

## Strategies registered

## bb_rsi_mr  
**Class:** `src.strategies.bb_rsi_mr.BBRSIMeanReversion` — ✅ implemented

Bollinger + RSI mean reversion with ADX no-trend filter.

    Args:
        params: Strategy parameters.
        cash_per_trade_usd: Cash committed per trade (default: settings value).
        leverage: Leverage multiplier (default: settings value).

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `bb_length` | `int` | `20` | Bollinger Band window (CLAUDE.md sweep grid: 15/20/30). |
| `bb_std` | `float` | `2.0` | Bollinger Band std multiplier (1.8/2.0/2.2). |
| `rsi_length` | `int` | `14` | RSI window (10/14/21). |
| `rsi_long_threshold` | `float` | `30.0` | Long when RSI below this (25/30/35). |
| `rsi_short_threshold` | `float` | `70.0` | Short when RSI above this (65/70/75). |
| `adx_threshold` | `float` | `20.0` | Only trade when ADX below this (no-trend filter). |
| `adx_length` | `int` | `14` | ADX window. |
| `atr_length` | `int` | `14` | ATR window for SL distance. |
| `sl_atr_multiplier` | `float` | `1.5` | SL = entry ∓ k × ATR. |
| `tp_band` | `str` | `'opposite'` | ``"opposite"`` (exit at opposite BB) or ``"midline"``. |
| `asian_session_only` | `bool` | `True` | Restrict entries to 23:00–07:00 UTC window. |

### Notes (human-authored overlay)

**Academic basis.** Mean-reversion on high-frequency FX has the strongest
evidence of the three starter strategies in the literature. Bollinger
extremes combined with RSI give a robust noise filter; the ADX < 20
condition is the no-trend gate that keeps us out of momentum regimes where
MR blows up.

**Failure modes to watch in backtest:**
- **Trend periods.** NFP release, SNB-style interventions, or regime
  breaks can produce sustained BB breakouts. The ADX < 20 filter should
  handle most of these, but OOS drawdowns concentrated around news
  releases are a red flag.
- **Illiquid session boundaries.** 23:00–00:00 UTC Sunday (re-open) has
  wide spreads and stale quotes. Our weekend gate starts Sunday entries
  at `SUNDAY_OPEN_UTC_HOUR` (22:00 UTC) per RUNBOOK, but the first hour
  should be treated skeptically in OOS analysis.
- **Asian session carry.** If London traders revisit overnight levels,
  a MR entry at 06:00 UTC can get stopped out as London trends through.
  Our default `asian_session_only=True` avoids this, but the Day-4 sweep
  will test the `False` case.

**Spread-honest pricing.** SL/TP are anchored to `ask_close` (LONG) or
`bid_close` (SHORT) so backtest metrics match what the broker actually
fills. If the backtest uses mid anywhere for SL/TP, results are ~0.5 pip
optimistic per side, which compounds into a meaningful Sharpe inflation
on 5-minute bars.

**Day-4 sweep targets** (per CLAUDE.md):
- BB length ∈ {15, 20, 30}, std ∈ {1.8, 2.0, 2.2}
- RSI length ∈ {10, 14, 21}, thresholds ∈ {65/35, 70/30, 75/25}
- ADX threshold ∈ {18, 20, 22}
- Pairs: EUR/USD, USD/JPY
- Walk-forward: train 2023, test 2024–2025
- Candidate criteria: OOS post-cost Sharpe > 0.5, max DD < 15%

## session_breakout  
**Class:** `src.strategies.session_breakout.SessionBreakout` — 🚧 stub

Day 6 — not yet implemented.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `london_open_utc_hour` | `int` | `8` |  |
| `london_close_utc_hour` | `int` | `12` |  |
| `overlap_end_utc_hour` | `int` | `16` |  |
| `reward_risk_ratio` | `float` | `2.0` |  |

### Notes (human-authored overlay)

_Day-6 stub. Notes go here when implementation starts._

## trend_momentum  
**Class:** `src.strategies.trend_momentum.TrendMomentum` — 🚧 stub

Day 5 — not yet implemented. Raises :class:`NotImplementedError`.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ema_length` | `int` | `200` |  |
| `rsi_length` | `int` | `14` |  |
| `adx_length` | `int` | `14` |  |
| `adx_threshold` | `float` | `25.0` |  |
| `sl_atr_multiplier` | `float` | `1.5` |  |
| `tp_atr_multiplier` | `float` | `2.5` |  |
| `chandelier_atr_multiplier` | `float` | `3.0` |  |

### Notes (human-authored overlay)

_Day-5 stub. Notes go here when implementation starts._
