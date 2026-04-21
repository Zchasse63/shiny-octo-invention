# FX Scalper — Project Kickoff for Claude Code

## Mission

Build a production-grade Python forex scalping bot that trades EUR/USD, GBP/USD, and USD/JPY on 30-second to 5-minute timeframes through OANDA, with profit-driven exits via dynamic trailing stops. Trades are NOT closed by time — only by price action against the trail or by hitting stop loss. Start tiny, prove the system doesn't self-destruct, then scale.

**Account starting parameters (hardcoded, do not change without asking):**
- Starting capital: **$500 USD**
- Cash committed per trade: **$100 margin**
- Leverage: **50:1** (OANDA US regulatory maximum)
- Resulting position size per trade: **$5,000 notional** (~0.05 standard lots, ~$0.50/pip)
- Broker: **OANDA** (CFTC/NFA regulated, US entity)
- Dev platform: **macOS** (Mac-native, no Windows VPS required for dev)
- Production runtime: **Linux VPS** (cheap, stable, 24/7)

**Non-negotiable principles:**
1. Honest backtests over optimistic ones. Always model real spreads, real commission, real slippage.
2. Walk-forward validation required before any live deployment.
3. Circuit breakers have absolute priority — they halt the bot regardless of what the strategy says.
4. Magic number discipline on every order. Log every request and response to SQLite.
5. Never compute signals on the forming bar. Use the last CLOSED bar only.

---

## Pre-Decided Architecture (Do NOT Revisit)

These decisions are final. Do not suggest alternatives unless directly asked.

### Execution Venue
- **OANDA v20 REST + streaming API** via the official `oandapyV20` Python package
- Account type: **Standard** (no commission, ~0.6 pip spread on EUR/USD)
- US entity: `api-fxtrade.oanda.com` (live) / `api-fxpractice.oanda.com` (paper)

### Core Libraries

```bash
# OANDA SDK
pip install oandapyV20

# Data + indicators
pip install pandas numpy scipy
pip install pandas-ta-classic==0.4.47
pip install TA-Lib==0.6.8

# Backtesting (paid, installed separately via private GitHub)
pip install -U "vectorbtpro[base] @ git+https://github.com/polakowo/vectorbt.pro.git"

# Execution-realism validation
pip install nautilus_trader

# Dev tools
pip install python-dotenv loguru pydantic
pip install pytest pytest-asyncio pytest-mock
pip install ruff mypy
```

### Library Roles

- **pandas-ta-classic**: Primary indicator API. We write `df.ta.rsi()`, `df.ta.bbands()`, etc.
- **TA-Lib**: Silent speed booster under pandas-ta-classic. 50–300× faster indicator compute. No direct code calls.
- **vectorbt Pro**: Backtesting, parameter sweeps, walk-forward validation. Native SL/TP/trailing stop support + leverage modeling — required for realistic forex backtests.
- **NautilusTrader**: Final execution-realism pass before paper trading. Proper L1 FillModel and FX margin accounting.
- **oandapyV20**: Live + paper execution. REST for orders and account info, streaming WebSocket for prices.

### Historical Data
- **Primary:** Dukascopy tick data (free, 2003–present, institutional ECN bid/ask)
- **Pull with:** `duka` library (`pip install duka`) — simpler Mac-native option than TickVault
- **Period:** 2023-01-01 to today for EUR/USD, GBP/USD, USD/JPY
- **Storage:** Partitioned Parquet at `data/processed/{symbol}/year={YYYY}/month={MM}/bars.parquet`
- **NOT Massive API** — spreads are too tight compared to retail reality
- **NOT HistData** — bid-only, can't model spread
- **OANDA historical candles** only for final broker-feed sanity check (via `/v3/instruments/{instrument}/candles`)

### Starter Strategies

All three have academic or rigorous backtest support. Implement in order.

**Strategy 1: Bollinger-Band + RSI Mean Reversion** (strongest evidence)
- Pairs: EUR/USD, USD/JPY
- Timeframe: M5 or M15
- Session: Asian (23:00–07:00 UTC) preferred
- Signal:
  - Long: close < lower BB(20, 2.0) AND RSI(14) < 30 AND ADX(14) < 20
  - Short: close > upper BB(20, 2.0) AND RSI(14) > 70 AND ADX(14) < 20
- Exit: BB midline, opposite band, RSI back through 50, or ATR-based TP
- SL: 1.5× ATR(14)
- Trail: 2× ATR, tighten to 0.5× ATR if RSI reverses through 50

**Strategy 2: Trend-Filtered M15 Momentum**
- Pair: GBP/USD primary
- Filter: price vs EMA200 on H1
- Signal: RSI(14) crosses 50 in trend direction, ADX(14) > 25
- SL: 1.5× ATR, TP: 2.5× ATR
- Trail: Chandelier exit (highest_since_entry − 3× ATR for longs)

**Strategy 3: London–NY Overlap Range Breakout**
- Pair: GBP/USD primary, EUR/USD secondary
- Mark London session range: 08:00–12:00 UTC
- Trade breakout during overlap: 12:00–16:00 UTC
- Fixed 2:1 reward:risk, hard time exit at 16:00 UTC

---

## Project Structure

Build exactly this. Do not deviate without asking.

```
fx-scalper/
├── .env.example
├── .gitignore
├── README.md
├── pyproject.toml
├── requirements.txt
├── config/
│   ├── __init__.py
│   ├── settings.py           # all config, risk params, circuit breakers
│   └── secrets.py            # loads from .env
├── data/                     # .gitignored
│   ├── raw/                  # Dukascopy dumps
│   └── processed/            # Parquet M1 bid/ask
├── src/
│   ├── __init__.py
│   ├── oanda/
│   │   ├── __init__.py
│   │   ├── client.py         # authenticated oandapyV20 client factory
│   │   ├── data.py           # candle fetchers, streaming prices
│   │   ├── orders.py         # market, limit, SL/TP modify
│   │   ├── account.py        # balance, margin, open positions
│   │   └── instruments.py    # symbol metadata, pip value, precision
│   ├── indicators/
│   │   ├── __init__.py
│   │   └── engine.py         # pandas-ta-classic wrapper, consistent column names
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base.py           # abstract Strategy class
│   │   ├── bb_rsi_mr.py      # Strategy 1
│   │   ├── trend_momentum.py # Strategy 2
│   │   └── session_breakout.py
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── data_loader.py    # Dukascopy → pandas DataFrame
│   │   ├── harness.py        # vectorbt Pro wrapper
│   │   └── metrics.py        # Sharpe, max DD, profit factor
│   ├── live/
│   │   ├── __init__.py
│   │   ├── bot.py            # main polling loop
│   │   ├── trailing.py       # dynamic trailing stop logic
│   │   ├── risk.py           # circuit breakers, position sizing
│   │   └── sizing.py         # $100-per-trade → units conversion
│   └── utils/
│       ├── __init__.py
│       ├── logger.py         # loguru config
│       └── journal.py        # SQLite trade log
├── scripts/
│   ├── pull_dukascopy.py
│   ├── run_backtest.py
│   ├── run_paper.py          # paper trading entry point
│   └── run_live.py           # live trading entry point
├── notebooks/
│   └── research/
└── tests/
    ├── test_indicators.py
    ├── test_strategies.py
    ├── test_oanda_mock.py
    ├── test_risk.py          # circuit breaker tests
    └── test_sizing.py
```

---

## Position Sizing Logic (Critical — Code This Exactly)

The user commits **$100 cash per trade**. Leverage is 50:1. Position size is always derived from committed cash, not from risk-percentage formulas.

**Formula:**
```python
def compute_position_units(
    cash_committed_usd: float,  # $100
    leverage: int,              # 50
    current_price: float,       # e.g. 1.0800 for EUR/USD
    instrument: str,            # "EUR_USD"
    account_currency: str = "USD",
) -> int:
    """Returns integer units (OANDA uses units, not lots)."""
    notional_usd = cash_committed_usd * leverage  # $5,000
    
    base_ccy = instrument.split("_")[0]  # "EUR"
    if base_ccy == account_currency:
        units = notional_usd
    else:
        units = notional_usd / current_price  # convert to base ccy
    
    return int(units)
```

**Expected values on EUR/USD at 1.0800:** `notional = $5,000`, `units = 4,630` (~0.046 lots, ~$0.46/pip).

**Hardcoded config values in `config/settings.py`:**
```python
ACCOUNT_STARTING_BALANCE_USD = 500
CASH_PER_TRADE_USD = 100
MAX_LEVERAGE = 50
MAX_CONCURRENT_POSITIONS = 2       # never more than $200 at risk in margin at once
DAILY_LOSS_LIMIT_USD = 50          # bot halts for rest of UTC day
ACCOUNT_FLOOR_USD = 400            # bot halts and alerts
MAX_CONSECUTIVE_LOSSES = 3         # pause for 1 hour
SINGLE_TRADE_MAX_LOSS_USD = 30     # bot halts, something is wrong
```

---

## Circuit Breakers (Non-Negotiable)

Must be checked on EVERY poll cycle, not just at entry. These override all strategy logic.

1. **Account floor:** If account NAV drops below `$400`, halt bot, close all positions, send alert.
2. **Daily loss limit:** If realized + unrealized PnL for current UTC day ≤ `-$50`, halt until 00:00 UTC next day.
3. **Consecutive losses:** If last 3 trades closed at loss, pause new entries for 1 hour. Trailing on existing positions continues.
4. **Single-trade blowout:** If any open position shows unrealized loss > `$30`, halt bot, alert, do NOT auto-close (investigate first).
5. **OANDA disconnect:** If API fails 3 consecutive polls, halt new entries, continue trying to manage open positions, alert.
6. **Sunday/Friday boundary:** No new entries 17:00 ET Friday → 17:00 ET Sunday. Close all positions by 16:55 ET Friday.

Implement in `src/live/risk.py` as a `RiskGuard` class. The main loop asks `risk_guard.check()` before every entry AND every poll cycle. Returns enum: `OK`, `HALT_NEW_ENTRIES`, `HALT_ALL`, `EMERGENCY_SHUTDOWN`.

---

## Seven-Day Implementation Plan

### Day 1: Environment & OANDA Connection
- Initialize git repo (local + private GitHub remote)
- Create `fx-scalper/` structure per layout above
- Set up `pyproject.toml` + pin all versions in `requirements.txt`
- Implement `config/settings.py` with all hardcoded values from this doc
- Implement `config/secrets.py` loading from `.env`:
  - `OANDA_API_KEY`
  - `OANDA_ACCOUNT_ID`
  - `OANDA_ENVIRONMENT` = `practice` or `live`
- Implement `src/oanda/client.py`:
  - Factory function returning authenticated `oandapyV20.API` instance
  - Environment-aware (practice vs live)
  - Handles auth header, request retries, rate limit backoff
- Implement `src/oanda/account.py`:
  - `get_balance()`, `get_nav()`, `get_margin_used()`, `get_margin_available()`
  - `get_open_positions()`, `get_open_trades()`
- Implement `src/oanda/instruments.py`:
  - Fetch and cache instrument metadata (pip location, min trade size, precision)
  - `pip_value_usd(instrument, units, current_price)` helper
- **Smoke test:** Connect to practice account, print balance, open positions, instruments list.

### Day 2: Data & Indicators
- Implement `src/oanda/data.py`:
  - `get_candles(instrument, granularity, count, price="BA")` → pandas DataFrame with bid/ask OHLCV
  - Proper UTC timestamp handling
  - Price type `BA` = bid + ask (for spread modeling)
- Implement `scripts/pull_dukascopy.py` using `duka`:
  - Pull EUR/USD, GBP/USD, USD/JPY tick data 2023-01-01 → today
  - Resample tick → M1 bid/ask bars
  - Save as Parquet partitioned by symbol/year/month
- Implement `src/indicators/engine.py`:
  - Thin wrapper over pandas-ta-classic with consistent column naming
  - Include: `rsi`, `bbands`, `atr`, `adx`, `ema`, `macd`, `stoch`, `supertrend`
  - All functions take a DataFrame, return DataFrame with new columns
- **Validation test:** Verify TA-Lib is being used under the hood (check `pandas_ta_classic.Imports`). Confirm `talib.RSI` output matches `df.ta.rsi()` output bit-for-bit.

### Day 3: Backtest Harness
- Implement `src/backtest/data_loader.py`:
  - Load Parquet → single DataFrame per symbol
  - Helper to slice by date range
- Implement `src/backtest/harness.py` using vectorbt Pro:
  - Takes close prices DataFrame + bid/ask spreads DataFrame
  - `Portfolio.from_signals()` with native `sl_stop`, `tp_stop`, `sl_trail`
  - `leverage=50`, `freq='1min'`
  - Time-varying slippage from spread data
  - Session awareness: exclude Fri 17:00 ET → Sun 17:00 ET
- Implement `src/backtest/metrics.py`:
  - Post-cost Sharpe, Sortino, Calmar
  - Profit factor, win rate, expectancy
  - Max drawdown, avg drawdown duration
  - Trade-level statistics
- **Validation test:** Buy-and-hold baseline. Each "trade" should lose ~1 pip (the spread). If not, harness is broken.

### Day 4: Strategy 1 (Bollinger + RSI Mean Reversion)
- Implement `src/strategies/base.py` abstract class
- Implement `src/strategies/bb_rsi_mr.py` per spec above
- Parameter sweep grid:
  - BB length: {15, 20, 30}
  - BB std: {1.8, 2.0, 2.2}
  - RSI length: {10, 14, 21}
  - RSI threshold: {65/35, 70/30, 75/25}
  - ADX threshold: {18, 20, 22}
- 243 combos × 3 pairs = 729 backtests
- Walk-forward: train 2023, test 2024–2025
- Candidate criteria: OOS post-cost Sharpe > 0.5, max DD < 15%
- Output: CSV of top 20 parameter sets, equity curves, drawdown charts

### Day 5: Strategy 2 (Trend-Filtered Momentum)
- Implement `src/strategies/trend_momentum.py` per spec
- Same harness, same sweep structure
- **Red-flag check:** If Strategy 2 OOS beats Strategy 1 OOS, flag overfitting — literature says simple momentum on minutes is dead. Investigate before trusting.

### Day 6: Strategy 3 (Session Breakout)
- Implement `src/strategies/session_breakout.py`
- Test three variants: London-only, NY-only, London-NY overlap
- Expect naive London-open on EUR/USD to lose money (calibration test)
- Single-asset backtests acceptable

### Day 7: NautilusTrader Validation + Go/No-Go
- Install NautilusTrader
- Port best strategy from Day 4–6 to Nautilus strategy class
- Run with L1 FillModel on same OOS period
- **Deployment decision rule:**
  - vbt-Pro-to-Nautilus Sharpe drop <30% → proceed to OANDA practice account
  - Drop 30–50% → strategy fragile, iterate before deploying
  - Drop >50% → kill, vbt result was fill-model artifact
- **If no strategy passes:** Do NOT fund live account. Return to research mode.

### Week 2+: Paper Trading then Live
- Port winning strategy to `src/live/bot.py`
- Run on OANDA practice account for minimum 14 days
- Compare paper results to backtest OOS Sharpe/PF/win-rate within 1σ
- If paper tracks backtest, switch `.env` to live, fund with $500, run
- Do NOT add capital until 30+ days of live results track paper results

---

## Code Standards

- **Type hints everywhere.** `from __future__ import annotations` at the top of every module.
- **Google-style docstrings** on all public functions and classes.
- **No `print()` in `src/`** — use `loguru` logger via `src/utils/logger.py`.
- **No magic numbers in strategy files.** All config in `config/settings.py`.
- **Secrets in `.env` only.** Never commit. `.env.example` shows structure without values.
- **All timestamps are tz-aware UTC.** On ingress, localize or verify. Never compare naive to aware.
- **Never compute signals on the forming bar.** Use `iloc[-2]` (last CLOSED bar).
- **Every order carries `clientExtensions.id` with our magic number + strategy name + trade UUID.**
- **Check every API response for error codes.** OANDA returns rich error structures — parse them, log them, don't swallow.
- **Log every order request AND response to SQLite journal** (`src/utils/journal.py`). Audit trail is mandatory.
- **Tests that aren't flaky.** Mock OANDA responses for unit tests. Integration tests hit practice account only.
- **Pydantic models** for all data crossing API boundaries (OANDA responses, strategy configs, order requests).

---

## OANDA-Specific Gotchas to Code Around

Copy into module comments as reminders:

1. **Instrument naming is underscore-separated.** `EUR_USD`, `GBP_USD`, `USD_JPY` — not `EURUSD`.
2. **Units, not lots.** OANDA position size is in base currency units. 100,000 units = 1 standard lot. Our `$5,000 notional` on EUR/USD at 1.08 = ~4,630 units.
3. **Pip location varies.** EUR/USD pip = 0.0001 (location -4). USD/JPY pip = 0.01 (location -2). Fetch from instrument metadata, don't hardcode.
4. **Precision on stops/limits.** Prices must be submitted at instrument's `displayPrecision`. `round(price, 5)` for EUR/USD, `round(price, 3)` for USD/JPY.
5. **Streaming API is separate from REST.** `oandapyV20.endpoints.pricing.PricingStream` for live prices; REST for orders.
6. **Rate limits: 120 requests/second per account** (generous, but batch where possible).
7. **Trailing stops are server-side on OANDA.** `trailingStopLossOnFill` in order requests. We can also manage client-side for finer control — decide per strategy.
8. **Account margin rates differ by currency pair.** Fetch current rates, don't assume 2% across the board.
9. **FIFO enforcement in US.** Can't partially close positions in arbitrary order. Use full-close + re-enter pattern if needed.
10. **No hedging in US accounts.** Opposing positions in same instrument net out. Bot logic must track net position, not gross.
11. **Weekend gap risk.** Positions held over weekend can gap through stops. Circuit breaker #6 mandates flat-by-Friday.
12. **Practice account has unlimited resets** — use it aggressively during development.

---

## Environment Setup Checklist

Before Day 1 begins, the user must have:

- [ ] OANDA live account opened and funded with $500
- [ ] OANDA practice account created (free, separate from live)
- [ ] API token generated from OANDA web dashboard → Manage API Access
- [ ] Account ID noted (format: `001-001-XXXXXXX-001`)
- [ ] Private GitHub repo created (can be empty, we'll push to it Day 1)
- [ ] Python 3.11+ installed on Mac (verify: `python3 --version`)
- [ ] Linux VPS provisioned (Contabo $9/mo, Vultr $6/mo, or equivalent) — can wait until Week 2
- [ ] vectorbt Pro purchased ($20/mo) and GitHub account submitted to Oleg
- [ ] GitHub access to vectorbt Pro repo confirmed

---

## First Command to Claude Code

Once this file is in place, the first instruction to Claude Code should be:

> "Read CLAUDE.md. Build Day 1 scope exactly as specified. Stop after Day 1 smoke test and report results. Do not proceed to Day 2 until I say so."

Each day completes and stops for review. No leapfrogging. No scope creep.

---

## Reference Documents in Project Root

- `CLAUDE.md` — this file, architectural North Star
- `STRATEGIES.md` — detailed signal logic for all three strategies (create Day 4)
- `RUNBOOK.md` — operational procedures, alerts, recovery steps (create Week 2)
- `JOURNAL.md` — running log of decisions, changes, learnings (update continuously)
