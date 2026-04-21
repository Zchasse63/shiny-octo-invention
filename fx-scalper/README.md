# fx-scalper

Production-grade Python forex scalping bot trading EUR/USD, GBP/USD, and USD/JPY on 30s–5m timeframes via OANDA.

Architecture, risk parameters, and implementation plan are defined in [../CLAUDE.md](../CLAUDE.md) — treat that file as the North Star.

## Quick start (macOS dev)

```bash
# 1. System deps
brew install python@3.11 ta-lib

# 2. Project venv
cd fx-scalper
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. Secrets
cp .env.example .env
# edit .env — fill OANDA_API_KEY, OANDA_ACCOUNT_ID, OANDA_ENVIRONMENT

# 4. Smoke test OANDA connection
python scripts/smoke_oanda.py
```

## Layout

```
fx-scalper/
├── config/           # settings + .env loader
├── data/             # raw Dukascopy + processed Parquet (.gitignored)
├── src/
│   ├── oanda/        # client, data, orders, account, instruments
│   ├── indicators/   # pandas-ta-classic wrapper
│   ├── strategies/   # Strategy base + the three starter strategies
│   ├── backtest/     # vectorbt Pro harness + metrics
│   ├── live/         # polling bot, trailing stop, risk guards, sizing
│   └── utils/        # logger + SQLite journal
├── scripts/          # entry points: pull_dukascopy, run_backtest, run_paper, run_live
└── tests/            # unit + mocked OANDA + integration (practice account)
```

## Testing

```bash
pytest                           # unit tests only
pytest -m integration            # integration tests (hits OANDA practice)
```

## Non-negotiable rules

See CLAUDE.md §"Code Standards" and §"Circuit Breakers". Summary:

- Never compute signals on the forming bar. Always `iloc[-2]` (last closed).
- Every order carries a magic number + strategy name + trade UUID in `clientExtensions.id`.
- Every request/response logged to SQLite (`journal.db`).
- All timestamps tz-aware UTC.
- Circuit breakers override strategy — checked every poll cycle.

## Pipeline status

See [CLAUDE.md §Seven-Day Implementation Plan](../CLAUDE.md). Each day stops for review before the next begins.
