# RUNBOOK

Partially auto-generated. The **config snapshot** below is rendered from `fx-scalper/config/settings.py`. The **operational procedures** section is merged from `fx-scalper/docs/runbook_manual.md`.

Last rendered: `2026-04-21 19:36:20` UTC

## Config snapshot

_Derived from `config/settings.py`. All risk params + circuit-breaker values + session windows._

- `ACCOUNT_CURRENCY` = `'USD'`
- `ACCOUNT_FLOOR_USD` = `400.0`
- `ACCOUNT_STARTING_BALANCE_USD` = `500.0`
- `ASIAN_SESSION_END_UTC_HOUR` = `7`
- `ASIAN_SESSION_START_UTC_HOUR` = `23`
- `BACKTEST_START_DATE` = `'2023-01-01'`
- `BACKTEST_TEST_START` = `'2024-01-01'`
- `BACKTEST_TRAIN_END` = `'2023-12-31'`
- `CASH_PER_TRADE_USD` = `100.0`
- `CONSECUTIVE_LOSS_PAUSE_MINUTES` = `60`
- `DAILY_LOSS_LIMIT_USD` = `50.0`
- `DATA_PROCESSED_DIR` = `'data/processed'`
- `DATA_RAW_DIR` = `'data/raw'`
- `FRIDAY_CLOSE_UTC_HOUR` = `21`
- `FRIDAY_FLAT_BY_UTC_HOUR` = `20`
- `INSTRUMENTS` = `('EUR_USD', 'GBP_USD', 'USD_JPY')`
- `LONDON_CLOSE_UTC_HOUR` = `12`
- `LONDON_OPEN_UTC_HOUR` = `8`
- `MAGIC_NUMBER` = `'FXSCALPER-V1'`
- `MAX_CONCURRENT_POSITIONS` = `2`
- `MAX_CONSECUTIVE_LOSSES` = `3`
- `MAX_LEVERAGE` = `50`
- `NY_CLOSE_UTC_HOUR` = `16`
- `NY_OPEN_UTC_HOUR` = `12`
- `OANDA_BACKOFF_SECONDS` = `0.5`
- `OANDA_CONSECUTIVE_FAILURE_LIMIT` = `3`
- `OANDA_LIVE_HOSTNAME` = `'api-fxtrade.oanda.com'`
- `OANDA_MAX_RETRIES` = `3`
- `OANDA_PRACTICE_HOSTNAME` = `'api-fxpractice.oanda.com'`
- `OANDA_RATE_LIMIT_RPS` = `120`
- `OANDA_REQUEST_TIMEOUT_SECONDS` = `10.0`
- `OANDA_STREAM_LIVE_HOSTNAME` = `'stream-fxtrade.oanda.com'`
- `OANDA_STREAM_PRACTICE_HOSTNAME` = `'stream-fxpractice.oanda.com'`
- `SINGLE_TRADE_MAX_LOSS_USD` = `30.0`
- `SUNDAY_OPEN_UTC_HOUR` = `22`

## Operational procedures

# Operational procedures

_Human-authored. Merged into `RUNBOOK.md` below the auto-generated config snapshot._

## Smoke-test checklist (pre-paper)

1. `.env` present and valid (API key, account id, environment=`practice`).
2. `fx-scalper/.venv/bin/python fx-scalper/scripts/smoke_oanda.py`
   — expect "Smoke test OK." and non-zero balance.
3. `fx-scalper/.venv/bin/pytest -q` — all tests pass.
4. `fx-scalper/docs/external/INDEX.md` reviewed; `docs/external/` cloned
   (run `scripts/refresh_external_docs.sh` if stale).

## Pre-paper checklist (Week 2)

1. A winning strategy has cleared the Day 7 NautilusTrader gate
   (vbt→Nautilus Sharpe drop <30%).
2. `OANDA_ENVIRONMENT=practice` in `.env`.
3. Journal DB (`fx-scalper/journal.db`) is fresh — delete if carrying
   over contaminated state.
4. Risk params in `config/settings.py` match intended sizing: verify with
   the config snapshot above.

## Pre-live checklist (Week 3+)

1. ≥14 days of paper trading results track backtest OOS within 1σ for
   Sharpe, profit factor, and win rate. Document the comparison under a
   fresh ADR before flipping.
2. Account funded at exactly `$500.00` USD. NAV = $500 at start.
3. `OANDA_ENVIRONMENT=live`.
4. Tag the release: `git tag v1.0-first-live-trade`.
5. First live day supervised in-person. Do not start `run_live.py` from
   an unattended VPS on day one.

## Incident response

1. **Stop the bleeding.** Force-stop the bot: `Ctrl-C`, or `kill -TERM <pid>`.
2. **Do not flatten positions automatically unless asked.** CLAUDE.md §4
   says single-trade blowout → halt bot, alert, investigate. Manual close
   via OANDA web dashboard if needed.
3. **Snapshot state.** Copy `fx-scalper/journal.db` and `fx-scalper/logs/events.jsonl`
   to a dated folder.
4. **Log an `incident` event** via `src.utils.diary.log_event(...)` with
   severity, summary, and (once known) resolution.
5. **Write an ADR** if the incident warrants a design change.

## Recovery procedures

- **OANDA API down:** bot auto-halts new entries after N consecutive
  failures (see `OANDA_CONSECUTIVE_FAILURE_LIMIT` in config snapshot).
  Wait for OANDA status page to recover, then restart.
- **Account floor breached:** bot exits with `EMERGENCY_SHUTDOWN`.
  Investigate, refund if appropriate, do NOT restart until approved.
- **Daily loss limit hit:** auto-halt until next UTC midnight. Re-enable
  by restarting the bot after the window expires.

## Keeping this file relevant

Procedures here are operational. Strategy-level "why" belongs in ADRs or
`docs/strategies_manual.md`. Config values don't belong here — they're
already in the auto-generated snapshot above.
