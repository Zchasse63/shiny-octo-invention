# ADR 0001: In-house Dukascopy tick downloader

Date: 2026-04-21
Status: accepted

## Context

CLAUDE.md §Historical Data designates Dukascopy tick data (2003-present,
institutional ECN bid/ask) as the primary historical source for backtests,
pulled via the `duka` Python library (`pip install duka`). Per the 7-day
plan, Day 2 wires up the pull and Day 3 feeds it into the backtest harness.

During Day 2 implementation we discovered two problems with `duka==0.2.0`:

1. **URL rot.** `duka` ships a hardcoded base URL at
   `https://www.dukascopy.com/datafeed/{...}`. Dukascopy now 301-redirects
   that host to `https://datafeed.dukascopy.com/datafeed/{...}`. `requests`
   follows redirects by default, so that alone wouldn't break it — but…

2. **asyncio pipeline drops entire days on single-hour failures.**
   `duka.core.fetch.fetch_day` submits 24 hourly fetch tasks via
   `asyncio.wait(tasks)`, then collects results with
   `reduce(add, tasks, BytesIO())` where `add` calls `task.result()`. If
   **any** hour fails (rate limit, transient 5xx, weekend-closed hour),
   `task.result()` raises, caught by the outer `try` in `app.do_work`,
   and the ENTIRE day's data is discarded. In testing this caused every
   pulled day to produce a zero-row CSV — no tick data, no way forward.

Reproduced by running `scripts/pull_dukascopy.py --start 2024-01-02
--end 2024-01-05 --instruments EUR_USD`: output CSV was 36 bytes
(header only) despite the URL being reachable via plain `curl`.

## Decision

Write a minimal synchronous Dukascopy downloader in-tree at
`src/backtest/dukascopy_client.py` (~150 LOC). Use plain `requests.get`
per hour, handle missing hours silently (weekend boundaries legitimately
return 200 OK with 0 bytes), decompress the bi5 payload with the stdlib
`lzma` module, parse the 20-byte record format into pandas DataFrames.
Abandon `duka` for the primary path.

Keep `duka` in `fx-scalper/docs/external/duka/` as historical reference.

## Consequences

**Gained:**
- Reliable per-hour fetches. Friday 21:00 UTC returning empty doesn't
  destroy Friday's earlier 17 hours.
- Clear error surface: `fetch_day_ticks` logs missing hours, returns
  whatever it did get.
- No asyncio/executor/stream surprises.
- ~150 LOC we fully understand vs a dependency that silently drops data.

**Given up:**
- Parallelism (duka used 20 threads). Our version is sequential, 24
  requests per day, ~15s per day of EUR/USD. For a 2-year backfill
  (~520 weekdays × 3 pairs = 1,560 day-fetches × 15s ≈ 6.5 hours of
  wall time). Acceptable — runs once, then data is local Parquet.
- If Dukascopy's URL or bi5 format changes, we fix it ourselves. Diff
  surface is small.

**Follow-up work:**
- If backfill runtime becomes a problem, add `ThreadPoolExecutor` per-day
  parallelism in our client while keeping the robust per-hour error
  handling.
- If bi5 record layout changes, the `struct.unpack('!IIIff', ...)` is the
  one line to update.

## Alternatives considered

1. **Patch duka upstream** — too slow; issue would need a release, the
   maintainer hasn't merged in months, and we'd still be one dependency
   release away from the next break.
2. **Monkey-patch `duka.core.fetch.URL` at runtime.** Tested — fixes the
   URL but does nothing for the asyncio-task-drops-day bug. Partial fix
   is worse than a rewrite.
3. **Fetch tick CSVs from TrueFX or another free provider.** Lower
   quality than Dukascopy (retail-skewed spreads), contradicts CLAUDE.md.
4. **Pay for TickVault / QuantCheckout** — premature optimization at $50+/mo.
5. **Use OANDA historical candles** — explicitly rejected by CLAUDE.md for
   backtests; reserved for final broker-feed sanity check.
