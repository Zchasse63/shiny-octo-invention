# External Knowledge Base

Shallow-clones of third-party libraries we depend on. Stored locally so we
have inherent offline knowledge and never need to consult a website during
development. Grep-able from anywhere in the project.

## Convention

**Before writing code that uses any of these libraries, grep this directory first.**
See [../../CONVENTIONS.md](../../../CONVENTIONS.md) for the full rule.

## Libraries

| Library | Upstream | Role | Size | Key dirs |
|---|---|---|---|---|
| [oandapyV20](oandapyV20/) | [hootnot/oanda-api-v20](https://github.com/hootnot/oanda-api-v20) | OANDA v20 REST SDK — order placement, account queries, streaming prices | 1 MB | `oandapyV20/endpoints/`, `samples/`, `README.rst` |
| [pandas-ta-classic](pandas-ta-classic/) | [xgboosted/pandas-ta-classic](https://github.com/xgboosted/pandas-ta-classic) | Primary indicator library (wraps TA-Lib) | 5.6 MB | `pandas_ta_classic/`, `examples/`, `README.md` |
| [ta-lib-python](ta-lib-python/) | [TA-Lib/ta-lib-python](https://github.com/TA-Lib/ta-lib-python) | Low-level TA-Lib bindings — silent speed booster under pandas-ta | 5.8 MB | `talib/`, `docs/`, `README.md` |
| [loguru](loguru/) | [Delgan/loguru](https://github.com/Delgan/loguru) | Logging library used by `src/utils/logger.py` | 2.3 MB | `loguru/`, `docs/`, `README.rst` |
| [nautilus_trader](nautilus_trader/) | [nautechsystems/nautilus_trader](https://github.com/nautechsystems/nautilus_trader) | Day-7 execution-realism validation harness | 18 MB | `docs/`, `examples/`, `nautilus_trader/`, `python/` |
| [duka](duka/) | [giuse88/duka](https://github.com/giuse88/duka) | Historical reference — the Dukascopy library we abandoned (see `DECISIONS/0001`) | 372 KB | `duka/`, `README.md` |
| [vectorbt-pro](vectorbt-pro/) | [polakowo/vectorbt.pro](https://github.com/polakowo/vectorbt.pro) (private) | Backtest harness + Knowledge / Intelligence RAG module + native MCP server. See `DECISIONS/0002` + `docs/research/vectorbtpro_capabilities.md` | 15 MB | `vectorbtpro/` (source), `vectorbtpro/knowledge/` (Intelligence), `vectorbtpro/portfolio/`, `vectorbtpro/mcp_server.py`, `tests/` |

## OANDA v20 REST API — notes

The oandapyV20 SDK wraps the v20 REST API one-to-one. For endpoint semantics:

- **Endpoint catalog** — [oandapyV20/oandapyV20/endpoints/](oandapyV20/oandapyV20/endpoints/) — one Python file per REST resource (`accounts.py`, `orders.py`, `trades.py`, `pricing.py`, `instruments.py`, `transactions.py`, `positions.py`).
- **Samples** — [oandapyV20/samples/](oandapyV20/samples/) and [oandapyV20/tests/](oandapyV20/tests/) — real request/response examples for every endpoint.
- **Live docs** — use the Context7 MCP tool (`plugin:context7:context7`) for current REST API reference when the local clone is stale.

## Refreshing this knowledge base

```bash
scripts/refresh_external_docs.sh
```

Re-clones each repo (shallow) and prunes nautilus_trader's test/crate dirs.
Safe to run at any time — the only loss is local edits you shouldn't be
making anyway (this directory is *read-only reference*, do not modify).

## Policy

- **Read-only.** Never edit anything under `docs/external/`. If you find a bug
  in one of these libraries, open a PR upstream and add a note to
  `DECISIONS/` explaining the workaround we ship.
- **Commit to our repo.** Yes, 33 MB is bulky, but we pay it once for
  guaranteed offline availability.
- **Do not add new clones silently.** If a new library is pulled in, add it
  to this table AND to the refresh script.
