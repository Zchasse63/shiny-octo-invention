"""Round 8 — multi-pair shared-cash portfolio for EUR/USD + GBP/USD.

Per vbt.chat round-6 artifact, the canonical pattern for measuring
realized strategy correlation + combined DD is:

  1. Build aligned multi-column DataFrames via ``pd.concat({...}, axis=1)``
     with pair as the column key.
  2. Pass into ``vbt.Portfolio.from_signals(..., cash_sharing=True,
     group_by=True, size=5000, size_type='value', leverage=50)``.

The two columns represent the same strategy (the post-round-6 winning
config) run simultaneously on EUR/USD and GBP/USD, sharing a single
$500 starting cash. Output:

 - ``backtest_results/round8_portfolio_YYYYMMDD/portfolio_stats.json``
 - Combined equity curve parquet
 - Realized per-bar correlation of the two columns' returns
 - Inter-column drawdown overlap statistic

This is the last piece before Phase-3 shortlist: it tells us whether
running EUR+GBP together actually reduces portfolio DD vs running
each alone (diversification benefit).
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import vectorbtpro as vbt  # noqa: E402

from src.backtest.data_loader import load_symbol_bars  # noqa: E402
from src.backtest.resample import resample_bars  # noqa: E402
from src.indicators.engine import add_atr  # noqa: E402
from src.strategies.exits import ExitConfig, config_to_vbt_params  # noqa: E402
from src.strategies.families import get_family_by_name  # noqa: E402
from src.utils.logger import get_logger, init_logger  # noqa: E402


ROUND5_CSV = "backtest_results/explore_multi_tf_20260422T0026/combined_results.csv"
PAIRS = ["EUR_USD", "GBP_USD"]


def _top1_config(csv_path: Path) -> dict:
    """Pick the single best config that survived cross-pair validation.

    Round 4 re-run with fixed sizing showed that M15 bb_rsi_mr_filtered
    with session + weekday filter is what actually generalizes to GBP.
    We restrict to *_filtered families for this.
    """
    df = pd.read_csv(csv_path)
    oos = df[(df["kind"] == "OOS") & (df["total_trades"] >= 30)
             & df["family"].str.endswith("_filtered")
             & (df["timeframe"] == "15min")]
    agg = (
        oos.groupby(["timeframe", "family", "family_params", "exit_config"])
        .agg(
            pf=("profit_factor", "mean"),
            exp=("expectancy_usd", "mean"),
            n=("split", "count"),
            min_pf=("profit_factor", "min"),
        )
        .reset_index()
    )
    full = agg[(agg["n"] == 3) & (agg["min_pf"] >= 1.2)]
    return full.sort_values("pf", ascending=False).iloc[0].to_dict()


def _build_signals_for_pair(
    config: dict, pair: str
) -> dict:
    """Load bars, generate signals + exit params for one pair."""
    m1 = load_symbol_bars(pair, start="2023-01-01", end="2026-04-20")
    bars = resample_bars(m1, config["timeframe"])

    family_cls = get_family_by_name(config["family"])
    family_params = json.loads(config["family_params"])
    family = family_cls(family_cls.params_cls(**family_params))
    signals = family.generate(bars)

    close_col = "mid_close" if "mid_close" in bars.columns else "bid_close"
    close = bars[close_col].rename(pair)
    atr_len = int(json.loads(config["exit_config"]).get("atr_length", 14))
    atr = add_atr(bars, length=atr_len)[f"atr_{atr_len}"]

    exit_dict = json.loads(config["exit_config"])
    exit_cfg = ExitConfig(
        sl_atr_mult=exit_dict["sl_atr_mult"],
        atr_length=atr_len,
        tp_r_mult=exit_dict.get("tp_r_mult"),
        trail_kind=exit_dict["trail_kind"],
        trail_atr_mult=exit_dict.get("trail_atr_mult"),
    )
    vbt_params = config_to_vbt_params(
        entries_long=signals.entries_long,
        entries_short=signals.entries_short,
        close=close,
        atr=atr,
        config=exit_cfg,
    )

    # Slippage = half-spread per bar.
    if "bid_close" in bars.columns and "ask_close" in bars.columns:
        slip = ((bars["ask_close"] - bars["bid_close"]) / 2.0)
    else:
        slip = pd.Series(0.0, index=bars.index)

    return {
        "close": close,
        "entries_long": signals.entries_long.rename(pair),
        "entries_short": signals.entries_short.rename(pair),
        "sl_stop": vbt_params.sl_stop.fillna(0.0).rename(pair),
        "tp_stop": vbt_params.tp_stop.fillna(0.0).rename(pair),
        "tsl_stop": (
            vbt_params.trail_distance_pct.fillna(0.0).rename(pair)
            if vbt_params.trail_distance_pct is not None else None
        ),
        "slippage": slip.rename(pair),
        "use_trail": bool(vbt_params.sl_trail and vbt_params.trail_distance_pct is not None),
    }


def main() -> int:
    init_logger()
    logger = get_logger(__name__)

    cfg = _top1_config(PROJECT_ROOT / ROUND5_CSV)
    logger.info(f"Top config: {cfg['family']} {cfg['timeframe']} pf={cfg['pf']:.2f}")

    per_pair = {pair: _build_signals_for_pair(cfg, pair) for pair in PAIRS}
    for p, d in per_pair.items():
        logger.info(f"  {p}: {len(d['close']):,} bars, "
                    f"entries_long={int(d['entries_long'].sum())}, "
                    f"entries_short={int(d['entries_short'].sum())}")

    # Align columns. pd.concat with axis=1 unions the indices; missing
    # timestamps on one side become NaN. We fill numeric arrays with 0
    # (stop=0 means "no stop", entries=False means "no signal") and keep
    # bool arrays as False.
    close = pd.concat([per_pair[p]["close"] for p in PAIRS], axis=1).ffill()
    entries_long = pd.concat([per_pair[p]["entries_long"] for p in PAIRS], axis=1).fillna(False).astype(bool)
    entries_short = pd.concat([per_pair[p]["entries_short"] for p in PAIRS], axis=1).fillna(False).astype(bool)
    sl_stop = pd.concat([per_pair[p]["sl_stop"] for p in PAIRS], axis=1).fillna(0.0)
    tp_stop = pd.concat([per_pair[p]["tp_stop"] for p in PAIRS], axis=1).fillna(0.0)
    tsl_list = [per_pair[p]["tsl_stop"] for p in PAIRS]
    use_trail = all(per_pair[p]["use_trail"] for p in PAIRS)
    if use_trail and all(s is not None for s in tsl_list):
        tsl_stop = pd.concat(tsl_list, axis=1).fillna(0.0)
    else:
        tsl_stop = None
    slippage = pd.concat([per_pair[p]["slippage"] for p in PAIRS], axis=1).fillna(0.0)

    logger.info(f"Aligned matrix: {close.shape} (bars × pairs)")

    kwargs = dict(
        close=close,
        entries=entries_long,
        short_entries=entries_short,
        tp_stop=tp_stop.to_numpy(),
        size=5000.0,
        size_type="value",
        leverage=50,
        init_cash=500.0,
        cash_sharing=True,
        group_by=True,
        freq="15min",
        fees=0.0,
        slippage=slippage.to_numpy(),
    )
    if tsl_stop is not None:
        kwargs["tsl_stop"] = tsl_stop.to_numpy()
    else:
        kwargs["sl_stop"] = sl_stop.to_numpy()

    pf = vbt.Portfolio.from_signals(**kwargs)
    logger.info("Portfolio built with cash_sharing=True, group_by=True")

    # Per-column performance. Extract per-column arrays robustly.
    trades = pf.trades
    def _as_list(x, n):
        import numpy as np
        if hasattr(x, "values"):
            v = x.values
        elif hasattr(x, "to_numpy"):
            v = x.to_numpy()
        else:
            v = np.atleast_1d(x)
        v = list(v)
        if len(v) < n:
            v = v + [None] * (n - len(v))
        return v[:n]

    n = len(PAIRS)
    stats_per_col = pd.DataFrame({
        "pair": PAIRS,
        "n_trades": _as_list(trades.count(), n),
        "win_rate": _as_list(trades.win_rate, n),
        "expectancy_usd": _as_list(trades.pnl.mean(), n),
    })

    # Group-level stats (combined portfolio).
    group_stats = pf.stats(group_by=True)
    group_stats_json = {k: (float(v) if isinstance(v, int | float | np.floating | np.integer)
                            else str(v))
                        for k, v in group_stats.items()}

    # Per-bar return correlation (from per-column returns).
    returns_ungrouped = pf.get_returns(group_by=False)
    corr = returns_ungrouped.corr()
    logger.info(f"Per-bar return correlation:\n{corr.round(3)}")

    # Simultaneous drawdown: fraction of bars both pairs are in DD.
    ddseries = pf.get_drawdowns(group_by=False)
    # Binary in-DD flags per column.
    try:
        returns_per_col = returns_ungrouped
        eq = (1 + returns_per_col).cumprod()
        peak = eq.cummax()
        in_dd = (eq < peak).astype(int)
        joint_dd = (in_dd.iloc[:, 0] & in_dd.iloc[:, 1]).mean()
    except Exception:
        joint_dd = float("nan")

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M")
    out_dir = PROJECT_ROOT / "backtest_results" / f"round8_portfolio_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "per_col_stats.csv").write_text(stats_per_col.to_csv(index=False))
    (out_dir / "group_stats.json").write_text(json.dumps(group_stats_json, indent=2, default=str))
    corr.to_csv(out_dir / "return_correlation.csv")

    print()
    print("=" * 76)
    print("Round 8 — EUR+GBP shared-cash portfolio")
    print("=" * 76)
    print("Per-pair:")
    print(stats_per_col.round(3).to_string(index=False))
    print()
    print("Combined (cash_sharing=True):")
    keys = ["Total Trades", "Total Return [%]", "Max Drawdown [%]",
            "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
            "Win Rate [%]", "Profit Factor", "Expectancy"]
    for k in keys:
        if k in group_stats_json:
            v = group_stats_json[k]
            print(f"  {k:25s}: {v}")
    print()
    print("Per-bar return correlation:")
    print(corr.round(3).to_string())
    print()
    print(f"Joint-drawdown fraction (both in DD same bar): {joint_dd:.1%}")
    print()
    logger.info(f"Artifacts → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
