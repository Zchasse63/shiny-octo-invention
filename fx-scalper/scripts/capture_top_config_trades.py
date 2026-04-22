"""Round 7 — capture per-trade records + MAE/MFE for round-5 top-N configs.

Reads ``backtest_results/explore_multi_tf_20260422T0026/combined_results.csv``
(the round-5 raw output), picks the top-N OOS configurations by mean
profit factor across the 3 walk-forward splits, then re-runs each on the
full EUR/USD history at the matching timeframe and saves the
``pf.trades.records_readable`` plus per-trade MAE / MFE to parquet.

Outputs land under ``backtest_results/trade_records_YYYYMMDDTHHMM/``:
 - ``top<N>_summary.csv`` with config + MAE/MFE summary stats
 - ``top<N>_trades_<rank>.parquet`` with full trade records for each config

This is the round-7 deliverable that the ROUND_CHECKLIST flagged as
mandatory (per-trade records captured for top configs).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.backtest.data_loader import load_symbol_bars  # noqa: E402
from src.backtest.explorer import capture_trade_records  # noqa: E402
from src.backtest.resample import resample_bars  # noqa: E402
from src.strategies.exits import ExitConfig  # noqa: E402
from src.strategies.families import get_family_by_name  # noqa: E402
from src.utils.logger import get_logger, init_logger  # noqa: E402


TIMEFRAME_TO_RESAMPLE = {
    "5min": "5min", "M5": "5min",
    "15min": "15min", "M15": "15min",
    "30min": "30min", "M30": "30min",
    "1H": "1H", "H1": "1H",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capture trade records for top-N configs")
    p.add_argument(
        "--results-csv",
        default="backtest_results/explore_multi_tf_20260422T0026/combined_results.csv",
        help="Path to round-N combined_results.csv",
    )
    p.add_argument("--top", type=int, default=10, help="Top-N configs to capture")
    p.add_argument("--symbol", default="EUR_USD")
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2026-04-20")
    p.add_argument("--min-trades", type=int, default=30)
    return p.parse_args()


def _aggregate_top(df: pd.DataFrame, top: int, min_trades: int) -> pd.DataFrame:
    """Collapse IS+OOS × 3 splits into per-config aggregates, rank OOS."""
    oos = df[df["kind"] == "OOS"].copy()
    # Group by the (family, family_params, exit_config, timeframe) identity.
    keys = ["family", "family_params", "exit_config", "timeframe"]
    agg = (
        oos.groupby(keys)
        .agg(
            n_splits=("profit_factor", "size"),
            mean_pf=("profit_factor", "mean"),
            mean_expectancy=("expectancy_usd", "mean"),
            mean_win_rate=("win_rate", "mean"),
            mean_max_dd=("max_drawdown_pct", "mean"),
            mean_trades=("total_trades", "mean"),
            min_pf=("profit_factor", "min"),
            min_expectancy=("expectancy_usd", "min"),
        )
        .reset_index()
    )
    # Require all 3 splits + min trades + PF>1.2 floor.
    agg = agg[
        (agg["n_splits"] >= 3)
        & (agg["mean_trades"] >= min_trades)
        & (agg["min_pf"] >= 1.2)
        & (agg["min_expectancy"] > 0)
    ]
    agg = agg.sort_values("mean_pf", ascending=False).head(top).reset_index(drop=True)
    return agg


def main() -> int:
    init_logger()
    logger = get_logger(__name__)
    args = parse_args()

    results_path = PROJECT_ROOT / args.results_csv
    if not results_path.exists():
        logger.error(f"results CSV not found: {results_path}")
        return 1

    df = pd.read_csv(results_path)
    logger.info(f"Loaded {len(df):,} rows from {results_path.name}")

    top = _aggregate_top(df, args.top, args.min_trades)
    if top.empty:
        logger.error("No configs passed the PF/trade floors.")
        return 1
    logger.info(f"Top {len(top)} configs selected:")
    print(top[[
        "timeframe", "family", "mean_pf", "mean_expectancy",
        "mean_win_rate", "mean_max_dd", "mean_trades",
    ]].round(3).to_string(index=False))

    logger.info(f"Loading {args.symbol} M1 {args.start}..{args.end}")
    m1 = load_symbol_bars(args.symbol, start=args.start, end=args.end)
    # Cache per timeframe so we don't resample repeatedly.
    tf_cache: dict[str, pd.DataFrame] = {}

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M")
    out_dir = PROJECT_ROOT / "backtest_results" / f"trade_records_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    for rank, row in top.iterrows():
        tf = row["timeframe"]
        resample_tf = TIMEFRAME_TO_RESAMPLE.get(tf, tf)
        if resample_tf not in tf_cache:
            tf_cache[resample_tf] = resample_bars(m1, resample_tf)
        bars = tf_cache[resample_tf]

        family_cls = get_family_by_name(row["family"])
        if family_cls is None:
            logger.warning(f"Unknown family {row['family']} — skipping rank {rank}")
            continue

        family_params = json.loads(row["family_params"])
        exit_dict = json.loads(row["exit_config"])
        exit_cfg = ExitConfig(
            sl_atr_mult=exit_dict["sl_atr_mult"],
            atr_length=int(exit_dict["atr_length"]),
            tp_r_mult=exit_dict["tp_r_mult"],
            trail_kind=exit_dict["trail_kind"],
            trail_atr_mult=exit_dict.get("trail_atr_mult"),
        )

        out_path = out_dir / f"top{rank+1:02d}_{row['family']}_{tf}.parquet"
        logger.info(f"[{rank+1}/{len(top)}] capture: {row['family']} {tf} → {out_path.name}")
        try:
            trades = capture_trade_records(
                bars, family_cls(), family_params, exit_cfg,
                initial_cash=500.0,
                output_path=out_path,
            )
        except Exception as e:
            logger.error(f"rank {rank+1} failed: {e}")
            continue

        summary_rows.append({
            "rank": rank + 1,
            "family": row["family"],
            "timeframe": tf,
            "parquet": out_path.name,
            "n_trades": len(trades),
            "mean_pf_oos": row["mean_pf"],
            "mean_expectancy_oos": row["mean_expectancy"],
            "mae_median_pct": float(trades["mae_pct"].median()) if len(trades) else None,
            "mae_p10_pct": float(trades["mae_pct"].quantile(0.1)) if len(trades) else None,
            "mfe_median_pct": float(trades["mfe_pct"].median()) if len(trades) else None,
            "mfe_p90_pct": float(trades["mfe_pct"].quantile(0.9)) if len(trades) else None,
            "wr_full_sample": float((trades["PnL"] > 0).mean()) if len(trades) else None,
            "family_params": row["family_params"],
            "exit_config": row["exit_config"],
        })

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / f"top{args.top}_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info(f"Wrote summary: {summary_path}")
    print()
    print("=" * 76)
    print("Trade-record capture summary (round 7)")
    print("=" * 76)
    print(summary[[
        "rank", "family", "timeframe", "n_trades",
        "mean_pf_oos", "mae_median_pct", "mfe_p90_pct", "wr_full_sample",
    ]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
