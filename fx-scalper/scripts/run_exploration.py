"""Run the Phase-2 exploratory sweep on a symbol's processed Parquet data.

Usage:
    python scripts/run_exploration.py --symbol EUR_USD --start 2023-01-01 --end 2026-04-20

Outputs:
    backtest_results/explore_<ts>/full_results.csv  — every (family, params, exit, split) row
    backtest_results/explore_<ts>/ranking_summary.md — human-readable summary
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.backtest.data_loader import load_symbol_bars  # noqa: E402
from src.backtest.explorer import ExploreConfig, explore  # noqa: E402
from src.strategies.families import ALL_FAMILIES  # noqa: E402
from src.utils.logger import get_logger, init_logger  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase-2 exploratory sweep")
    p.add_argument("--symbol", default="EUR_USD")
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default=date.today().isoformat())
    p.add_argument("--walk-forward-windows", type=int, default=4)
    p.add_argument("--family-random-subset", type=int, default=30,
                   help="Sample N param combos per family (None for full grid)")
    p.add_argument("--exit-random-subset", type=int, default=30,
                   help="Sample N exit configs (None for full grid)")
    p.add_argument("--initial-cash", type=float, default=500.0)
    p.add_argument("--leverage", type=int, default=50)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--limit-families", default=None,
                   help="Comma-separated subset of family names")
    return p.parse_args()


def _ranking_summary(df: pd.DataFrame) -> str:
    """Build a markdown ranking summary from the flat results DataFrame."""
    if df.empty or "kind" not in df.columns:
        return "_(no results — all runs failed; check the explorer log)_\n"
    oos = df[df["kind"] == "OOS"].copy()
    if oos.empty:
        return "_(no OOS rows)_"

    agg = (
        oos.groupby(["family", "family_params", "exit_config"])
        .agg(
            avg_profit_factor=("profit_factor", "mean"),
            avg_sharpe=("sharpe", "mean"),
            avg_win_rate=("win_rate", "mean"),
            avg_total_trades=("total_trades", "mean"),
            avg_max_dd=("max_drawdown_pct", "mean"),
            avg_expectancy=("expectancy_usd", "mean"),
            n_splits=("split", "count"),
        )
        .reset_index()
    )

    lines = [
        "# Exploration ranking summary",
        "",
        f"Generated: {pd.Timestamp.utcnow()} UTC",
        f"Total config combos: {len(agg):,}",
        "",
    ]

    for metric in ("avg_profit_factor", "avg_sharpe", "avg_win_rate"):
        top = agg.sort_values(metric, ascending=False).head(10)
        lines.append(f"## Top 10 by {metric}")
        lines.append("")
        lines.append(top.to_markdown(index=False))
        lines.append("")

    lines.append("## Per-family best (by avg_profit_factor)")
    lines.append("")
    best_per_family = (
        agg.sort_values("avg_profit_factor", ascending=False)
        .groupby("family")
        .head(1)
        .sort_values("avg_profit_factor", ascending=False)
    )
    lines.append(best_per_family.to_markdown(index=False))
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    init_logger()
    logger = get_logger(__name__)
    args = parse_args()

    logger.info(f"Loading {args.symbol} bars {args.start} → {args.end}")
    bars = load_symbol_bars(args.symbol, start=args.start, end=args.end)
    if bars.empty:
        logger.error(f"No bars loaded for {args.symbol}. Did you run pull_dukascopy?")
        return 1
    logger.info(f"Loaded {len(bars):,} bars")

    families = list(ALL_FAMILIES)
    if args.limit_families:
        wanted = {s.strip() for s in args.limit_families.split(",")}
        families = [f for f in families if f.name in wanted]
        if not families:
            logger.error(f"No families match {args.limit_families}")
            return 1

    # Instantiate with default params — explorer will re-instantiate per combo.
    family_instances = [f() for f in families]

    cfg = ExploreConfig(
        data_range=(args.start, args.end),
        walk_forward_windows=args.walk_forward_windows,
        random_subset_per_family=args.family_random_subset,
        exit_random_subset=args.exit_random_subset,
        leverage=args.leverage,
        initial_cash=args.initial_cash,
    )

    output_dir = Path(args.output_dir) if args.output_dir else None
    df = explore(bars, family_instances, cfg, output_dir=output_dir)

    # Find the directory explore() wrote into and drop a ranking summary there.
    latest = max(
        (PROJECT_ROOT / "backtest_results").glob("explore_*"),
        key=lambda p: p.stat().st_mtime,
        default=None,
    )
    if latest is not None:
        (latest / "ranking_summary.md").write_text(_ranking_summary(df))
        logger.info(f"Ranking summary → {latest / 'ranking_summary.md'}")

    logger.info(f"Exploration complete. {len(df)} total rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
