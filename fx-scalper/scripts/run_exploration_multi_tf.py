"""Round 3 — multi-timeframe exploration.

Resamples M1 EUR/USD to M5, M15, M30, H1 and runs the full family sweep
on each. Writes a combined CSV with a ``timeframe`` column for cross-TF
analysis.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.backtest.data_loader import load_symbol_bars  # noqa: E402
from src.backtest.explorer import ExploreConfig, explore  # noqa: E402
from src.backtest.resample import resample_bars  # noqa: E402
from src.strategies.families import ALL_FAMILIES  # noqa: E402
from src.utils.logger import get_logger, init_logger  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Round 3 multi-timeframe exploration")
    p.add_argument("--symbol", default="EUR_USD")
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2026-04-20")
    p.add_argument(
        "--timeframes",
        default="5min,15min,30min,1H",
        help="Comma-separated timeframes to test",
    )
    p.add_argument("--walk-forward-windows", type=int, default=3)
    p.add_argument("--family-random-subset", type=int, default=20)
    p.add_argument("--exit-random-subset", type=int, default=15)
    return p.parse_args()


def main() -> int:
    init_logger()
    logger = get_logger(__name__)
    args = parse_args()

    logger.info(f"Loading M1 bars for {args.symbol}")
    m1 = load_symbol_bars(args.symbol, start=args.start, end=args.end)
    if m1.empty:
        logger.error(f"No bars for {args.symbol}")
        return 1
    logger.info(f"Loaded {len(m1):,} M1 bars")

    timeframes = [t.strip() for t in args.timeframes.split(",") if t.strip()]
    all_frames: list[pd.DataFrame] = []

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M")
    combined_dir = PROJECT_ROOT / "backtest_results" / f"explore_multi_tf_{stamp}"
    combined_dir.mkdir(parents=True, exist_ok=True)

    for tf in timeframes:
        logger.info(f"=== Resampling to {tf} ===")
        bars = resample_bars(m1, tf)  # type: ignore[arg-type]
        logger.info(f"{tf}: {len(bars):,} bars ({bars.index.min()} → {bars.index.max()})")

        family_instances = [f() for f in ALL_FAMILIES]
        cfg = ExploreConfig(
            data_range=(args.start, args.end),
            walk_forward_windows=args.walk_forward_windows,
            random_subset_per_family=args.family_random_subset,
            exit_random_subset=args.exit_random_subset,
        )
        tf_dir = combined_dir / f"tf_{tf}"
        tf_dir.mkdir(parents=True, exist_ok=True)
        df = explore(bars, family_instances, cfg, output_dir=tf_dir)
        df["timeframe"] = tf
        df["symbol"] = args.symbol
        all_frames.append(df)
        logger.info(f"{tf}: {len(df)} rows")

    combined = pd.concat(all_frames, ignore_index=True)
    combined_path = combined_dir / "combined_results.csv"
    combined.to_csv(combined_path, index=False)
    logger.info(f"Combined results → {combined_path} ({len(combined):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
