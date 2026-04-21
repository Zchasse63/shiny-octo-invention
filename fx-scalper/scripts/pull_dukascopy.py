"""Day 2: pull Dukascopy tick data and resample to M1 Parquet.

Usage:
    python scripts/pull_dukascopy.py --start 2023-01-01 --end 2023-12-31

Per CLAUDE.md §Historical Data: Dukascopy bid/ask tick data is the primary
source. We store as Parquet partitioned by ``symbol/year/month``.

Uses :mod:`src.backtest.dukascopy_client` — a direct synchronous downloader
written in-house because duka==0.2.0's asyncio fetch pipeline drops hours.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from config.settings import DATA_RAW_DIR, INSTRUMENTS  # noqa: E402
from src.backtest.data_loader import resample_ticks_to_m1, save_symbol_bars  # noqa: E402
from src.backtest.dukascopy_client import fetch_range  # noqa: E402
from src.utils.logger import get_logger, init_logger  # noqa: E402

# Map OANDA underscore names to Dukascopy instrument codes.
OANDA_TO_DUKA = {
    "EUR_USD": "EURUSD",
    "GBP_USD": "GBPUSD",
    "USD_JPY": "USDJPY",
}


def _duka_symbol(instrument: str) -> str:
    try:
        return OANDA_TO_DUKA[instrument]
    except KeyError:
        raise ValueError(f"No Dukascopy mapping for {instrument}") from None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pull Dukascopy tick → M1 Parquet")
    p.add_argument("--start", type=str, default="2023-01-01", help="YYYY-MM-DD")
    p.add_argument(
        "--end",
        type=str,
        default=date.today().isoformat(),
        help="YYYY-MM-DD (inclusive)",
    )
    p.add_argument(
        "--instruments",
        type=str,
        default=",".join(INSTRUMENTS),
        help="Comma-separated OANDA names",
    )
    p.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Override raw dir (default: data/raw)",
    )
    p.add_argument(
        "--skip-raw-cache",
        action="store_true",
        help="Skip writing per-day raw CSV (only write M1 Parquet)",
    )
    return p.parse_args()


def main() -> int:
    init_logger()
    logger = get_logger(__name__)
    args = parse_args()

    start = datetime.fromisoformat(args.start).date()
    end = datetime.fromisoformat(args.end).date()
    instruments = [s.strip() for s in args.instruments.split(",") if s.strip()]
    raw_root = Path(args.raw_dir) if args.raw_dir else PROJECT_ROOT / DATA_RAW_DIR

    for oanda_sym in instruments:
        duka_sym = _duka_symbol(oanda_sym)
        sym_raw_dir = raw_root / duka_sym
        sym_raw_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[{oanda_sym}] raw dir: {sym_raw_dir}")

        all_ticks: list[pd.DataFrame] = []
        for day, ticks in fetch_range(duka_sym, start, end):
            if ticks.empty:
                continue
            all_ticks.append(ticks)
            if not args.skip_raw_cache:
                out = sym_raw_dir / f"{duka_sym}_{day.isoformat()}.csv"
                ticks.to_csv(out)

        if not all_ticks:
            logger.warning(f"[{oanda_sym}] no ticks in range {start}..{end}")
            continue

        combined = pd.concat(all_ticks).sort_index()
        logger.info(
            f"[{oanda_sym}] resampling {len(combined):,} ticks → M1"
        )
        bars = resample_ticks_to_m1(combined)
        if bars.empty:
            logger.warning(f"[{oanda_sym}] resample returned empty")
            continue
        paths = save_symbol_bars(bars, symbol=oanda_sym)
        logger.info(
            f"[{oanda_sym}] wrote {len(paths)} partitions "
            f"covering {bars.index.min()} → {bars.index.max()}"
        )

    logger.info("pull_dukascopy finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
