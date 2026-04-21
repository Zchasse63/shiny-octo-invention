"""Load processed Parquet bars for backtesting, and resample ticks → M1.

Layout per CLAUDE.md §Historical Data:
``data/processed/{symbol}/year={YYYY}/month={MM}/bars.parquet``

Each Parquet file has columns: ``bid_open, bid_high, bid_low, bid_close,
ask_open, ask_high, ask_low, ask_close, volume`` indexed by tz-aware UTC
timestamp (minute bars).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from config.settings import DATA_PROCESSED_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _processed_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent / DATA_PROCESSED_DIR


def load_symbol_bars(
    symbol: str,
    *,
    start: str | None = None,
    end: str | None = None,
    root: Path | None = None,
) -> pd.DataFrame:
    """Load M1 bars for ``symbol`` between optional ``start`` / ``end``.

    Args:
        symbol: OANDA-style symbol (e.g. ``EUR_USD``).
        start: Inclusive ISO date string (``"2023-01-01"``) or None.
        end: Inclusive ISO date string or None.
        root: Override processed-data root for tests.

    Returns:
        DataFrame indexed by tz-aware UTC timestamp, sorted ascending.
        Empty if no files match.
    """
    base = (root or _processed_root()) / symbol
    if not base.exists():
        logger.warning(f"No processed data for {symbol} at {base}")
        return pd.DataFrame()

    # Walk year=*/month=*/bars.parquet — pyarrow/pandas handles partitioning.
    files = sorted(base.rglob("bars.parquet"))
    if not files:
        logger.warning(f"No bars.parquet under {base}")
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for f in files:
        frames.append(pd.read_parquet(f))
    df = pd.concat(frames).sort_index()

    # Ensure tz-aware UTC index.
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df.index <= pd.Timestamp(end, tz="UTC")]
    return df


def resample_ticks_to_m1(ticks: pd.DataFrame) -> pd.DataFrame:
    """Resample tick-level bid/ask quotes to M1 OHLCV bid + ask bars.

    Args:
        ticks: DataFrame indexed by tz-aware UTC timestamp, columns
            ``bid`` and ``ask`` (prices), optional ``volume`` (tick count).

    Returns:
        M1 frame with bid_/ask_ OHLCV columns.
    """
    if ticks.empty:
        return pd.DataFrame()
    if ticks.index.tz is None:
        raise ValueError("ticks index must be tz-aware UTC")
    bid = ticks["bid"].resample("1min").ohlc()
    ask = ticks["ask"].resample("1min").ohlc()
    bid.columns = [f"bid_{c}" for c in bid.columns]
    ask.columns = [f"ask_{c}" for c in ask.columns]
    vol = (
        ticks.get("volume", pd.Series(1, index=ticks.index))
        .resample("1min")
        .sum()
        .rename("volume")
    )
    out = pd.concat([bid, ask, vol], axis=1).dropna(subset=["bid_open", "ask_open"])
    return out


def save_symbol_bars(
    bars: pd.DataFrame,
    *,
    symbol: str,
    root: Path | None = None,
) -> list[Path]:
    """Write ``bars`` to Parquet partitioned by year/month.

    Args:
        bars: M1 bars indexed by tz-aware UTC timestamp.
        symbol: OANDA-style symbol.
        root: Override processed-data root for tests.

    Returns:
        List of Parquet paths written.
    """
    if bars.empty:
        return []
    if bars.index.tz is None:
        raise ValueError("bars index must be tz-aware UTC")

    base = (root or _processed_root()) / symbol
    written: list[Path] = []
    for (year, month), chunk in _group_by_year_month(bars):
        out_dir = base / f"year={year:04d}" / f"month={month:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "bars.parquet"
        chunk.to_parquet(path)
        written.append(path)
    logger.info(f"Wrote {len(written)} Parquet partitions for {symbol}")
    return written


def _group_by_year_month(
    bars: pd.DataFrame,
) -> Iterable[tuple[tuple[int, int], pd.DataFrame]]:
    """Yield ((year, month), sub-frame) pairs.

    Uses boolean masking on the period index so it works correctly even if
    ``bars`` is a filtered slice of a larger frame (where iloc-by-position
    would silently pick wrong rows).
    """
    # Strip tz for to_period() — ``to_period`` drops tz anyway, but converting
    # explicitly silences the pandas UserWarning. The boolean mask stays on the
    # tz-aware index so we return the original tz-aware rows.
    index_naive = bars.index.tz_convert(None) if bars.index.tz is not None else bars.index
    ym = index_naive.to_period("M")
    for period in pd.unique(ym):
        mask = ym == period
        chunk = bars[mask]
        yield (period.year, period.month), chunk
