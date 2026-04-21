"""Phase-2 exploratory sweep orchestrator.

Given a price DataFrame (bid/ask OHLCV) and a list of signal families,
runs:

1. For each family, iterate its param grid.
2. For each (family, params), generate entries/exits.
3. Combine with each :class:`ExitConfig` from the exit-framework grid.
4. Run ``vbt.Portfolio.from_signals`` with walk-forward splits.
5. Collect OOS metrics into a flat DataFrame.

Results land in both:
- A CSV at ``backtest_results/explore_YYYYMMDD_HHMM/full_results.csv``
- The ``backtest_runs`` SQLite table via :mod:`src.backtest.registry`

Walk-forward is mandatory (anti-overfitting). No other filters applied.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import MAX_LEVERAGE
from src.backtest.metrics import BacktestMetrics, compute_metrics
from src.strategies.exits import ExitConfig, enumerate_exit_configs
from src.strategies.families.base_family import SignalFamily
from src.utils.diary import log_event
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ExploreConfig:
    """Top-level runtime config for :func:`explore`.

    Attributes:
        data_range: ``(start_iso, end_iso)`` — the full period to cover.
        train_frac: Fraction of each walk-forward window used for training.
        walk_forward_windows: Number of rolling walk-forward windows.
            Set 0 to skip walk-forward (single split over full range).
        random_subset_per_family: If set, randomly sample N param combos
            per family rather than full Cartesian. Recommended for broad
            exploration.
        exit_random_subset: Same for exit configs.
        leverage: Position leverage.
        initial_cash: Account size for the simulation.
    """

    data_range: tuple[str, str]
    train_frac: float = 0.5
    walk_forward_windows: int = 4
    random_subset_per_family: int | None = 30
    exit_random_subset: int | None = 30
    leverage: int = MAX_LEVERAGE
    initial_cash: float = 500.0


@dataclass(frozen=True, slots=True)
class SingleRunResult:
    """One (family, family_params, exit_config, split) result."""

    family_name: str
    family_params: dict[str, Any]
    exit_config: dict[str, Any]
    split_label: str
    split_kind: str  # "IS" or "OOS"
    metrics: BacktestMetrics


def _iter_param_combos(grid: dict[str, list[Any]]) -> Iterator[dict[str, Any]]:
    """Yield each combo from a named grid as a dict."""
    if not grid:
        yield {}
        return
    keys = list(grid.keys())
    for values in product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values, strict=False))


def _subsample(items: list[Any], n: int | None, seed: int = 42) -> list[Any]:
    if n is None or len(items) <= n:
        return items
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(items), size=n, replace=False)
    return [items[i] for i in sorted(idx.tolist())]


def _walk_forward_slices(
    close: pd.Series,
    n_windows: int,
    train_frac: float,
) -> list[tuple[str, slice, slice]]:
    """Return (label, is_slice, oos_slice) tuples.

    If ``n_windows == 0``: single split over entire range — use top
    ``train_frac`` for IS and the rest for OOS.
    """
    n = len(close)
    if n_windows == 0:
        split_at = int(n * train_frac)
        return [
            ("full", slice(0, split_at), slice(split_at, n)),
        ]
    win_len = n // n_windows
    splits: list[tuple[str, slice, slice]] = []
    for i in range(n_windows):
        start = i * win_len
        end = start + win_len
        train_end = start + int(win_len * train_frac)
        splits.append((f"win{i + 1}", slice(start, train_end), slice(train_end, end)))
    return splits


def _run_single_backtest(
    *,
    close: pd.Series,
    entries_long: pd.Series,
    entries_short: pd.Series,
    sl_frac: pd.Series,
    tp_frac: pd.Series,
    trail_frac: pd.Series | None,
    use_trail: bool,
    leverage: int,
    initial_cash: float,
    slippage: np.ndarray | None,
) -> BacktestMetrics:
    """Run ``vbt.Portfolio.from_signals`` for one config + compute metrics."""
    import vectorbtpro as vbt

    # sl/tp as fractions; NaN where no entry bar → vbt ignores.
    kwargs: dict[str, Any] = {
        "close": close,
        "entries": entries_long,
        "short_entries": entries_short,
        "sl_stop": sl_frac.fillna(0.0).to_numpy(),
        "tp_stop": tp_frac.fillna(0.0).to_numpy(),
        "sl_trail": use_trail,
        "leverage": leverage,
        "init_cash": initial_cash,
        "freq": "1min",
        "fees": 0.0,
    }
    if slippage is not None:
        kwargs["slippage"] = slippage

    try:
        pf = vbt.Portfolio.from_signals(**kwargs)
    except Exception as e:
        logger.warning(f"Portfolio.from_signals failed: {e}")
        return BacktestMetrics(
            total_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            expectancy_usd=0.0,
            sharpe=0.0,
            sortino=0.0,
            calmar=0.0,
            max_drawdown_pct=0.0,
            avg_drawdown_duration_bars=0.0,
            cagr=0.0,
        )

    returns = pd.Series(pf.returns.values, index=close.index)
    trade_pnl = pd.Series(
        pf.trades.pnl.values if hasattr(pf.trades, "pnl") else []
    )
    return compute_metrics(
        returns=returns,
        trade_pnl_usd=trade_pnl,
        initial_cash=initial_cash,
    )


def _half_spread_slippage(bars: pd.DataFrame) -> np.ndarray:
    """Per-bar half-spread as a numpy array, aligned to bars.index.

    Prefers ``bid_close`` + ``ask_close`` columns; falls back to zero
    slippage if only mid data is available.
    """
    if "bid_close" in bars.columns and "ask_close" in bars.columns:
        half = (bars["ask_close"] - bars["bid_close"]) / 2.0
        return half.to_numpy()
    return np.zeros(len(bars), dtype=float)


def explore(
    bars: pd.DataFrame,
    families: list[SignalFamily],
    config: ExploreConfig,
    *,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Run the exploratory sweep and return a flat results DataFrame.

    Args:
        bars: OHLCV bars with mid_/bid_/ask_ columns, tz-aware UTC index.
        families: Signal families to include. Each family exposes its own
            param grid via :meth:`SignalFamily.param_grid`.
        config: Runtime sweep config.
        output_dir: Directory to write CSVs to. Defaults to
            ``backtest_results/explore_<ts>/`` under the project root.

    Returns:
        DataFrame with one row per (family, family_params, exit_config,
        split) result.
    """
    from src.strategies.exits import config_to_vbt_params

    logger.info(
        f"explore: families={[type(f).__name__ for f in families]} "
        f"bars={len(bars):,} range={bars.index.min()}..{bars.index.max()}"
    )
    # Determine mid close once.
    mid_close_col = (
        "mid_close" if "mid_close" in bars.columns else "bid_close"
    )
    close = bars[mid_close_col]
    slippage = _half_spread_slippage(bars)

    splits = _walk_forward_slices(
        close, config.walk_forward_windows, config.train_frac
    )
    exit_configs_full = enumerate_exit_configs()
    exit_configs = _subsample(exit_configs_full, config.exit_random_subset)
    logger.info(
        f"explore: {len(splits)} splits × {len(exit_configs)} exit configs"
    )

    rows: list[dict[str, Any]] = []
    artifacts_dir = output_dir or _default_output_dir()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for family in families:
        grid = family.param_grid()
        combos = list(_iter_param_combos(grid))
        combos = _subsample(combos, config.random_subset_per_family)
        logger.info(
            f"explore [{family.name}]: {len(combos)} param combos × "
            f"{len(exit_configs)} exits × {len(splits)} splits = "
            f"{len(combos) * len(exit_configs) * len(splits)} runs"
        )

        for family_params in combos:
            try:
                fam_instance = _instantiate(family, family_params)
            except Exception as e:
                logger.warning(f"Cannot instantiate {family.name} with {family_params}: {e}")
                continue

            signals = fam_instance.generate(bars)
            # ATR needs to be computed once per family (some families use it).
            from src.indicators.engine import add_atr

            try:
                atr_frame = add_atr(bars, length=14)
                atr = atr_frame["atr_14"]
            except Exception:
                atr = pd.Series(close.diff().abs().rolling(14).mean(), index=close.index)

            for exit_cfg in exit_configs:
                vbt_params = config_to_vbt_params(
                    entries_long=signals.entries_long,
                    entries_short=signals.entries_short,
                    close=close,
                    atr=atr,
                    config=exit_cfg,
                )

                for split_label, is_slice, oos_slice in splits:
                    for kind, sl in [("IS", is_slice), ("OOS", oos_slice)]:
                        if sl.stop - sl.start < 200:
                            continue  # skip micro-splits
                        sub_close = close.iloc[sl]
                        sub_long = signals.entries_long.iloc[sl]
                        sub_short = signals.entries_short.iloc[sl]
                        sub_sl = vbt_params.sl_stop.iloc[sl]
                        sub_tp = vbt_params.tp_stop.iloc[sl]
                        sub_slip = (
                            slippage[sl.start : sl.stop] if slippage is not None else None
                        )

                        metrics = _run_single_backtest(
                            close=sub_close,
                            entries_long=sub_long,
                            entries_short=sub_short,
                            sl_frac=sub_sl,
                            tp_frac=sub_tp,
                            trail_frac=None,
                            use_trail=vbt_params.sl_trail,
                            leverage=config.leverage,
                            initial_cash=config.initial_cash,
                            slippage=sub_slip,
                        )

                        row = {
                            "family": family.name,
                            "family_params": _json_safe(family_params),
                            "exit_config": _json_safe(_exit_config_dict(exit_cfg)),
                            "split": split_label,
                            "kind": kind,
                            **metrics.as_dict(),
                        }
                        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = artifacts_dir / "full_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"explore: wrote {len(df)} rows to {csv_path}")

    log_event(
        "exploration_complete",
        artifacts_dir=str(artifacts_dir),
        families=[f.name for f in families],
        total_runs=len(df),
        csv_path=str(csv_path),
    )
    return df


def _default_output_dir() -> Path:
    root = Path(__file__).resolve().parent.parent.parent
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M")
    return root / "backtest_results" / f"explore_{stamp}"


def _json_safe(obj: Any) -> str:
    import json

    return json.dumps(obj, default=str, sort_keys=True)


def _exit_config_dict(cfg: ExitConfig) -> dict[str, Any]:
    return {
        "sl_atr_mult": cfg.sl_atr_mult,
        "atr_length": cfg.atr_length,
        "tp_r_mult": cfg.tp_r_mult,
        "trail_kind": cfg.trail_kind,
        "trail_atr_mult": cfg.trail_atr_mult,
    }


def _instantiate(family: SignalFamily, params: dict[str, Any]) -> SignalFamily:
    """Create a new family instance from a params dict.

    Each family's constructor accepts an optional typed params dataclass.
    We inspect the class for its params dataclass type and build one.
    """
    import inspect

    cls = type(family)
    sig = inspect.signature(cls.__init__)
    # The first arg after `self` should be `params` — its annotation is the
    # dataclass type.
    params_param = list(sig.parameters.values())[1]
    params_type = params_param.annotation
    # Strip Optional / Union wrappers if any.
    import typing

    origin = typing.get_origin(params_type)
    if origin is typing.Union:
        args = [a for a in typing.get_args(params_type) if a is not type(None)]
        params_type = args[0]
    return cls(params_type(**params))
