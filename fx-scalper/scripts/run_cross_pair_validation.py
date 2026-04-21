"""Round 4 — cross-pair validation.

Takes the top-N performing (family, params, exit_config) from rounds 2+3
and re-runs each on GBP/USD and USD/JPY. A config "survives" if it
maintains PF > 1.0 with positive expectancy on ALL three pairs.

Survivors are the candidates for Phase-3 formalization; non-survivors
were EUR/USD-specific artifacts.
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.backtest.data_loader import load_symbol_bars  # noqa: E402
from src.backtest.explorer import (  # noqa: E402
    _instantiate,
    _run_single_backtest,
    _walk_forward_slices,
)
from src.backtest.resample import resample_bars  # noqa: E402
from src.strategies.exits import ExitConfig, config_to_vbt_params  # noqa: E402
from src.strategies.families import ALL_FAMILIES  # noqa: E402
from src.utils.logger import get_logger, init_logger  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Round 4 cross-pair validation")
    p.add_argument(
        "--round2-csv",
        default="backtest_results/explore_20260421T2116/full_results.csv",
    )
    p.add_argument(
        "--round3-csv",
        default="backtest_results/explore_multi_tf_20260421T2137/combined_results.csv",
    )
    p.add_argument(
        "--pairs",
        default="EUR_USD,GBP_USD,USD_JPY",
        help="Comma-separated instruments to test",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Top N configs per (timeframe, family) to re-validate",
    )
    p.add_argument("--walk-forward-windows", type=int, default=3)
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2026-04-20")
    return p.parse_args()


def _top_configs_from_round2(
    csv_path: Path, top_n: int = 20
) -> list[dict]:
    """Extract round-2 top performers (all on M1)."""
    df = pd.read_csv(csv_path)
    oos = df[(df["kind"] == "OOS") & (df["total_trades"] >= 50)]
    agg = (
        oos.groupby(["family", "family_params", "exit_config"])
        .agg(pf=("profit_factor", "mean"), exp=("expectancy_usd", "mean"),
             n=("split", "count"))
        .reset_index()
    )
    full = agg[(agg["n"] == 3) & (agg["pf"] > 1.1) & (agg["exp"] > 0)]
    top = full.sort_values("pf", ascending=False).head(top_n)
    out = []
    for _, r in top.iterrows():
        out.append({
            "source_round": 2,
            "timeframe": "1min",
            "family": r["family"],
            "family_params": r["family_params"],
            "exit_config": r["exit_config"],
            "original_pf": r["pf"],
            "original_exp": r["exp"],
        })
    return out


def _top_configs_from_round3(csv_path: Path, top_n: int = 20) -> list[dict]:
    """Extract round-3 top performers per (timeframe, family)."""
    df = pd.read_csv(csv_path)
    oos = df[(df["kind"] == "OOS") & (df["total_trades"] >= 30)]
    agg = (
        oos.groupby(["timeframe", "family", "family_params", "exit_config"])
        .agg(pf=("profit_factor", "mean"), exp=("expectancy_usd", "mean"),
             n=("split", "count"))
        .reset_index()
    )
    full = agg[(agg["n"] == 3) & (agg["pf"] > 1.1) & (agg["exp"] > 0)]
    # Top N overall, not per family — we want diversity but also best overall.
    top = full.sort_values("pf", ascending=False).head(top_n)
    out = []
    for _, r in top.iterrows():
        out.append({
            "source_round": 3,
            "timeframe": r["timeframe"],
            "family": r["family"],
            "family_params": r["family_params"],
            "exit_config": r["exit_config"],
            "original_pf": r["pf"],
            "original_exp": r["exp"],
        })
    return out


def _half_spread_slippage(bars: pd.DataFrame) -> np.ndarray:
    if "bid_close" in bars.columns and "ask_close" in bars.columns:
        return ((bars["ask_close"] - bars["bid_close"]) / 2.0).to_numpy()
    return np.zeros(len(bars), dtype=float)


def _rerun_config(
    config: dict,
    bars: pd.DataFrame,
    walk_forward_windows: int,
) -> pd.DataFrame:
    """Re-run a single (family, params, exit_config) on the given pair's bars.

    Returns a DataFrame: one row per walk-forward IS+OOS split with metrics.
    """
    family_params_dict = json.loads(config["family_params"])
    exit_config_dict = json.loads(config["exit_config"])

    # Find family class.
    family_cls = next((f for f in ALL_FAMILIES if f.name == config["family"]), None)
    if family_cls is None:
        return pd.DataFrame()

    try:
        family = _instantiate(family_cls(), family_params_dict)
    except Exception:
        return pd.DataFrame()

    signals = family.generate(bars)
    close_col = "mid_close" if "mid_close" in bars.columns else "bid_close"
    close = bars[close_col]

    # Rebuild ATR series used by the exit framework.
    from src.indicators.engine import add_atr
    atr_df = add_atr(bars, length=exit_config_dict.get("atr_length", 14))
    atr = atr_df[f"atr_{exit_config_dict.get('atr_length', 14)}"]

    exit_cfg = ExitConfig(
        sl_atr_mult=exit_config_dict["sl_atr_mult"],
        atr_length=exit_config_dict["atr_length"],
        tp_r_mult=exit_config_dict.get("tp_r_mult"),
        trail_kind=exit_config_dict["trail_kind"],
        trail_atr_mult=exit_config_dict["trail_atr_mult"],
    )
    vbt_params = config_to_vbt_params(
        entries_long=signals.entries_long,
        entries_short=signals.entries_short,
        close=close,
        atr=atr,
        config=exit_cfg,
    )

    slippage = _half_spread_slippage(bars)
    splits = _walk_forward_slices(close, walk_forward_windows, 0.5)

    rows = []
    for split_label, is_sl, oos_sl in splits:
        for kind, sl in [("IS", is_sl), ("OOS", oos_sl)]:
            if sl.stop - sl.start < 200:
                continue
            metrics = _run_single_backtest(
                close=close.iloc[sl],
                entries_long=signals.entries_long.iloc[sl],
                entries_short=signals.entries_short.iloc[sl],
                sl_frac=vbt_params.sl_stop.iloc[sl],
                tp_frac=vbt_params.tp_stop.iloc[sl],
                trail_frac=(vbt_params.trail_distance_pct.iloc[sl]
                            if vbt_params.trail_distance_pct is not None else None),
                use_trail=vbt_params.sl_trail,
                leverage=50,
                initial_cash=500.0,
                slippage=slippage[sl.start:sl.stop],
            )
            rows.append({
                "split": split_label,
                "kind": kind,
                **metrics.as_dict(),
            })
    return pd.DataFrame(rows)


def main() -> int:
    init_logger()
    logger = get_logger(__name__)
    args = parse_args()

    r2 = _top_configs_from_round2(PROJECT_ROOT / args.round2_csv, args.top_n)
    r3 = _top_configs_from_round3(PROJECT_ROOT / args.round3_csv, args.top_n)
    configs = r2 + r3
    logger.info(f"Loaded {len(configs)} candidate configs "
                f"(r2={len(r2)}, r3={len(r3)})")

    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M")
    out_dir = PROJECT_ROOT / "backtest_results" / f"cross_pair_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    # Cache resampled data per (pair, timeframe) to avoid redundant work.
    bars_cache: dict[tuple[str, str], pd.DataFrame] = {}

    for pair in pairs:
        logger.info(f"Loading {pair} M1 bars")
        m1 = load_symbol_bars(pair, start=args.start, end=args.end)
        if m1.empty:
            logger.warning(f"No data for {pair} — skipping")
            continue
        for config in configs:
            tf = config["timeframe"]
            key = (pair, tf)
            if key not in bars_cache:
                bars_cache[key] = resample_bars(m1, tf)  # type: ignore[arg-type]
            bars = bars_cache[key]
            split_df = _rerun_config(config, bars, args.walk_forward_windows)
            if split_df.empty:
                continue
            oos = split_df[split_df["kind"] == "OOS"]
            if len(oos) == 0:
                continue
            result = {
                "pair": pair,
                "source_round": config["source_round"],
                "timeframe": tf,
                "family": config["family"],
                "family_params": config["family_params"],
                "exit_config": config["exit_config"],
                "original_pf": config["original_pf"],
                "original_exp": config["original_exp"],
                "oos_pf": oos["profit_factor"].mean(),
                "oos_exp": oos["expectancy_usd"].mean(),
                "oos_sharpe": oos["sharpe"].mean(),
                "oos_wr": oos["win_rate"].mean(),
                "oos_trades": oos["total_trades"].mean(),
                "oos_dd": oos["max_drawdown_pct"].mean(),
                "n_oos_splits": len(oos),
            }
            results.append(result)

    results_df = pd.DataFrame(results)
    results_path = out_dir / "cross_pair_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Wrote {len(results_df)} rows to {results_path}")

    # Pivot table — for each config, show OOS PF on each pair.
    if not results_df.empty:
        pivot = results_df.pivot_table(
            index=["timeframe", "family", "family_params", "exit_config"],
            columns="pair",
            values="oos_pf",
        ).reset_index()
        pivot_path = out_dir / "cross_pair_pivot.csv"
        pivot.to_csv(pivot_path, index=False)
        logger.info(f"Pivot table → {pivot_path}")

        # Survivors: PF > 1.0 on ALL 3 pairs
        pair_cols = [c for c in pivot.columns if c in pairs]
        if len(pair_cols) == len(pairs):
            mask = (pivot[pair_cols] > 1.0).all(axis=1)
            survivors = pivot[mask]
            survivors_path = out_dir / "cross_pair_survivors.csv"
            survivors.to_csv(survivors_path, index=False)
            logger.info(f"{len(survivors)} configs survive all {len(pairs)} pairs → "
                       f"{survivors_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
