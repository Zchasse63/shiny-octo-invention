"""Round 4b — cross-pair validation for MOMENTUM families only.

Round 4's MR-heavy candidate set failed catastrophically on USD/JPY (0/40
configs PF>0 — JPY is in a trending BoJ-cycle regime where mean reversion
breaks). This follow-on tests the top momentum / breakout configs from
round 5 specifically to see whether a momentum strategy survives on JPY
AND also behaves reasonably on EUR/USD + GBP/USD.

If a range_breakout / ema_cross config survives all 3 pairs, it's a
candidate for the "JPY track" of a per-pair portfolio.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


MOMENTUM_FAMILIES = {"range_breakout", "ema_cross", "pullback_ema"}
PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY"]
ROUND5_CSV = "backtest_results/explore_multi_tf_20260422T0026/combined_results.csv"


def _top_momentum_configs(csv_path: Path, top_n: int = 20) -> list[dict]:
    df = pd.read_csv(csv_path)
    oos = df[
        (df["kind"] == "OOS")
        & (df["family"].isin(MOMENTUM_FAMILIES))
        & (df["total_trades"] >= 30)
    ]
    agg = (
        oos.groupby(["timeframe", "family", "family_params", "exit_config"])
        .agg(
            pf=("profit_factor", "mean"),
            exp=("expectancy_usd", "mean"),
            n=("split", "count"),
            min_pf=("profit_factor", "min"),
            min_trades=("total_trades", "min"),
        )
        .reset_index()
    )
    # Purpose of round 4b is to see if ANY momentum config survives JPY,
    # so filter loosely: mean PF > 1.0, positive expectancy, min trades >= 10.
    # In the round-5 EUR/USD sweep no momentum config has a robust WFA edge
    # (best mean PF is 1.08), so we take the most marginal as candidates
    # and see whether they excel on GBP/JPY instead.
    full = agg[
        (agg["n"] == 3) & (agg["pf"] >= 1.0)
        & (agg["exp"] > 0) & (agg["min_trades"] >= 10)
    ]
    top = full.sort_values("pf", ascending=False).head(top_n)
    out = []
    for _, r in top.iterrows():
        out.append({
            "timeframe": r["timeframe"],
            "family": r["family"],
            "family_params": r["family_params"],
            "exit_config": r["exit_config"],
            "original_pf": r["pf"],
            "original_exp": r["exp"],
        })
    return out


def _half_spread_slippage(bars: pd.DataFrame):
    import numpy as np
    if "bid_close" in bars.columns and "ask_close" in bars.columns:
        return ((bars["ask_close"] - bars["bid_close"]) / 2.0).to_numpy()
    return np.zeros(len(bars), dtype=float)


def _rerun_config(config: dict, bars: pd.DataFrame) -> pd.DataFrame:
    family_params = json.loads(config["family_params"])
    exit_dict = json.loads(config["exit_config"])
    family_cls = next((f for f in ALL_FAMILIES if f.name == config["family"]), None)
    if family_cls is None:
        return pd.DataFrame()
    try:
        family = _instantiate(family_cls(), family_params)
    except Exception:
        return pd.DataFrame()

    signals = family.generate(bars)
    close_col = "mid_close" if "mid_close" in bars.columns else "bid_close"
    close = bars[close_col]

    from src.indicators.engine import add_atr
    atr_len = exit_dict.get("atr_length", 14)
    atr = add_atr(bars, length=atr_len)[f"atr_{atr_len}"]

    exit_cfg = ExitConfig(
        sl_atr_mult=exit_dict["sl_atr_mult"],
        atr_length=exit_dict["atr_length"],
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
    slippage = _half_spread_slippage(bars)
    splits = _walk_forward_slices(close, 3, 0.5)

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
            rows.append({"split": split_label, "kind": kind, **metrics.as_dict()})
    return pd.DataFrame(rows)


def main() -> int:
    init_logger()
    logger = get_logger(__name__)

    configs = _top_momentum_configs(PROJECT_ROOT / ROUND5_CSV, top_n=20)
    logger.info(f"Loaded {len(configs)} momentum configs from round 5")
    if not configs:
        logger.error("No momentum configs passed gates; round 5 grid is MR-heavy")
        return 1

    print("\nTop momentum candidates (round-5 OOS):")
    for i, c in enumerate(configs[:15], 1):
        print(f"  {i:2d}. {c['family']:16s} {c['timeframe']:6s}  "
              f"PF {c['original_pf']:.2f}  exp ${c['original_exp']:.2f}")

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M")
    out_dir = PROJECT_ROOT / "backtest_results" / f"cross_pair_momentum_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    bars_cache: dict[tuple[str, str], pd.DataFrame] = {}
    results: list[dict] = []
    for pair in PAIRS:
        logger.info(f"Loading {pair}")
        m1 = load_symbol_bars(pair, start="2023-01-01", end="2026-04-20")
        if m1.empty:
            continue
        for c in configs:
            key = (pair, c["timeframe"])
            if key not in bars_cache:
                bars_cache[key] = resample_bars(m1, c["timeframe"])  # type: ignore[arg-type]
            splits_df = _rerun_config(c, bars_cache[key])
            if splits_df.empty:
                continue
            oos = splits_df[splits_df["kind"] == "OOS"]
            if oos.empty:
                continue
            results.append({
                "pair": pair,
                "timeframe": c["timeframe"],
                "family": c["family"],
                "family_params": c["family_params"],
                "exit_config": c["exit_config"],
                "original_pf": c["original_pf"],
                "oos_pf": float(oos["profit_factor"].mean()),
                "oos_exp": float(oos["expectancy_usd"].mean()),
                "oos_wr": float(oos["win_rate"].mean()),
                "oos_trades": float(oos["total_trades"].mean()),
                "oos_dd": float(oos["max_drawdown_pct"].mean()),
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "cross_pair_momentum_results.csv", index=False)
    pivot = results_df.pivot_table(
        index=["timeframe", "family", "family_params", "exit_config"],
        columns="pair",
        values="oos_pf",
    ).reset_index()
    pivot.to_csv(out_dir / "cross_pair_momentum_pivot.csv", index=False)

    logger.info(f"Wrote {len(results_df)} rows → {out_dir}")

    if {"EUR_USD", "GBP_USD", "USD_JPY"}.issubset(pivot.columns):
        mask = (pivot[["EUR_USD", "GBP_USD", "USD_JPY"]] > 1.0).all(axis=1)
        survivors = pivot[mask]
        print()
        print("=" * 76)
        print(f"Round 4b — momentum cross-pair: {len(survivors)}/{len(pivot)} "
              f"configs PF>1.0 on ALL 3 pairs")
        print("=" * 76)
        if not survivors.empty:
            print(survivors[["timeframe", "family",
                             "EUR_USD", "GBP_USD", "USD_JPY"]].round(2)
                  .to_string(index=False))
        else:
            print("(no survivors; showing JPY-only winners below)")
            jpy_ok = pivot[pivot["USD_JPY"] > 1.0].sort_values(
                "USD_JPY", ascending=False)
            print(jpy_ok[["timeframe", "family",
                          "EUR_USD", "GBP_USD", "USD_JPY"]].head(10).round(2)
                  .to_string(index=False))
        survivors.to_csv(out_dir / "cross_pair_momentum_survivors.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
