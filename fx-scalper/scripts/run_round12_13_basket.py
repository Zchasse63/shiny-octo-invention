"""Rounds 12+13 — G10 basket trend-following with volatility-targeted sizing.

Per research review consensus (Carver, 2022 CTA Index, cross-paper
evidence): the evidence-based edge for retail FX is NOT scalping on
single pairs but running a simple trend primitive across a diversified
basket with vol-targeted sizing. This script tests exactly that.

Design:

  1. Load all G10 pairs at D1. Resample consistently.
  2. Apply a single trend rule (MA 8/32 EMA crossover — the Round 10
     anomaly that survived friction 2x) to EVERY pair simultaneously.
  3. Use ``cash_sharing=True, group_by=True`` and ``size_type='value'``
     so each pair trade commits fixed $notional. Vol-targeted sizing:
     each pair gets size scaled by 1 / realized_vol_60d so all
     positions contribute equal risk.
  4. Report per-pair PF, combined portfolio stats, and per-bar return
     correlation across pairs.
  5. Apply BCa bootstrap + Deflated Sharpe on the COMBINED return
     stream. Key insight: when we test ONE rule across ten pairs,
     n_trials is 10, not 30000 — DSR passes dramatically lower bar.

This is the "one rule, many pairs" test that maximizes independent
evidence per computation.
"""

from __future__ import annotations

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
from src.backtest.statistics import bca_bootstrap_pf, deflated_sharpe_ratio  # noqa: E402
from src.indicators.engine import add_atr  # noqa: E402
from src.strategies.exits import ExitConfig, config_to_vbt_params  # noqa: E402
from src.strategies.families import (  # noqa: E402
    DonchianBreakoutFamily,
    DonchianBreakoutParams,
    MACrossoverFamily,
    MACrossoverParams,
)
from src.utils.logger import get_logger, init_logger  # noqa: E402


# All pairs we aim to include. Will fall back to whatever's on disk.
G10_PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "NZD_USD",
    "USD_CHF", "USD_CAD", "EUR_GBP", "EUR_JPY",
]


def _have_data(pair: str) -> bool:
    return (PROJECT_ROOT / "data" / "processed" / pair).exists()


def _load_pair(pair: str, tf: str) -> pd.DataFrame | None:
    try:
        m1 = load_symbol_bars(pair, start="2023-01-01", end="2026-04-20")
        if m1.empty:
            return None
        return resample_bars(m1, tf)
    except Exception:
        return None


def _vol_targeted_size(
    close: pd.Series, target_notional: float, vol_lookback: int = 60
) -> pd.Series:
    """Size each trade inversely to realized volatility.

    Vol-target of, say, 10% annualized. If current realized vol is 20%,
    downscale by half. If 5%, upscale by 2x. This equalizes risk
    contribution across pairs in the basket.
    """
    # Daily returns, annualized vol.
    rets = close.pct_change()
    vol = rets.rolling(vol_lookback, min_periods=20).std() * np.sqrt(252)
    # Target 10% annualized vol per position.
    target_vol = 0.10
    scale = (target_vol / vol).clip(upper=5.0, lower=0.2)  # cap leverage
    return (target_notional * scale).fillna(target_notional)


def run_basket_one_rule(
    family_name: str,
    tf: str,
    params: dict,
    pairs: list[str],
    *,
    vol_target: bool = False,
) -> dict:
    """Run a single trend rule across all pairs simultaneously."""
    logger = get_logger(__name__)
    family_cls = {
        "donchian_breakout": DonchianBreakoutFamily,
        "ma_crossover": MACrossoverFamily,
    }[family_name]

    # Step 1: load all bars, resample.
    bars_by_pair: dict[str, pd.DataFrame] = {}
    for p in pairs:
        if not _have_data(p):
            continue
        bars = _load_pair(p, tf)
        if bars is None or len(bars) < 200:
            continue
        bars_by_pair[p] = bars
    if not bars_by_pair:
        return {"error": "no pair data available"}
    logger.info(f"Pairs with data at {tf}: {list(bars_by_pair.keys())}")

    # Step 2: generate signals per pair, build aligned matrices.
    close_frames, entries_long_frames, entries_short_frames = {}, {}, {}
    tsl_frames, size_frames = {}, {}
    for p, bars in bars_by_pair.items():
        close_col = "mid_close" if "mid_close" in bars.columns else "bid_close"
        close = bars[close_col].rename(p)
        atr = add_atr(bars, length=14)["atr_14"]
        fam = family_cls(family_cls.params_cls(**params))
        sigs = fam.generate(bars)
        exit_cfg = ExitConfig(
            sl_atr_mult=2.0, atr_length=14, tp_r_mult=None,
            trail_kind="chandelier", trail_atr_mult=3.0,
        )
        vp = config_to_vbt_params(sigs.entries_long, sigs.entries_short, close, atr, exit_cfg)
        close_frames[p] = close
        entries_long_frames[p] = sigs.entries_long.rename(p)
        entries_short_frames[p] = sigs.entries_short.rename(p)
        tsl_frames[p] = vp.trail_distance_pct.fillna(0.0).rename(p) if vp.trail_distance_pct is not None else pd.Series(0.0, index=close.index, name=p)
        # Size per pair — vol-targeted or flat $5K notional
        if vol_target:
            size_frames[p] = _vol_targeted_size(close, 5000.0).rename(p)
        else:
            size_frames[p] = pd.Series(5000.0, index=close.index, name=p)

    # Step 3: align all into one matrix (union of indexes).
    close_mat = pd.concat(list(close_frames.values()), axis=1).ffill()
    entries_long_mat = pd.concat(list(entries_long_frames.values()), axis=1).fillna(False).astype(bool)
    entries_short_mat = pd.concat(list(entries_short_frames.values()), axis=1).fillna(False).astype(bool)
    tsl_mat = pd.concat(list(tsl_frames.values()), axis=1).fillna(0.0)
    size_mat = pd.concat(list(size_frames.values()), axis=1).fillna(5000.0)

    # Build slippage mat
    slip_frames = []
    for p, bars in bars_by_pair.items():
        if "bid_close" in bars.columns and "ask_close" in bars.columns:
            slip = ((bars["ask_close"] - bars["bid_close"]) / 2.0).rename(p)
        else:
            slip = pd.Series(0.0, index=bars.index, name=p)
        slip_frames.append(slip)
    slip_mat = pd.concat(slip_frames, axis=1).fillna(0.0)
    # Reindex all to match close_mat
    entries_long_mat = entries_long_mat.reindex(close_mat.index, fill_value=False)
    entries_short_mat = entries_short_mat.reindex(close_mat.index, fill_value=False)
    tsl_mat = tsl_mat.reindex(close_mat.index, fill_value=0.0)
    size_mat = size_mat.reindex(close_mat.index, fill_value=5000.0)
    slip_mat = slip_mat.reindex(close_mat.index, fill_value=0.0)

    logger.info(f"Aligned: close={close_mat.shape} entries_long sum={int(entries_long_mat.to_numpy().sum())}")

    # Step 4: run as ONE portfolio with cash_sharing + group_by
    pf = vbt.Portfolio.from_signals(
        close=close_mat,
        entries=entries_long_mat,
        short_entries=entries_short_mat,
        tsl_stop=tsl_mat.to_numpy(),
        size=size_mat.to_numpy() if vol_target else 5000.0,
        size_type="value",
        leverage=50,
        init_cash=5000.0,
        cash_sharing=True,
        group_by=True,
        freq=tf,
        fees=0.0,
        slippage=slip_mat.to_numpy(),
    )

    # Group-level stats
    n_trades = int(pf.trades.count())
    total_ret = float(pf.total_return)
    max_dd = float(abs(pf.max_drawdown))
    final_eq = float(pf.final_value)

    # Per-column PF via readable trade records (cash_sharing=True
    # doesn't allow regrouping; use trade records['Column'] to partition)
    all_recs = pf.trades.records_readable
    per_pair_stats = []
    for i, col in enumerate(close_mat.columns):
        # vbt indexes columns 0-based; records have 'Column' field with ints
        col_recs = all_recs[all_recs["Column"] == i]
        if len(col_recs) < 5:
            continue
        pnls = col_recs["PnL"].to_numpy()
        wins = pnls[pnls > 0].sum()
        losses = abs(pnls[pnls < 0].sum())
        pfr = float(wins / losses) if losses > 0 else float("inf")
        per_pair_stats.append({
            "pair": col,
            "n_trades": len(col_recs),
            "pf": pfr,
            "win_rate": float((pnls > 0).mean()),
            "expectancy": float(pnls.mean()),
        })

    # Combined trade-stream PnL for BCa + DSR
    all_pnls = all_recs["PnL"].to_numpy() if len(all_recs) else np.array([])

    stats: dict = {
        "family": family_name, "tf": tf, "params": params,
        "pairs_tested": list(close_mat.columns),
        "n_pairs": len(close_mat.columns),
        "group_n_trades": n_trades,
        "group_total_return": total_ret,
        "group_max_dd": max_dd,
        "group_final_equity": final_eq,
        "group_sharpe": float(pf.sharpe_ratio),
        "group_sortino": float(pf.sortino_ratio) if not np.isnan(float(pf.sortino_ratio)) else None,
        "per_pair": per_pair_stats,
    }

    if len(all_pnls) >= 30:
        bca = bca_bootstrap_pf(all_pnls, n_resamples=5000)
        wins = all_pnls[all_pnls > 0].sum()
        losses = abs(all_pnls[all_pnls < 0].sum())
        stats["combined_pf_point"] = float(wins/losses) if losses > 0 else float("inf")
        stats["combined_pf_bca_lo"] = bca.ci_lower
        stats["combined_pf_bca_hi"] = bca.ci_upper
        mu, sigma = float(np.mean(all_pnls)), float(np.std(all_pnls, ddof=1))
        trades_per_year = len(all_pnls) / 3.3
        sharpe = (mu/sigma*np.sqrt(trades_per_year)) if sigma > 0 else 0.0
        from scipy import stats as _st
        skew = float(_st.skew(all_pnls)) if len(all_pnls) >= 3 else 0.0
        kurt = float(_st.kurtosis(all_pnls, fisher=False)) if len(all_pnls) >= 4 else 3.0
        dsr = deflated_sharpe_ratio(
            sharpe, n_trials=len(close_mat.columns),  # 1 rule × N pairs = N effective tests
            n_observations=len(all_pnls), skew=skew, kurt=kurt,
        )
        stats["combined_sharpe_trade"] = sharpe
        stats["combined_dsr_z"] = dsr["dsr_z"]
        stats["combined_dsr_prob"] = dsr["dsr_prob"]
        stats["combined_expected_max_sharpe"] = dsr["expected_max_sharpe"]

    return stats


def main() -> int:
    init_logger()
    logger = get_logger(__name__)

    pairs_available = [p for p in G10_PAIRS if _have_data(p)]
    logger.info(f"G10 pairs with data: {pairs_available}")

    test_configs = [
        # The Round 10 D1 anomaly
        ("ma_crossover", "1D", {"fast_length": 8, "slow_length": 32, "ma_type": "ema"}),
        # Classic Carver speed buckets
        ("ma_crossover", "1D", {"fast_length": 16, "slow_length": 64, "ma_type": "ema"}),
        ("ma_crossover", "1D", {"fast_length": 32, "slow_length": 128, "ma_type": "ema"}),
        # H4 variants
        ("ma_crossover", "4H", {"fast_length": 16, "slow_length": 64, "ma_type": "ema"}),
        ("ma_crossover", "4H", {"fast_length": 32, "slow_length": 128, "ma_type": "ema"}),
        # Donchian classics
        ("donchian_breakout", "1D", {"entry_lookback": 20, "exit_lookback": 10, "use_both_sides": True}),
        ("donchian_breakout", "1D", {"entry_lookback": 40, "exit_lookback": 20, "use_both_sides": True}),
        ("donchian_breakout", "4H", {"entry_lookback": 60, "exit_lookback": 30, "use_both_sides": True}),
    ]

    results = []
    for family_name, tf, params in test_configs:
        logger.info(f"--- {family_name} {tf} {params} ---")
        stats = run_basket_one_rule(family_name, tf, params, pairs_available)
        if "error" in stats:
            continue
        results.append(stats)

    if not results:
        logger.error("No results")
        return 1

    # Flat summary DataFrame
    rows = []
    for s in results:
        rows.append({
            "family": s["family"],
            "tf": s["tf"],
            "params": str(s["params"]),
            "n_pairs": s["n_pairs"],
            "n_trades": s["group_n_trades"],
            "total_return": s["group_total_return"],
            "max_dd": s["group_max_dd"],
            "sharpe_group": s["group_sharpe"],
            "pf_combined": s.get("combined_pf_point"),
            "pf_bca_lo": s.get("combined_pf_bca_lo"),
            "pf_bca_hi": s.get("combined_pf_bca_hi"),
            "dsr_prob": s.get("combined_dsr_prob"),
            "expected_max_sharpe": s.get("combined_expected_max_sharpe"),
        })
    df = pd.DataFrame(rows)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M")
    out_dir = PROJECT_ROOT / "backtest_results" / f"round12_13_basket_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "basket_summary.csv", index=False)

    # Per-pair detail dump
    per_pair_rows = []
    for s in results:
        for pp in s.get("per_pair", []):
            per_pair_rows.append({
                "family": s["family"], "tf": s["tf"], "params": str(s["params"]), **pp,
            })
    per_pair_df = pd.DataFrame(per_pair_rows)
    per_pair_df.to_csv(out_dir / "per_pair_detail.csv", index=False)

    print()
    print("=" * 110)
    print(f"Rounds 12+13 — G10 basket trend-following ({len(pairs_available)} pairs × one rule, cash_sharing)")
    print("=" * 110)
    print(df.round(3).to_string(index=False))
    print()
    print("Per-pair detail (summary):")
    if not per_pair_df.empty:
        summary = per_pair_df.groupby(["family", "tf", "pair"])[["pf", "n_trades"]].mean().round(2)
        print(summary.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
