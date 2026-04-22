"""Round 6 — SL-width ablation + corrected position sizing.

Two jobs combined into one round per the round-4 / round-7 diagnostics:

 1. POSITION-SIZING FIX. Earlier rounds passed no size= to vbt and let
    it default to full-equity × leverage, inflating dollar expectancy
    and compounding losses destructively. This round uses
    ``size=5000, size_type='value', leverage=50`` so each trade commits
    exactly $100 margin / $5,000 notional per the CLAUDE.md spec.

 2. SL-WIDTH ABLATION. Round 7's per-trade MAE/MFE showed the 0.5× ATR
    stop is inside the MAE p25 → ~35-40% of winning signals are getting
    stopped out. Sweep sl_atr_mult ∈ {0.5, 0.75, 1.0, 1.25, 1.5} on
    the top-5 round-5 configs, across EUR/USD + GBP/USD. Use vbt.Param
    so all widths run in one Portfolio.from_signals call (canonical
    pattern confirmed in vbt.chat round-6 artifact).

Output: ``backtest_results/sl_ablation_YYYYMMDDTHHMM/`` with
 - ``sl_ablation_eurusd.csv``
 - ``sl_ablation_gbpusd.csv``
 - ``sl_ablation_summary.csv`` — pivot by (config_id, sl_atr_mult, pair)
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


SL_MULTS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
PAIRS = ["EUR_USD", "GBP_USD"]  # USD/JPY deferred per round-4b finding
ROUND5_CSV = "backtest_results/explore_multi_tf_20260422T0026/combined_results.csv"


def _top_configs(
    csv_path: Path, top_n: int = 5, *, filtered_only: bool = True
) -> list[dict]:
    """Return the top-N round-5 configs.

    ``filtered_only=True`` restricts to ``bb_rsi_mr_filtered`` /
    ``rsi_extreme_filtered`` — the session+weekday-filtered variants
    that survived GBP/USD cross-pair validation with corrected sizing.
    """
    df = pd.read_csv(csv_path)
    oos = df[(df["kind"] == "OOS") & (df["total_trades"] >= 30)]
    if filtered_only:
        oos = oos[oos["family"].str.endswith("_filtered")]
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
    top = full.sort_values("pf", ascending=False).head(top_n)
    return top.to_dict("records")


def _half_spread_slippage(bars: pd.DataFrame) -> np.ndarray:
    if "bid_close" in bars.columns and "ask_close" in bars.columns:
        return ((bars["ask_close"] - bars["bid_close"]) / 2.0).to_numpy()
    return np.zeros(len(bars), dtype=float)


def _run_ablation_for_config(
    config: dict, pair: str, bars: pd.DataFrame
) -> pd.DataFrame:
    """Run the SL-width ablation for one config on one pair.

    Uses vbt.Param to sweep sl_atr_mult across SL_MULTS in a single
    Portfolio.from_signals call. Returns a DataFrame with per-mult
    PF / WR / expectancy / max_dd / trades.
    """
    family_cls = get_family_by_name(config["family"])
    if family_cls is None:
        return pd.DataFrame()
    family_params = json.loads(config["family_params"])
    exit_dict = json.loads(config["exit_config"])

    # Build signals.
    family = family_cls(family_cls.params_cls(**family_params))
    signals = family.generate(bars)
    close_col = "mid_close" if "mid_close" in bars.columns else "bid_close"
    close = bars[close_col]
    atr_len = int(exit_dict.get("atr_length", 14))
    atr = add_atr(bars, length=atr_len)[f"atr_{atr_len}"]

    # Build vbt_params once to extract tp and trail fractions. We will
    # OVERRIDE sl_stop via vbt.Param.
    exit_cfg = ExitConfig(
        sl_atr_mult=exit_dict["sl_atr_mult"],  # original — will be replaced
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

    use_trail = bool(vbt_params.sl_trail and vbt_params.trail_distance_pct is not None)

    # Build sl_stop arrays for each multiplier. Per vbt.chat round-6
    # guidance, stops should NOT be masked to entry bars — they need a
    # value at every timestamp. We use (mult * atr / close) everywhere
    # (nan→0.0 just for safety), and vbt picks up the value at each
    # entry bar. Since original sl_frac (= sl_atr_mult*atr/close masked
    # to entries) gave correct results in rounds 1-5, keeping the same
    # computation at every timestamp is a superset.
    sl_arrays = []
    for mult in SL_MULTS:
        arr = (mult * atr / close).fillna(0.0).to_numpy()
        sl_arrays.append(arr)

    slippage = _half_spread_slippage(bars)
    # Mask tp / trail to entries as before for fidelity with round 5.
    tp_array = vbt_params.tp_stop.fillna(0.0).to_numpy()
    kwargs: dict = {
        "close": close,
        "entries": signals.entries_long,
        "short_entries": signals.entries_short,
        "tp_stop": tp_array,
        "size": 5000.0,
        "size_type": "value",
        "leverage": 50,
        "init_cash": 500.0,
        "freq": bars.index.freq or "15min",
        "fees": 0.0,
        "slippage": slippage,
    }

    # SL (parameterized) vs trail: trail takes precedence if set.
    if use_trail:
        # Trail uses the same mult×atr distance expressed per-bar.
        trail_arrays = []
        for mult in SL_MULTS:
            arr = (mult * atr / close).fillna(0.0).to_numpy()
            trail_arrays.append(arr)
        kwargs["tsl_stop"] = vbt.Param(
            trail_arrays,
            keys=pd.Index(SL_MULTS, name="sl_atr_mult"),
        )
    else:
        kwargs["sl_stop"] = vbt.Param(
            sl_arrays,
            keys=pd.Index(SL_MULTS, name="sl_atr_mult"),
        )

    try:
        pf = vbt.Portfolio.from_signals(**kwargs)
    except Exception as e:
        logger = get_logger(__name__)
        logger.warning(f"from_signals failed ({config['family']} on {pair}): {e}")
        return pd.DataFrame()

    # Extract per-column metrics. vbt.Param produces columns indexed by
    # the param keys, which we read via pf.trades.count() etc.
    n_trades = pf.trades.count()
    win_rate = pf.trades.win_rate
    pnl_sum_wins = pf.trades.winning.pnl.sum()
    pnl_sum_losses = pf.trades.losing.pnl.sum().abs()
    pf_per_col = pnl_sum_wins / pnl_sum_losses.replace(0, np.nan)
    expectancy = pf.trades.pnl.mean()
    max_dd = pf.max_drawdown.abs()

    # Column index is the sl_atr_mult param keys.
    out = pd.DataFrame({
        "sl_atr_mult": SL_MULTS,
        "n_trades": n_trades.values if hasattr(n_trades, "values") else [n_trades],
        "win_rate": win_rate.values if hasattr(win_rate, "values") else [win_rate],
        "profit_factor": pf_per_col.values if hasattr(pf_per_col, "values") else [pf_per_col],
        "expectancy_usd": expectancy.values if hasattr(expectancy, "values") else [expectancy],
        "max_dd": max_dd.values if hasattr(max_dd, "values") else [max_dd],
    })
    out["timeframe"] = config["timeframe"]
    out["family"] = config["family"]
    out["original_pf"] = config["pf"]
    out["pair"] = pair
    out["family_params"] = config["family_params"]
    out["exit_config"] = config["exit_config"]
    return out


def main() -> int:
    init_logger()
    logger = get_logger(__name__)

    configs = _top_configs(PROJECT_ROOT / ROUND5_CSV, top_n=5)
    logger.info(f"Loaded {len(configs)} top round-5 configs")
    for i, c in enumerate(configs, 1):
        logger.info(f"  {i}. {c['family']:25s} {c['timeframe']:6s} pf={c['pf']:.2f}")

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M")
    out_dir = PROJECT_ROOT / "backtest_results" / f"sl_ablation_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[pd.DataFrame] = []
    bars_cache: dict[tuple[str, str], pd.DataFrame] = {}
    for pair in PAIRS:
        logger.info(f"Loading {pair}")
        m1 = load_symbol_bars(pair, start="2023-01-01", end="2026-04-20")
        if m1.empty:
            logger.warning(f"no data for {pair}")
            continue
        for c in configs:
            tf = c["timeframe"]
            key = (pair, tf)
            if key not in bars_cache:
                bars_cache[key] = resample_bars(m1, tf)
            bars = bars_cache[key]
            logger.info(f"  [{pair} {c['family']} {tf}] sweeping SL_MULTS={SL_MULTS}")
            df = _run_ablation_for_config(c, pair, bars)
            if not df.empty:
                all_results.append(df)

    if not all_results:
        logger.error("No results produced")
        return 1
    combined = pd.concat(all_results, ignore_index=True)
    combined_path = out_dir / "sl_ablation_all.csv"
    combined.to_csv(combined_path, index=False)
    logger.info(f"Wrote {len(combined)} rows → {combined_path}")

    # Summary pivot: mean PF by (timeframe, family, sl_atr_mult, pair)
    summary = (
        combined.groupby(["timeframe", "family", "sl_atr_mult", "pair"])
        [["profit_factor", "win_rate", "expectancy_usd", "max_dd", "n_trades"]]
        .mean()
        .reset_index()
    )
    summary.to_csv(out_dir / "sl_ablation_summary.csv", index=False)

    # Headline: PF by SL multiplier, averaged across top-5 configs per pair.
    print()
    print("=" * 76)
    print("Round 6 — SL-width ablation (PF averaged across top-5 configs, per pair)")
    print("=" * 76)
    hi = (
        combined.groupby(["pair", "sl_atr_mult"])
        [["profit_factor", "win_rate", "expectancy_usd", "max_dd", "n_trades"]]
        .mean().round(3)
    )
    print(hi.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
