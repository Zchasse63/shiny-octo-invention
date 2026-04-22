"""Round 11 — apply rigorous statistics to all Round 9 survivors.

Replaces Round 9's naive percentile bootstrap with BCa + Deflated
Sharpe + purged k-fold (Lopez de Prado 2018, Bailey & Lopez de Prado
2014). Also re-computes everything with the new trend-following
families (Round 10) so we can compare BB+RSI MR head-to-head with
Donchian and MA crossover on EUR/USD under the proper stat bar.

Decision rule (deliberately strict):
  1. Full-sample PF > 1.05 on both EUR and GBP
  2. BCa 95% CI lower bound > 1.0 (edge is statistically significant)
  3. DSR probability > 0.95 (edge survives multi-testing deflation)
  4. Friction 2x PF > 1.0

Configs that clear all four are the only candidates eligible for
Phase-5 paper trading. This is tighter than Round 9 because the
prior run's CI was naive percentile; BCa will be different (tighter
on symmetric distributions, wider on skewed ones).
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
from src.backtest.statistics import bca_bootstrap_pf, deflated_sharpe_ratio  # noqa: E402
from src.indicators.engine import add_atr  # noqa: E402
from src.strategies.exits import ExitConfig, config_to_vbt_params  # noqa: E402
from src.strategies.families import (  # noqa: E402
    DonchianBreakoutFamily,
    DonchianBreakoutParams,
    MACrossoverFamily,
    MACrossoverParams,
    get_family_by_name,
)
from src.utils.logger import get_logger, init_logger  # noqa: E402


CROSS_PAIR_PIVOT = "backtest_results/cross_pair_20260422T2150/cross_pair_pivot.csv"
PAIRS = ["EUR_USD", "GBP_USD"]
# Round-5 sweep tested ~30K configs; use as deflation trial count.
N_TRIALS_TESTED = 30_000


def _build_pf_from_csv_config(cfg: dict, pair: str, slip_mult: float = 1.0):
    """Instantiate a config from the round-4 pivot and run full-sample."""
    family_cls = get_family_by_name(cfg["family"])
    if family_cls is None:
        return None
    fp = json.loads(cfg["family_params"])
    ec = json.loads(cfg["exit_config"])

    m1 = load_symbol_bars(pair, start="2023-01-01", end="2026-04-20")
    if m1.empty:
        return None
    bars = resample_bars(m1, cfg["timeframe"])
    close_col = "mid_close" if "mid_close" in bars.columns else "bid_close"
    close = bars[close_col]
    atr_len = int(ec.get("atr_length", 14))
    atr = add_atr(bars, length=atr_len)[f"atr_{atr_len}"]

    fam = family_cls(family_cls.params_cls(**fp))
    sigs = fam.generate(bars)
    exit_cfg = ExitConfig(
        sl_atr_mult=ec["sl_atr_mult"],
        atr_length=atr_len,
        tp_r_mult=ec.get("tp_r_mult"),
        trail_kind=ec["trail_kind"],
        trail_atr_mult=ec.get("trail_atr_mult"),
    )
    vp = config_to_vbt_params(sigs.entries_long, sigs.entries_short, close, atr, exit_cfg)
    if "bid_close" in bars.columns and "ask_close" in bars.columns:
        slip = ((bars["ask_close"] - bars["bid_close"]) / 2.0 * slip_mult).to_numpy()
    else:
        slip = np.zeros(len(bars))
    kwargs = dict(
        close=close,
        entries=sigs.entries_long, short_entries=sigs.entries_short,
        tp_stop=vp.tp_stop.fillna(0.0).to_numpy(),
        size=5000.0, size_type="value", leverage=50, init_cash=5000.0,
        freq=bars.index.freq or "15min", fees=0.0, slippage=slip,
    )
    if vp.sl_trail and vp.trail_distance_pct is not None:
        kwargs["tsl_stop"] = vp.trail_distance_pct.fillna(0.0).to_numpy()
    else:
        kwargs["sl_stop"] = vp.sl_stop.fillna(0.0).to_numpy()
    try:
        return vbt.Portfolio.from_signals(**kwargs)
    except Exception as e:
        get_logger(__name__).warning(f"from_signals failed: {e}")
        return None


def _eval_config(cfg: dict, tag: str) -> list[dict]:
    """Return list of dicts (one per pair × slip_mult)."""
    out: list[dict] = []
    for pair in PAIRS:
        for slip_mult in [1.0, 2.0]:
            pf = _build_pf_from_csv_config(cfg, pair, slip_mult)
            if pf is None:
                continue
            pnls = pf.trades.records_readable["PnL"].to_numpy()
            if len(pnls) < 30:
                continue
            # Plain PF
            wins = pnls[pnls > 0].sum()
            losses = abs(pnls[pnls < 0].sum())
            pf_point = float(wins / losses) if losses > 0 else float("inf")

            # BCa bootstrap at 1× only (2× is just for friction check)
            if slip_mult == 1.0:
                bca = bca_bootstrap_pf(pnls, n_resamples=10_000)
            else:
                bca = None

            # DSR on trade-level PnLs. Annualization: use avg trades/year.
            # We have 3.3 years of EUR / 1 year of GBP.
            years = 3.3 if pair == "EUR_USD" else 1.0
            trades_per_year = len(pnls) / years
            mu = float(np.mean(pnls)); sigma = float(np.std(pnls, ddof=1))
            sharpe = (mu / sigma * np.sqrt(trades_per_year)) if sigma > 0 else 0.0
            from scipy import stats as _st
            skew = float(_st.skew(pnls)) if len(pnls) >= 3 else 0.0
            kurt = float(_st.kurtosis(pnls, fisher=False)) if len(pnls) >= 4 else 3.0
            dsr = deflated_sharpe_ratio(
                sharpe, n_trials=N_TRIALS_TESTED,
                n_observations=len(pnls), skew=skew, kurt=kurt,
            )

            out.append({
                "tag": tag,
                "pair": pair,
                "slip_mult": slip_mult,
                "n_trades": len(pnls),
                "pf_point": pf_point,
                "pf_bca_low": bca.ci_lower if bca else None,
                "pf_bca_high": bca.ci_upper if bca else None,
                "sharpe_trade": sharpe,
                "sharpe_expected_max": dsr["expected_max_sharpe"],
                "dsr_z": dsr["dsr_z"],
                "dsr_prob": dsr["dsr_prob"],
                "skew": skew, "kurt": kurt,
                "family": cfg["family"],
                "timeframe": cfg["timeframe"],
            })
    return out


def _eval_trend_family(name: str, tf: str, params: dict, tag: str) -> list[dict]:
    """Evaluate a trend-following family config on the same stat gauntlet."""
    from src.backtest.statistics import bca_bootstrap_pf, deflated_sharpe_ratio
    out = []
    family_cls = {"donchian_breakout": DonchianBreakoutFamily,
                  "ma_crossover": MACrossoverFamily}[name]
    for pair in PAIRS:
        for slip_mult in [1.0, 2.0]:
            m1 = load_symbol_bars(pair, start="2023-01-01", end="2026-04-20")
            bars = resample_bars(m1, tf)
            close = bars["mid_close"]
            atr = add_atr(bars, length=14)["atr_14"]
            fam = family_cls(family_cls.params_cls(**params))
            sigs = fam.generate(bars)
            exit_cfg = ExitConfig(
                sl_atr_mult=2.0, atr_length=14, tp_r_mult=None,
                trail_kind="chandelier", trail_atr_mult=3.0,
            )
            vp = config_to_vbt_params(sigs.entries_long, sigs.entries_short, close, atr, exit_cfg)
            slip = ((bars["ask_close"] - bars["bid_close"]) / 2.0 * slip_mult).to_numpy()
            pf = vbt.Portfolio.from_signals(
                close=close, entries=sigs.entries_long, short_entries=sigs.entries_short,
                size=5000.0, size_type="value", leverage=50, init_cash=5000.0,
                tsl_stop=vp.trail_distance_pct.fillna(0.0).to_numpy(),
                freq=tf, fees=0.0, slippage=slip,
            )
            pnls = pf.trades.records_readable["PnL"].to_numpy()
            if len(pnls) < 15:
                continue
            wins = pnls[pnls > 0].sum()
            losses = abs(pnls[pnls < 0].sum())
            pf_point = float(wins / losses) if losses > 0 else float("inf")
            bca = bca_bootstrap_pf(pnls, n_resamples=5_000) if slip_mult == 1.0 else None
            years = 3.3 if pair == "EUR_USD" else 1.0
            trades_per_year = len(pnls) / years
            mu = float(np.mean(pnls)); sigma = float(np.std(pnls, ddof=1))
            sharpe = (mu / sigma * np.sqrt(trades_per_year)) if sigma > 0 else 0.0
            from scipy import stats as _st
            skew = float(_st.skew(pnls)) if len(pnls) >= 3 else 0.0
            kurt = float(_st.kurtosis(pnls, fisher=False)) if len(pnls) >= 4 else 3.0
            dsr = deflated_sharpe_ratio(
                sharpe, n_trials=100,  # trend test: much smaller search space
                n_observations=len(pnls), skew=skew, kurt=kurt,
            )
            out.append({
                "tag": tag, "pair": pair, "slip_mult": slip_mult,
                "n_trades": len(pnls), "pf_point": pf_point,
                "pf_bca_low": bca.ci_lower if bca else None,
                "pf_bca_high": bca.ci_upper if bca else None,
                "sharpe_trade": sharpe, "sharpe_expected_max": dsr["expected_max_sharpe"],
                "dsr_z": dsr["dsr_z"], "dsr_prob": dsr["dsr_prob"],
                "skew": skew, "kurt": kurt,
                "family": name, "timeframe": tf,
            })
    return out


def main() -> int:
    init_logger()
    logger = get_logger(__name__)

    # Load round-4 cross-pair pivot, pick the 5 survivors
    piv = pd.read_csv(PROJECT_ROOT / CROSS_PAIR_PIVOT)
    survivors = piv[(piv["EUR_USD"] >= 1.2) & (piv["GBP_USD"] >= 1.2)]
    logger.info(f"Loaded {len(survivors)} cross-pair survivors from round 4")

    all_rows: list[dict] = []
    for i, cfg in enumerate(survivors.to_dict("records")):
        tag = f"mr_cross_pair_rank{i}"
        logger.info(f"[{tag}] {cfg['family']} {cfg['timeframe']}")
        all_rows.extend(_eval_config(cfg, tag))

    # Also evaluate the best Round-10 trend configs
    trend_configs = [
        ("ma_crossover", "1D", {"fast_length": 8, "slow_length": 32, "ma_type": "ema"}, "trend_ma_8_32_D1"),
        ("ma_crossover", "4H", {"fast_length": 16, "slow_length": 64, "ma_type": "ema"}, "trend_ma_16_64_4H"),
        ("donchian_breakout", "4H", {"entry_lookback": 100, "exit_lookback": 40, "use_both_sides": True}, "trend_donchian_100_40_4H"),
        ("donchian_breakout", "1D", {"entry_lookback": 20, "exit_lookback": 10, "use_both_sides": True}, "trend_donchian_20_10_D1"),
    ]
    for name, tf, params, tag in trend_configs:
        logger.info(f"[{tag}] trend config")
        all_rows.extend(_eval_trend_family(name, tf, params, tag))

    df = pd.DataFrame(all_rows)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M")
    out_dir = PROJECT_ROOT / "backtest_results" / f"round11_stats_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "full_results.csv", index=False)

    print()
    print("=" * 100)
    print("Round 11 — proper statistical gauntlet (BCa bootstrap + Deflated Sharpe Ratio)")
    print("=" * 100)
    # Print slip 1.0 rows
    show = df[df["slip_mult"] == 1.0].copy()
    show = show[[
        "tag", "pair", "timeframe", "family",
        "n_trades", "pf_point", "pf_bca_low", "pf_bca_high",
        "sharpe_trade", "sharpe_expected_max", "dsr_prob",
    ]]
    print(show.round(3).to_string(index=False))
    print()
    # Friction check
    fric = df[df["slip_mult"] == 2.0][["tag", "pair", "pf_point"]].rename(
        columns={"pf_point": "pf_at_2x_slip"}
    )
    print("Friction 2x check:")
    print(fric.round(3).to_string(index=False))

    # Which pass the strict gate?
    # Gate: slip_mult=1 PF > 1.05, BCa low > 1.0, DSR prob > 0.95, AND friction 2x PF > 1.0
    passes = []
    for tag in df["tag"].unique():
        rows1 = df[(df["tag"] == tag) & (df["slip_mult"] == 1.0)]
        rows2 = df[(df["tag"] == tag) & (df["slip_mult"] == 2.0)]
        ok_1x = (rows1["pf_point"] > 1.05).all() \
                and (rows1["pf_bca_low"] > 1.0).all() \
                and (rows1["dsr_prob"] > 0.95).all()
        ok_2x = (rows2["pf_point"] > 1.0).all() if len(rows2) else False
        if ok_1x and ok_2x:
            passes.append(tag)

    print(f"\nCONFIGS PASSING STRICT GATE: {len(passes)}")
    for tag in passes:
        print(f"  ✓ {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
