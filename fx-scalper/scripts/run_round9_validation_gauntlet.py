"""Round 9 — the validation gauntlet.

Per FINAL_ROLLUP.md and the final vbt.chat review, the top candidate
configs haven't been stressed enough to clear for paper trading. This
round runs every survivor from round-4-rerun (PF > 1.2 on both EUR and
GBP with corrected sizing) through three concurrent gauntlets:

  1. FULL-SAMPLE TRUTH — simple single run on 3.3-year EUR/USD and
     1-year GBP/USD. If PF < 1.0 full-sample, KILL regardless of
     walk-forward.

  2. BOOTSTRAP PF CONFIDENCE INTERVAL — 1,000 resamples of the trade
     PnL series per config/pair. 95% CI. If CI straddles 1.0, KILL.

  3. FRICTION STRESS LADDER — re-run with slippage multipliers
     {1×, 2×, 5×, 10×} of current (half-spread) estimate. Report PF
     at each level. If PF < 1.0 at 2×, KILL — live execution easily
     hits that.

Outputs:
  backtest_results/round9_gauntlet_YYYYMMDDTHHMM/
    - full_sample.csv   (config × pair → PF/WR/exp/DD)
    - bootstrap_ci.csv  (config × pair → mean/lo95/hi95 PF)
    - friction_stress.csv  (config × pair × slippage_mult → PF)
    - survivors.csv  (configs that passed ALL THREE gauntlets)
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


CROSS_PAIR_PIVOT = "backtest_results/cross_pair_20260422T2150/cross_pair_pivot.csv"
PAIRS = ["EUR_USD", "GBP_USD"]
FRICTION_MULTS = [1.0, 2.0, 5.0, 10.0]
N_BOOTSTRAP = 1000
BOOTSTRAP_CI_ALPHA = 0.05


def _load_survivors(pivot_path: Path, min_pf: float = 1.2) -> list[dict]:
    """Read the round-4-rerun pivot and return configs that hit PF >= min_pf
    on both EUR_USD and GBP_USD (the cross-pair-survivor set)."""
    df = pd.read_csv(pivot_path)
    df = df[(df["EUR_USD"] >= min_pf) & (df["GBP_USD"] >= min_pf)]
    return df.to_dict("records")


def _build_pf(
    config: dict, pair: str, slippage_mult: float = 1.0
) -> vbt.Portfolio | None:
    """Instantiate the strategy and run a full-sample portfolio on `pair`."""
    family_cls = get_family_by_name(config["family"])
    if family_cls is None:
        return None
    family_params = json.loads(config["family_params"])
    exit_dict = json.loads(config["exit_config"])

    m1 = load_symbol_bars(pair, start="2023-01-01", end="2026-04-20")
    if m1.empty:
        return None
    bars = resample_bars(m1, config["timeframe"])
    close_col = "mid_close" if "mid_close" in bars.columns else "bid_close"
    close = bars[close_col]
    atr_len = int(exit_dict.get("atr_length", 14))
    atr = add_atr(bars, length=atr_len)[f"atr_{atr_len}"]

    family = family_cls(family_cls.params_cls(**family_params))
    sigs = family.generate(bars)

    exit_cfg = ExitConfig(
        sl_atr_mult=exit_dict["sl_atr_mult"],
        atr_length=atr_len,
        tp_r_mult=exit_dict.get("tp_r_mult"),
        trail_kind=exit_dict["trail_kind"],
        trail_atr_mult=exit_dict.get("trail_atr_mult"),
    )
    vp = config_to_vbt_params(sigs.entries_long, sigs.entries_short, close, atr, exit_cfg)

    if "bid_close" in bars.columns and "ask_close" in bars.columns:
        base_slip = (bars["ask_close"] - bars["bid_close"]) / 2.0
    else:
        base_slip = pd.Series(0.0, index=bars.index)
    slip = (base_slip * slippage_mult).to_numpy()

    kwargs = dict(
        close=close,
        entries=sigs.entries_long,
        short_entries=sigs.entries_short,
        tp_stop=vp.tp_stop.fillna(0.0).to_numpy(),
        size=5000.0,
        size_type="value",
        leverage=50,
        init_cash=500.0,
        freq=bars.index.freq or "15min",
        fees=0.0,
        slippage=slip,
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


def _pf_stats(pf: vbt.Portfolio) -> dict:
    trades = pf.trades
    wins = trades.winning.pnl.sum()
    losses = abs(trades.losing.pnl.sum())
    pf_ratio = float(wins / losses) if float(losses) > 0 else float("inf")
    return {
        "n_trades": int(trades.count()),
        "win_rate": float(trades.win_rate),
        "profit_factor": pf_ratio,
        "expectancy_usd": float(trades.pnl.mean()),
        "total_return": float(pf.total_return),
        "max_dd": float(abs(pf.max_drawdown)),
        "final_equity": float(pf.final_value),
    }


def _bootstrap_pf_ci(
    trade_pnls: np.ndarray, n: int = N_BOOTSTRAP, alpha: float = BOOTSTRAP_CI_ALPHA
) -> dict:
    """Bootstrap the PF distribution from a fixed set of trade PnLs.

    Resample with replacement ``n`` times, compute PF each time, return
    mean + 2-sided CI at confidence 1-alpha.
    """
    if len(trade_pnls) < 5:
        return {"bootstrap_mean_pf": float("nan"),
                "bootstrap_lo95_pf": float("nan"),
                "bootstrap_hi95_pf": float("nan"),
                "bootstrap_frac_above_1": float("nan")}
    rng = np.random.default_rng(42)
    size = len(trade_pnls)
    samples = []
    for _ in range(n):
        resampled = rng.choice(trade_pnls, size=size, replace=True)
        wins = resampled[resampled > 0].sum()
        losses = abs(resampled[resampled < 0].sum())
        if losses == 0:
            samples.append(float("inf"))
        else:
            samples.append(wins / losses)
    arr = np.array([s for s in samples if np.isfinite(s)])
    lo = float(np.quantile(arr, alpha / 2))
    hi = float(np.quantile(arr, 1 - alpha / 2))
    return {
        "bootstrap_mean_pf": float(np.mean(arr)),
        "bootstrap_lo95_pf": lo,
        "bootstrap_hi95_pf": hi,
        "bootstrap_frac_above_1": float((arr > 1.0).mean()),
    }


def main() -> int:
    init_logger()
    logger = get_logger(__name__)

    survivors = _load_survivors(PROJECT_ROOT / CROSS_PAIR_PIVOT, min_pf=1.2)
    logger.info(f"Loaded {len(survivors)} cross-pair survivor configs")
    if not survivors:
        logger.error("no cross-pair survivors; nothing to validate")
        return 1

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M")
    out_dir = PROJECT_ROOT / "backtest_results" / f"round9_gauntlet_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # === Gauntlet 1: full-sample truth ===
    logger.info("=== Gauntlet 1: full-sample truth ===")
    fs_rows = []
    bootstrap_cache: dict[tuple[int, str], dict] = {}
    for i, cfg in enumerate(survivors):
        for pair in PAIRS:
            pf = _build_pf(cfg, pair, slippage_mult=1.0)
            if pf is None:
                continue
            stats = _pf_stats(pf)
            stats.update({"config_rank": i, "pair": pair,
                          "timeframe": cfg["timeframe"], "family": cfg["family"]})
            fs_rows.append(stats)

            # Cache the trade PnL array for bootstrap. vbt MappedArray
            # exposes .values; not .to_numpy. Use the readable records.
            pnls = pf.trades.records_readable["PnL"].to_numpy()
            bootstrap_cache[(i, pair)] = {
                "trade_pnls": pnls,
                "full_sample_pf": stats["profit_factor"],
            }
    fs_df = pd.DataFrame(fs_rows)
    fs_df.to_csv(out_dir / "full_sample.csv", index=False)
    logger.info(f"wrote {len(fs_df)} full-sample rows")

    # === Gauntlet 2: bootstrap CI ===
    logger.info(f"=== Gauntlet 2: bootstrap PF CI (n={N_BOOTSTRAP}) ===")
    bs_rows = []
    for (i, pair), d in bootstrap_cache.items():
        ci = _bootstrap_pf_ci(d["trade_pnls"])
        bs_rows.append({
            "config_rank": i,
            "pair": pair,
            "n_trades": len(d["trade_pnls"]),
            "full_sample_pf": d["full_sample_pf"],
            **ci,
        })
    bs_df = pd.DataFrame(bs_rows)
    bs_df.to_csv(out_dir / "bootstrap_ci.csv", index=False)

    # === Gauntlet 3: friction stress ladder ===
    logger.info("=== Gauntlet 3: friction stress ladder ===")
    fr_rows = []
    for i, cfg in enumerate(survivors):
        for pair in PAIRS:
            for mult in FRICTION_MULTS:
                pf = _build_pf(cfg, pair, slippage_mult=mult)
                if pf is None:
                    continue
                stats = _pf_stats(pf)
                stats.update({"config_rank": i, "pair": pair,
                              "slippage_mult": mult,
                              "timeframe": cfg["timeframe"], "family": cfg["family"]})
                fr_rows.append(stats)
    fr_df = pd.DataFrame(fr_rows)
    fr_df.to_csv(out_dir / "friction_stress.csv", index=False)

    # === Survivor gate ===
    # Full-sample PF > 1.0 on both pairs
    # Bootstrap 95% CI lower-bound > 1.0 on both pairs
    # PF at 2x slippage > 1.0 on both pairs
    logger.info("=== Applying survivor gate ===")
    gate_rows = []
    for i in range(len(survivors)):
        eur_fs = fs_df[(fs_df.config_rank == i) & (fs_df.pair == "EUR_USD")]
        gbp_fs = fs_df[(fs_df.config_rank == i) & (fs_df.pair == "GBP_USD")]
        eur_bs = bs_df[(bs_df.config_rank == i) & (bs_df.pair == "EUR_USD")]
        gbp_bs = bs_df[(bs_df.config_rank == i) & (bs_df.pair == "GBP_USD")]
        eur_fr2 = fr_df[(fr_df.config_rank == i) & (fr_df.pair == "EUR_USD") & (fr_df.slippage_mult == 2.0)]
        gbp_fr2 = fr_df[(fr_df.config_rank == i) & (fr_df.pair == "GBP_USD") & (fr_df.slippage_mult == 2.0)]

        def _first_pf(df: pd.DataFrame) -> float:
            return float(df.profit_factor.iloc[0]) if len(df) else float("nan")

        def _first_lo(df: pd.DataFrame) -> float:
            return float(df.bootstrap_lo95_pf.iloc[0]) if len(df) else float("nan")

        pass_fs = _first_pf(eur_fs) > 1.0 and _first_pf(gbp_fs) > 1.0
        pass_bs = _first_lo(eur_bs) > 1.0 and _first_lo(gbp_bs) > 1.0
        pass_fr = _first_pf(eur_fr2) > 1.0 and _first_pf(gbp_fr2) > 1.0

        gate_rows.append({
            "config_rank": i,
            "family": survivors[i]["family"],
            "timeframe": survivors[i]["timeframe"],
            "eur_fs_pf": _first_pf(eur_fs),
            "gbp_fs_pf": _first_pf(gbp_fs),
            "eur_bs_lo95": _first_lo(eur_bs),
            "gbp_bs_lo95": _first_lo(gbp_bs),
            "eur_fr2_pf": _first_pf(eur_fr2),
            "gbp_fr2_pf": _first_pf(gbp_fr2),
            "pass_full_sample": pass_fs,
            "pass_bootstrap": pass_bs,
            "pass_friction_2x": pass_fr,
            "pass_all": pass_fs and pass_bs and pass_fr,
        })
    gate_df = pd.DataFrame(gate_rows)
    gate_df.to_csv(out_dir / "gate_summary.csv", index=False)
    survivors_df = gate_df[gate_df["pass_all"]]
    survivors_df.to_csv(out_dir / "survivors.csv", index=False)

    print()
    print("=" * 76)
    print(f"Round 9 — validation gauntlet on {len(survivors)} cross-pair survivors")
    print("=" * 76)
    print(gate_df.round(3).to_string(index=False))
    print()
    print(f"CONFIGS PASSING ALL 3 GAUNTLETS: {len(survivors_df)}")
    if len(survivors_df):
        print()
        print(survivors_df.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
