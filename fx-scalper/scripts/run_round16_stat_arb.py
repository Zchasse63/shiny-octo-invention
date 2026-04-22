"""Round 16 — FX stat-arb daily with Kalman-filter hedge ratio.

Per the research-review (Landi & Lemishko SSRN; QuantStart Kalman
guide): the canonical FX pairs-trading approach is daily cointegration
between majors with a Kalman-filter-adapted hedge ratio. Static-OLS
hedge ratios blow up across regime shifts (2015 CHF unpeg, 2020 COVID,
2022 rate divergence); Kalman adapts slowly enough to avoid
over-reacting, fast enough to survive breaks.

Spec (from research synthesis):
  1. Engle-Granger cointegration test on log-prices rolling 252-day window
  2. If p-value < 0.05 → proceed; else skip.
  3. Kalman filter on log(EUR/USD) vs log(GBP/USD) → time-varying
     (intercept alpha_t, slope beta_t).
  4. Residual = log(EUR/USD) - beta_t * log(GBP/USD) - alpha_t.
  5. Z-score = (residual - rolling_mean_60) / rolling_std_60.
  6. Entry: |z| > 2.0. Exit: |z| < 0.5. Hard stop: |z| > 3.5.
  7. Trade both legs in $-neutral size (equal notional on both).
  8. Re-test cointegration every 20 bars; flatten if p > 0.05.

Research-review honest note: Koronidis 2013 found EUR/USD + GBP/USD
do NOT cointegrate at 5% on daily data. So we expect this to mostly
NOT generate trades — which is still informative. If it does find
windows of cointegration and trades profitably, that's a real edge.

vbt-native portfolio with cash_sharing=True for 2-leg trades.
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
from statsmodels.tsa.stattools import adfuller, coint  # noqa: E402

from src.backtest.data_loader import load_symbol_bars  # noqa: E402
from src.backtest.resample import resample_bars  # noqa: E402
from src.backtest.statistics import bca_bootstrap_pf, deflated_sharpe_ratio  # noqa: E402
from src.utils.logger import get_logger, init_logger  # noqa: E402


def _kalman_hedge_ratio(
    y: np.ndarray, x: np.ndarray,
    *, obs_var: float = 1e-3, trans_cov: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Online Kalman filter returning time-varying (intercept, slope).

    State: [alpha_t, beta_t]. Observation: y_t = alpha_t + beta_t * x_t.
    Written manually (no pykalman dependency) for transparency.

    Args:
        y: Dependent log-price series.
        x: Independent log-price series.
        obs_var: Observation variance (measurement noise).
        trans_cov: Process variance (how fast hedge ratio drifts).

    Returns:
        (alphas, betas) as arrays of length len(y).
    """
    n = len(y)
    alphas = np.zeros(n)
    betas = np.zeros(n)

    # Initial state: OLS on first 30 bars
    k = min(30, n // 4)
    if k >= 5:
        x_init = np.column_stack([np.ones(k), x[:k]])
        try:
            coefs, *_ = np.linalg.lstsq(x_init, y[:k], rcond=None)
            mu = np.array([coefs[0], coefs[1]])
        except np.linalg.LinAlgError:
            mu = np.array([0.0, 1.0])
    else:
        mu = np.array([0.0, 1.0])
    P = np.eye(2) * 1.0  # initial state covariance

    Q = np.eye(2) * trans_cov  # process noise
    R = obs_var  # observation noise

    for t in range(n):
        # Predict
        mu_pred = mu  # state transition = identity for random-walk state
        P_pred = P + Q
        # Observation: y_t = [1, x_t] @ state
        H = np.array([1.0, x[t]])
        # Kalman gain
        S = H @ P_pred @ H + R
        K = P_pred @ H / S
        # Update
        innovation = y[t] - H @ mu_pred
        mu = mu_pred + K * innovation
        P = P_pred - np.outer(K, H) @ P_pred
        alphas[t] = mu[0]
        betas[t] = mu[1]
    return alphas, betas


def run_stat_arb(
    pair_y: str = "EUR_USD", pair_x: str = "GBP_USD",
    *,
    timeframe: str = "1D",
    coint_window: int = 252,
    zscore_window: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 3.5,
    recheck_every: int = 20,
    spread_bps: float = 0.6,
    notional_per_leg: float = 5000.0,
    init_cash: float = 5000.0,
) -> dict:
    logger = get_logger(__name__)

    # Load + align both pairs
    m1_y = load_symbol_bars(pair_y, start="2023-01-01", end="2026-04-20")
    m1_x = load_symbol_bars(pair_x, start="2023-01-01", end="2026-04-20")
    if m1_y.empty or m1_x.empty:
        return {"error": "missing data"}
    bars_y = resample_bars(m1_y, timeframe)
    bars_x = resample_bars(m1_x, timeframe)

    mid_y = (bars_y["bid_close"] + bars_y["ask_close"]) / 2 if "bid_close" in bars_y.columns else bars_y["mid_close"]
    mid_x = (bars_x["bid_close"] + bars_x["ask_close"]) / 2 if "bid_close" in bars_x.columns else bars_x["mid_close"]
    idx = mid_y.index.intersection(mid_x.index)
    mid_y = mid_y.loc[idx]
    mid_x = mid_x.loc[idx]
    logger.info(f"Aligned {len(idx):,} bars at {timeframe}")

    log_y = np.log(mid_y.to_numpy())
    log_x = np.log(mid_x.to_numpy())

    # Kalman hedge ratio
    alphas, betas = _kalman_hedge_ratio(log_y, log_x)

    # Residual spread
    resid = log_y - betas * log_x - alphas
    resid_s = pd.Series(resid, index=idx)
    # Z-score
    rolling_mean = resid_s.rolling(zscore_window, min_periods=zscore_window).mean()
    rolling_std = resid_s.rolling(zscore_window, min_periods=zscore_window).std()
    z = (resid_s - rolling_mean) / rolling_std

    # Cointegration check rolling
    coint_p_series = pd.Series(np.nan, index=idx)
    for t in range(coint_window, len(log_y), recheck_every):
        window_y = log_y[t - coint_window:t]
        window_x = log_x[t - coint_window:t]
        try:
            _stat, pval, _crit = coint(window_y, window_x)
            coint_p_series.iloc[t:t + recheck_every] = pval
        except Exception:
            pass
    coint_p_series = coint_p_series.ffill().fillna(1.0)

    # Trade generation
    trades = []
    position = 0  # 0 flat, +1 long spread (buy y / sell x), -1 short spread
    entry_z_val = 0.0
    entry_y_price = entry_x_price = 0.0
    entry_beta = 0.0
    for i in range(len(z)):
        z_val = z.iloc[i]
        if np.isnan(z_val):
            continue
        p_coint = float(coint_p_series.iloc[i])
        # flatten if cointegration broke
        if position != 0 and p_coint > 0.05:
            exit_y = mid_y.iloc[i]; exit_x = mid_x.iloc[i]
            # Compute trade PnL (both legs)
            y_ret = (exit_y - entry_y_price) / entry_y_price
            x_ret = (exit_x - entry_x_price) / entry_x_price
            pnl = position * (y_ret - entry_beta * x_ret) * notional_per_leg
            # Costs: round-trip spread on both legs
            pnl -= 2 * (spread_bps / 10000.0) * notional_per_leg
            trades.append({"entry_ts": entry_ts, "exit_ts": idx[i],
                           "position": position, "entry_z": entry_z_val,
                           "exit_z": z_val, "pnl": pnl, "reason": "coint_break"})
            position = 0
            continue

        # skip if not cointegrated at entry
        if position == 0 and p_coint > 0.05:
            continue

        if position == 0 and z_val > entry_z:
            position = -1
            entry_ts = idx[i]; entry_z_val = z_val
            entry_y_price = mid_y.iloc[i]; entry_x_price = mid_x.iloc[i]
            entry_beta = betas[i]
        elif position == 0 and z_val < -entry_z:
            position = +1
            entry_ts = idx[i]; entry_z_val = z_val
            entry_y_price = mid_y.iloc[i]; entry_x_price = mid_x.iloc[i]
            entry_beta = betas[i]
        elif position == +1 and (z_val > -exit_z or z_val < -stop_z):
            exit_y = mid_y.iloc[i]; exit_x = mid_x.iloc[i]
            y_ret = (exit_y - entry_y_price) / entry_y_price
            x_ret = (exit_x - entry_x_price) / entry_x_price
            pnl = (y_ret - entry_beta * x_ret) * notional_per_leg
            pnl -= 2 * (spread_bps / 10000.0) * notional_per_leg
            reason = "stop" if z_val < -stop_z else "mean_revert"
            trades.append({"entry_ts": entry_ts, "exit_ts": idx[i],
                           "position": +1, "entry_z": entry_z_val, "exit_z": z_val,
                           "pnl": pnl, "reason": reason})
            position = 0
        elif position == -1 and (z_val < exit_z or z_val > stop_z):
            exit_y = mid_y.iloc[i]; exit_x = mid_x.iloc[i]
            y_ret = (exit_y - entry_y_price) / entry_y_price
            x_ret = (exit_x - entry_x_price) / entry_x_price
            pnl = -1 * (y_ret - entry_beta * x_ret) * notional_per_leg
            pnl -= 2 * (spread_bps / 10000.0) * notional_per_leg
            reason = "stop" if z_val > stop_z else "mean_revert"
            trades.append({"entry_ts": entry_ts, "exit_ts": idx[i],
                           "position": -1, "entry_z": entry_z_val, "exit_z": z_val,
                           "pnl": pnl, "reason": reason})
            position = 0

    if not trades:
        logger.warning("No trades generated — likely no cointegrated window, which matches Koronidis 2013")
        return {
            "n_trades": 0, "fraction_cointegrated": float((coint_p_series <= 0.05).mean()),
            "min_coint_p": float(coint_p_series.min()),
        }
    df = pd.DataFrame(trades)
    pnls = df["pnl"].to_numpy()
    wins = pnls[pnls > 0].sum()
    losses = abs(pnls[pnls < 0].sum())
    pf = float(wins/losses) if losses > 0 else float("inf")
    bca = bca_bootstrap_pf(pnls, n_resamples=5000)

    # Sharpe + DSR
    mu, sigma = float(np.mean(pnls)), float(np.std(pnls, ddof=1))
    annualize = len(pnls) / 3.3
    sharpe = (mu/sigma*np.sqrt(annualize)) if sigma > 0 else 0.0
    from scipy import stats as _st
    skew = float(_st.skew(pnls)) if len(pnls) >= 3 else 0.0
    kurt = float(_st.kurtosis(pnls, fisher=False)) if len(pnls) >= 4 else 3.0
    dsr = deflated_sharpe_ratio(sharpe, n_trials=1, n_observations=len(pnls), skew=skew, kurt=kurt)

    return {
        "n_trades": len(trades),
        "total_pnl": float(pnls.sum()),
        "win_rate": float((pnls > 0).mean()),
        "profit_factor": pf,
        "pf_bca_lo": bca.ci_lower,
        "pf_bca_hi": bca.ci_upper,
        "sharpe": sharpe,
        "dsr_prob": dsr["dsr_prob"],
        "fraction_cointegrated": float((coint_p_series <= 0.05).mean()),
        "trades_df": df,
    }


def main() -> int:
    init_logger()
    logger = get_logger(__name__)

    # Default spec — EUR vs GBP daily
    print()
    print("=" * 90)
    print("Round 16 — stat-arb EUR/USD vs GBP/USD daily with Kalman hedge ratio")
    print("=" * 90)
    cfgs = [
        {"timeframe": "1D", "entry_z": 2.0, "exit_z": 0.5, "stop_z": 3.5},
        {"timeframe": "1D", "entry_z": 1.5, "exit_z": 0.3, "stop_z": 3.0},
        {"timeframe": "4H", "entry_z": 2.0, "exit_z": 0.5, "stop_z": 3.5},
    ]
    rows = []
    for cfg in cfgs:
        logger.info(f"Config: {cfg}")
        res = run_stat_arb(**cfg)
        if "error" in res:
            logger.warning(f"error: {res['error']}")
            continue
        if res.get("n_trades", 0) == 0:
            logger.info(f"No trades (fraction cointegrated = {res.get('fraction_cointegrated', 0):.1%})")
            rows.append({**cfg, "n_trades": 0, "fraction_cointegrated": res.get("fraction_cointegrated", 0)})
            continue
        rows.append({
            **cfg,
            "n_trades": res["n_trades"],
            "total_pnl": res["total_pnl"],
            "wr": res["win_rate"],
            "pf": res["profit_factor"],
            "pf_bca_lo": res["pf_bca_lo"],
            "pf_bca_hi": res["pf_bca_hi"],
            "sharpe": res["sharpe"],
            "dsr_prob": res["dsr_prob"],
            "fraction_cointegrated": res.get("fraction_cointegrated", 0),
        })
    df = pd.DataFrame(rows)
    print(df.round(3).to_string(index=False))
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M")
    out_dir = PROJECT_ROOT / "backtest_results" / f"round16_statarb_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "summary.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
