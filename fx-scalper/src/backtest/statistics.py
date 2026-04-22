"""Rigorous statistical validation primitives (Round 11).

Three tools the Round 9 gauntlet was missing, per the final vbt.chat
review + the research-agent findings:

  1. ``purged_kfold_splits`` — Lopez de Prado's purged k-fold with
     embargo. Prevents train/test leakage from overlapping label windows
     and autocorrelated features (`Advances in Financial Machine
     Learning`, Ch. 7).

  2. ``bca_bootstrap_pf`` — bias-corrected & accelerated bootstrap
     for profit factor. Proper CI construction (Efron 1987) instead
     of naive percentile bootstrap. Handles the skew inherent in PF.

  3. ``deflated_sharpe_ratio`` — Bailey & Lopez de Prado (2014).
     Penalizes observed Sharpe for multiple-testing and non-normality
     (skew / excess kurtosis). Directly addresses the "30K configs
     tested, PF 2.24 is optimistic" problem from Round 9.

Together these replace the naive Round-9 percentile bootstrap and 3-split
WFA with something that clears a serious statistical bar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as _stats


# ---------------------------------------------------------------------------
# Purged k-fold CV (Lopez de Prado, AFML Ch. 7)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PurgedKFoldSplit:
    """One train/test split from a purged k-fold.

    Attributes:
        train_idx: Integer positions of train samples (post-purge).
        test_idx: Integer positions of test samples.
        fold: 0-indexed fold number (0..k-1).
    """

    train_idx: np.ndarray
    test_idx: np.ndarray
    fold: int


def purged_kfold_splits(
    n_samples: int,
    *,
    k: int = 5,
    embargo_frac: float = 0.01,
) -> list[PurgedKFoldSplit]:
    """Yield k-fold CV splits with purging + embargo.

    The standard k-fold leaks when labels are overlapping (e.g. a
    trade opened at bar t and closed at bar t+h contaminates any
    bar in [t, t+h] being in the test set). Purging removes train
    samples whose labels overlap the test set; embargo further
    removes a buffer BEYOND the test-set end to prevent look-ahead
    via autocorrelated features (returns, ATR, etc.).

    Args:
        n_samples: Total observations (bars or trades).
        k: Number of folds. 5 or 10 is standard; Lopez de Prado
            recommends 10 for most financial applications.
        embargo_frac: Size of embargo band after each test fold as
            a fraction of n_samples. 0.01 = 1% = ~13 bars per 1,300.
            Tune based on feature autocorrelation horizon.

    Returns:
        List of :class:`PurgedKFoldSplit` objects.

    Note:
        This implementation assumes no label overlap between samples
        (one trade = one sample). For overlapping labels, pass a
        ``pd.DataFrame`` of ``(t0, t1)`` trade windows to a more
        general purging implementation.
    """
    if k < 2:
        raise ValueError("k must be >= 2")
    indices = np.arange(n_samples)
    fold_bounds = np.linspace(0, n_samples, k + 1, dtype=int)
    embargo = int(n_samples * embargo_frac)

    splits: list[PurgedKFoldSplit] = []
    for fold_i in range(k):
        test_start = fold_bounds[fold_i]
        test_end = fold_bounds[fold_i + 1]
        test_idx = indices[test_start:test_end]

        # Train = everything NOT in test and NOT in embargo zone.
        # Embargo zone: [test_end, test_end + embargo).
        # Purge zone: bar-before-test is already handled by simple
        # contiguous fold structure here.
        embargo_end = min(test_end + embargo, n_samples)
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[test_start:embargo_end] = False
        # Also purge bars immediately before test (symmetric) if embargo > 0
        train_mask[max(0, test_start - embargo):test_start] = False
        train_idx = indices[train_mask]

        splits.append(PurgedKFoldSplit(
            train_idx=train_idx, test_idx=test_idx, fold=fold_i,
        ))
    return splits


# ---------------------------------------------------------------------------
# BCa bootstrap for profit factor
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BCaBootstrapResult:
    """Bias-corrected & accelerated bootstrap result for a statistic.

    Attributes:
        point_estimate: Original-sample statistic value.
        ci_lower: Lower bound of 1-alpha CI.
        ci_upper: Upper bound of 1-alpha CI.
        alpha: Significance level (e.g. 0.05 for 95% CI).
        n_resamples: Number of bootstrap resamples used.
    """

    point_estimate: float
    ci_lower: float
    ci_upper: float
    alpha: float
    n_resamples: int


def _profit_factor(pnls: np.ndarray) -> float:
    """Plain PF = sum(wins) / |sum(losses)|. Returns inf if no losses."""
    wins = pnls[pnls > 0].sum()
    losses = abs(pnls[pnls < 0].sum())
    if losses == 0:
        return float("inf")
    return float(wins / losses)


def bca_bootstrap_pf(
    trade_pnls: np.ndarray | pd.Series,
    *,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    random_state: int | None = 42,
) -> BCaBootstrapResult:
    """BCa (bias-corrected & accelerated) bootstrap CI on profit factor.

    Uses ``scipy.stats.bootstrap`` with ``method='BCa'`` — the
    gold-standard bootstrap CI that handles skewed distributions
    properly. PF is strongly skewed (heavy right tail when wins
    cluster) so BCa is strictly better than percentile bootstrap.

    Args:
        trade_pnls: 1-D array of per-trade PnLs.
        n_resamples: Number of bootstrap resamples. 10,000 is the
            standard for BCa; below 5,000 the tail estimates get noisy.
        alpha: 2-sided significance level. 0.05 → 95% CI.
        random_state: Seed for reproducibility.

    Returns:
        :class:`BCaBootstrapResult`.
    """
    pnls = np.asarray(trade_pnls, dtype=float)
    pnls = pnls[~np.isnan(pnls)]
    if len(pnls) < 10:
        return BCaBootstrapResult(
            point_estimate=float("nan"),
            ci_lower=float("nan"),
            ci_upper=float("nan"),
            alpha=alpha,
            n_resamples=n_resamples,
        )

    point = _profit_factor(pnls)

    # Need a 1-D sample sequence — scipy expects (data,) tuple.
    res = _stats.bootstrap(
        (pnls,),
        statistic=_profit_factor,
        n_resamples=n_resamples,
        confidence_level=1 - alpha,
        method="BCa",
        random_state=random_state,
        vectorized=False,
    )
    return BCaBootstrapResult(
        point_estimate=point,
        ci_lower=float(res.confidence_interval.low),
        ci_upper=float(res.confidence_interval.high),
        alpha=alpha,
        n_resamples=n_resamples,
    )


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
# ---------------------------------------------------------------------------

def deflated_sharpe_ratio(
    observed_sharpe: float,
    *,
    n_trials: int,
    n_observations: int,
    skew: float = 0.0,
    kurt: float = 3.0,
    threshold_sharpe: float = 0.0,
) -> dict[str, float]:
    """Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio.

    Corrects the observed Sharpe ratio for:
     - Multiple-testing bias (``n_trials`` strategies evaluated — only
       the best is kept, inflating apparent Sharpe).
     - Non-normality of returns (``skew``, ``kurt`` — fat-tailed
       distributions have lower effective Sharpe than the naive number
       suggests).

    The null hypothesis is that the TRUE Sharpe is ``threshold_sharpe``.
    DSR answers: "given I tested ``n_trials`` strategies on
    ``n_observations`` returns, what's the probability the observed
    Sharpe is real (not random noise)?"

    Args:
        observed_sharpe: Annualized Sharpe of the best strategy.
        n_trials: Number of INDEPENDENT strategies tested. For our
            Round 5 sweep that's ~30,000; for Round 9's 5 survivors
            it's 5. Pass the effective-independent count if you can
            estimate it (e.g. after accounting for correlated param
            neighborhoods).
        n_observations: Number of return observations (bars or trades)
            used to compute ``observed_sharpe``. Typically trades for
            per-trade Sharpe, bars for return-based Sharpe.
        skew: Skew of the return distribution. 0.0 = normal assumption.
            Negative skew makes DSR more punitive.
        kurt: Kurtosis of the return distribution. 3.0 = normal; FX
            returns typically 5-15. Higher kurt = more punitive.
        threshold_sharpe: Null-hypothesis Sharpe (usually 0.0).

    Returns:
        Dict with:
         - ``dsr_z``: z-score of observed vs deflated-expected null
         - ``dsr_prob``: P(true Sharpe > threshold | observed)
         - ``expected_max_sharpe``: Expected Sharpe of best-of-n_trials
           random strategy (what you'd get by chance).

    Reference:
        Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio:
        Correcting for Selection Bias, Backtest Overfitting, and
        Non-Normality", Journal of Portfolio Management.
    """
    # Expected maximum Sharpe from n_trials random strategies under the
    # null. Approximation from Bailey & Lopez de Prado:
    #   E[max SR] ≈ (sqrt(2 ln n_trials)) * (1 - gamma_euler/sqrt(2 ln n_trials))
    if n_trials <= 1:
        expected_max = threshold_sharpe
    else:
        gamma_euler = 0.5772156649
        lnn = np.log(max(n_trials, 2))
        expected_max = float(
            threshold_sharpe
            + np.sqrt(2 * lnn)
            * (1 - gamma_euler / np.sqrt(2 * lnn))
            - _stats.norm.ppf(1 - 1 / (n_trials * np.e))
            / np.sqrt(2 * lnn)
        )
        # The closed form in the paper is:
        # E[max SR] = sqrt((1-gamma)*Z(1-1/N) + gamma*Z(1-1/(N*e)))
        # where Z is the inverse standard normal CDF. We use a simpler
        # asymptotic here which is accurate for N >= 10.

    # Deflation factor — accounts for skew and kurt.
    # Variance of observed SR under null (Mertens 2002):
    #   var(SR) ≈ (1 + 0.5*SR^2 - skew*SR + (kurt-3)/4 * SR^2) / n_obs
    sharpe_var = (
        1
        + 0.5 * observed_sharpe**2
        - skew * observed_sharpe
        + (kurt - 3) / 4.0 * observed_sharpe**2
    ) / max(n_observations - 1, 1)
    sharpe_std = float(np.sqrt(max(sharpe_var, 1e-12)))

    dsr_z = (observed_sharpe - expected_max) / sharpe_std
    dsr_prob = float(_stats.norm.cdf(dsr_z))

    return {
        "dsr_z": float(dsr_z),
        "dsr_prob": dsr_prob,
        "expected_max_sharpe": expected_max,
        "sharpe_std": sharpe_std,
    }


# ---------------------------------------------------------------------------
# Convenience: full statistical gauntlet
# ---------------------------------------------------------------------------

def full_stat_gauntlet(
    trade_pnls: np.ndarray | pd.Series,
    *,
    annualization_factor: float = 252.0,
    n_trials_tested: int = 1,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Run all three tests on a single strategy's trade-PnL series.

    Returns PF point estimate, BCa 95% CI, annualized Sharpe, and
    DSR. This is the new canonical "is this real?" check that
    replaces the Round-9 naive gauntlet.

    Args:
        trade_pnls: 1-D per-trade PnL.
        annualization_factor: For Sharpe annualization. 252 = trading
            days/year; use 252 * trades_per_day for per-trade Sharpe.
            Default 252 assumes the input is roughly daily aggregated.
        n_trials_tested: For DSR deflation. 30000 if you ran a full
            sweep; 1 if this is a truly a priori strategy.
        alpha: Significance level.

    Returns:
        Dict with ``pf_point``, ``pf_ci_low``, ``pf_ci_high``,
        ``sharpe_raw``, ``dsr_prob``, ``expected_max_sharpe``, and
        ``pass_all`` (all checks > 0.5 / > 1.0 / > threshold).
    """
    pnls = np.asarray(trade_pnls, dtype=float)
    pnls = pnls[~np.isnan(pnls)]
    if len(pnls) < 10:
        return {"pf_point": float("nan"), "pass_all": False}

    bca = bca_bootstrap_pf(pnls, alpha=alpha)

    # Raw Sharpe of trade PnLs (treated as i.i.d. observations).
    mu, sigma = float(np.mean(pnls)), float(np.std(pnls, ddof=1))
    raw_sharpe = (mu / sigma * np.sqrt(annualization_factor)) if sigma > 0 else 0.0
    skew = float(_stats.skew(pnls)) if len(pnls) >= 3 else 0.0
    kurt = float(_stats.kurtosis(pnls, fisher=False)) if len(pnls) >= 4 else 3.0

    dsr = deflated_sharpe_ratio(
        raw_sharpe,
        n_trials=n_trials_tested,
        n_observations=len(pnls),
        skew=skew,
        kurt=kurt,
    )

    pass_all = (
        bca.ci_lower > 1.0
        and dsr["dsr_prob"] > (1 - alpha)
        and raw_sharpe > dsr["expected_max_sharpe"]
    )
    return {
        "pf_point": bca.point_estimate,
        "pf_ci_low": bca.ci_lower,
        "pf_ci_high": bca.ci_upper,
        "sharpe_raw": raw_sharpe,
        "dsr_prob": dsr["dsr_prob"],
        "dsr_z": dsr["dsr_z"],
        "expected_max_sharpe": dsr["expected_max_sharpe"],
        "n_trades": len(pnls),
        "skew": skew,
        "kurt": kurt,
        "pass_all": bool(pass_all),
    }
