"""Post-cost backtest metrics.

Sharpe, Sortino, Calmar, profit factor, win rate, expectancy, max drawdown,
avg drawdown duration. Designed to work with a vectorbt Pro ``Portfolio`` when
available, and to operate on raw returns/trades DataFrames otherwise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

MINUTES_PER_YEAR = 365 * 24 * 60  # 525,600 (FX runs ~24/5 but annualize over calendar)


def _infer_bars_per_year(returns: pd.Series) -> int | None:
    """Infer the annualization factor from the returns index spacing.

    Returns ``MINUTES_PER_YEAR / median_bar_minutes`` — i.e. the number of
    bars per year at the detected cadence. Falls back to None if the index
    isn't a DatetimeIndex or has <2 entries.
    """
    idx = returns.index
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 2:
        return None
    # Median delta in minutes — robust against gaps.
    deltas = idx.to_series().diff().dropna()
    if deltas.empty:
        return None
    median_minutes = deltas.median().total_seconds() / 60.0
    if median_minutes <= 0:
        return None
    return int(MINUTES_PER_YEAR / median_minutes)


@dataclass(frozen=True, slots=True)
class BacktestMetrics:
    """Post-cost metrics for a run.

    Attributes:
        total_trades: Number of round-trip trades.
        win_rate: Wins / total_trades.
        profit_factor: Sum(wins) / |Sum(losses)|. Inf if no losses.
        expectancy_usd: Average PnL per trade in account currency.
        sharpe: Annualized Sharpe of per-bar returns (frequency auto-detected).
        sortino: Annualized Sortino ratio (same caveat).
        calmar: CAGR / |max_drawdown|.
        max_drawdown_pct: Maximum drawdown as a fraction (e.g. 0.12 = 12%).
        avg_drawdown_duration_bars: Mean duration of drawdowns in bars.
        cagr: Compound annual growth rate.
        coverage_years: Wall-clock years covered by the input returns series.
            Downstream code uses this to compute true annualized profit
            instead of the "12/N months" shortcuts that produced inconsistent
            numbers in rounds 3 and 3.5. See ROUND_CHECKLIST.md.
    """

    total_trades: int
    win_rate: float
    profit_factor: float
    expectancy_usd: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown_pct: float
    avg_drawdown_duration_bars: float
    cagr: float
    coverage_years: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dict."""
        return asdict(self)

    def annualized_profit_usd(self) -> float:
        """True annualized profit computed from ``coverage_years``.

        ``expectancy_usd × total_trades`` is the total profit over the
        backtest window; dividing by the actual wall-clock coverage gives
        the correct annual figure regardless of timeframe or split count.
        Returns 0 if coverage wasn't recorded.
        """
        if self.coverage_years <= 0 or self.total_trades == 0:
            return 0.0
        total = self.expectancy_usd * self.total_trades
        return total / self.coverage_years


def compute_metrics(
    *,
    returns: pd.Series,
    trade_pnl_usd: pd.Series,
    initial_cash: float,
    minutes_per_year: int | None = None,
) -> BacktestMetrics:
    """Compute post-cost metrics from per-bar returns and trade PnLs.

    Args:
        returns: Per-bar returns (e.g. equity.pct_change()). The bar frequency
            is auto-detected from the index when possible so Sharpe/Sortino
            annualize correctly for non-minute timeframes.
        trade_pnl_usd: Per-trade realized PnL in account currency.
        initial_cash: Starting equity.
        minutes_per_year: Override the annualization factor. If None,
            inferred from the returns index spacing (or falls back to
            ``MINUTES_PER_YEAR`` if inference fails).

    Returns:
        :class:`BacktestMetrics`.
    """
    returns = returns.dropna()

    # Auto-detect annualization if not overridden. The factor equals
    # (bars per year) = (minutes_per_year / bar_minutes).
    if minutes_per_year is None:
        minutes_per_year = _infer_bars_per_year(returns) or MINUTES_PER_YEAR

    # Wall-clock coverage in years (for honest annualization downstream).
    coverage_years = 0.0
    if isinstance(returns.index, pd.DatetimeIndex) and len(returns) >= 2:
        delta = returns.index[-1] - returns.index[0]
        coverage_years = delta.total_seconds() / (365.0 * 24.0 * 3600.0)

    total_trades = int(trade_pnl_usd.shape[0])
    if total_trades == 0:
        return BacktestMetrics(
            total_trades=0,
            win_rate=0.0,
            profit_factor=float("inf"),
            expectancy_usd=0.0,
            sharpe=0.0,
            sortino=0.0,
            calmar=0.0,
            max_drawdown_pct=0.0,
            avg_drawdown_duration_bars=0.0,
            cagr=0.0,
            coverage_years=coverage_years,
        )

    wins = trade_pnl_usd[trade_pnl_usd > 0]
    losses = trade_pnl_usd[trade_pnl_usd < 0]
    win_rate = len(wins) / total_trades
    profit_factor = (
        float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")
    )
    expectancy_usd = float(trade_pnl_usd.mean())

    # Sharpe / Sortino on minute returns, annualized.
    mean_r = returns.mean()
    std_r = returns.std(ddof=0)
    sharpe = float(mean_r / std_r * np.sqrt(minutes_per_year)) if std_r > 0 else 0.0

    downside = returns[returns < 0]
    downside_std = downside.std(ddof=0)
    sortino = (
        float(mean_r / downside_std * np.sqrt(minutes_per_year))
        if downside_std > 0
        else 0.0
    )

    # Drawdown.
    equity = (1.0 + returns).cumprod() * initial_cash
    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = float(dd.min())  # negative

    # Drawdown durations: bars where dd < 0.
    in_dd = dd < 0
    durations: list[int] = []
    run = 0
    for flag in in_dd:
        if flag:
            run += 1
        else:
            if run > 0:
                durations.append(run)
            run = 0
    if run > 0:
        durations.append(run)
    avg_dd_duration = float(np.mean(durations)) if durations else 0.0

    # CAGR.
    total_minutes = len(returns)
    years = total_minutes / minutes_per_year if minutes_per_year > 0 else 0.0
    final_value = float(equity.iloc[-1]) if len(equity) else initial_cash
    cagr = (
        float((final_value / initial_cash) ** (1.0 / years) - 1.0)
        if years > 0 and final_value > 0
        else 0.0
    )

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    return BacktestMetrics(
        total_trades=total_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        expectancy_usd=expectancy_usd,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown_pct=abs(max_dd),
        avg_drawdown_duration_bars=avg_dd_duration,
        cagr=cagr,
        coverage_years=coverage_years,
    )
