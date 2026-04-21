"""Backtest metric tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.metrics import compute_metrics


class TestComputeMetrics:
    def test_empty_trades(self) -> None:
        ret = pd.Series(dtype=float)
        pnl = pd.Series(dtype=float)
        m = compute_metrics(returns=ret, trade_pnl_usd=pnl, initial_cash=500.0)
        assert m.total_trades == 0
        assert m.sharpe == 0.0
        assert m.max_drawdown_pct == 0.0

    def test_all_winners(self, minute_equity_returns: pd.Series) -> None:
        pnl = pd.Series([10.0, 5.0, 20.0])
        m = compute_metrics(
            returns=minute_equity_returns,
            trade_pnl_usd=pnl,
            initial_cash=500.0,
        )
        assert m.total_trades == 3
        assert m.win_rate == 1.0
        assert m.profit_factor == float("inf")
        assert m.expectancy_usd > 0

    def test_mixed_trades_profit_factor(
        self, minute_equity_returns: pd.Series
    ) -> None:
        pnl = pd.Series([20.0, -10.0, 15.0, -5.0])
        m = compute_metrics(
            returns=minute_equity_returns,
            trade_pnl_usd=pnl,
            initial_cash=500.0,
        )
        assert m.total_trades == 4
        assert m.win_rate == 0.5
        # PF = (20 + 15) / (10 + 5) = 35 / 15 = 2.333...
        assert abs(m.profit_factor - (35 / 15)) < 1e-6

    def test_drawdown_from_declining_returns(self) -> None:
        idx = pd.date_range("2024-01-01", periods=100, freq="1min", tz="UTC")
        # Monotonically declining returns → increasing drawdown.
        ret = pd.Series(np.full(100, -0.001), index=idx)
        pnl = pd.Series([-5.0])
        m = compute_metrics(returns=ret, trade_pnl_usd=pnl, initial_cash=500.0)
        # (1 - 0.001) ^ 100 ≈ 0.9048; DD ≈ 9.52%.
        assert m.max_drawdown_pct > 0.09
        assert m.max_drawdown_pct < 0.10
