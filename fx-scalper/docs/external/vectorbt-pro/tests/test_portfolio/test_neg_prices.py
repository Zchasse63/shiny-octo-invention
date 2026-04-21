import os

import numpy as np
import pandas as pd
import pytest

import vectorbtpro as vbt
from vectorbtpro.portfolio import nb
from vectorbtpro.portfolio.enums import *


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True
    vbt.settings.portfolio["attach_call_seq"] = True


def teardown_module():
    vbt.settings.reset()


neg_price = pd.Series(
    [-10.0, -12.0, -8.0, -15.0, -11.0],
    index=pd.date_range("2020", periods=5),
)


class TestNegativePriceFromOrders:
    def test_long_buy_negative_price(self):
        pf = vbt.Portfolio.from_orders(
            neg_price,
            size=[1.0, 0, 0, 0, -1.0],
            init_cash=100.0,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        orders = pf.order_records
        assert len(orders) == 2
        assert orders["price"][0] == pytest.approx(-10.0)
        assert orders["size"][0] == pytest.approx(1.0)
        assert orders["side"][0] == OrderSide.Buy
        assert orders["price"][1] == pytest.approx(-11.0)
        assert orders["size"][1] == pytest.approx(1.0)
        assert orders["side"][1] == OrderSide.Sell
        np.testing.assert_allclose(pf.cash.values[0], 110.0)
        np.testing.assert_allclose(pf.cash.values[4], 99.0)

    def test_long_roundtrip_pnl(self):
        pf = vbt.Portfolio.from_orders(
            neg_price,
            size=[1.0, 0, 0, 0, -1.0],
            init_cash=1000.0,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        trades = pf.trades.records
        np.testing.assert_allclose(trades["pnl"].values[0], -1.0)

    def test_short_roundtrip_pnl(self):
        pf = vbt.Portfolio.from_orders(
            neg_price,
            size=[1.0, 0, 0, 0, -1.0],
            direction="shortonly",
            init_cash=1000.0,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        trades = pf.trades.records
        np.testing.assert_allclose(trades["pnl"].values[0], 1.0)

    def test_trade_return_signs_match_pnl(self):
        long_pf = vbt.Portfolio.from_orders(
            pd.Series([-10.0, -11.0], index=pd.date_range("2020", periods=2)),
            size=[1.0, -1.0],
            init_cash=1000.0,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        long_trades = long_pf.trades.records
        np.testing.assert_allclose(long_trades["pnl"].values[0], -1.0)
        np.testing.assert_allclose(long_trades["return"].values[0], -0.1)

        short_pf = vbt.Portfolio.from_orders(
            pd.Series([-10.0, -11.0], index=pd.date_range("2020", periods=2)),
            size=[1.0, -1.0],
            direction="shortonly",
            init_cash=1000.0,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        short_trades = short_pf.trades.records
        np.testing.assert_allclose(short_trades["pnl"].values[0], 1.0)
        np.testing.assert_allclose(short_trades["return"].values[0], 0.1)

    def test_asset_value_with_negative_price(self):
        pf = vbt.Portfolio.from_orders(
            neg_price,
            size=[1.0, 0, 0, 0, 0],
            init_cash=1000.0,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        np.testing.assert_allclose(pf.asset_value.values, neg_price.values * 1.0)

    def test_portfolio_value_consistency(self):
        pf = vbt.Portfolio.from_orders(
            neg_price,
            size=[1.0, 0, 0, 0, 0],
            init_cash=1000.0,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        np.testing.assert_allclose(
            pf.value.values,
            pf.cash.values + pf.asset_value.values,
        )

    def test_fees_on_absolute_value(self):
        pf = vbt.Portfolio.from_orders(
            pd.Series([-100.0]),
            size=1.0,
            init_cash=10000.0,
            fees=0.01,
            slippage=0.0,
            fixed_fees=0.0,
        )
        orders = pf.order_records
        np.testing.assert_allclose(orders["fees"][0], 1.0)

    def test_fixed_fees_with_negative_price(self):
        pf = vbt.Portfolio.from_orders(
            pd.Series([-100.0]),
            size=1.0,
            init_cash=10000.0,
            fees=0.01,
            slippage=0.0,
            fixed_fees=5.0,
        )
        orders = pf.order_records
        np.testing.assert_allclose(orders["fees"][0], 6.0)

    def test_slippage_long_buy_negative_price(self):
        pf = vbt.Portfolio.from_orders(
            pd.Series([-100.0]),
            size=1.0,
            init_cash=10000.0,
            fees=0.0,
            slippage=0.1,
            fixed_fees=0.0,
        )
        orders = pf.order_records
        assert orders["price"][0] == pytest.approx(-90.0)

    def test_slippage_long_sell_negative_price(self):
        pf = vbt.Portfolio.from_orders(
            pd.Series([-100.0, -100.0]),
            size=[1.0, -1.0],
            init_cash=10000.0,
            fees=0.0,
            slippage=0.1,
            fixed_fees=0.0,
        )
        orders = pf.order_records
        assert orders["price"][1] == pytest.approx(-110.0)

    def test_slippage_short_sell_negative_price(self):
        pf = vbt.Portfolio.from_orders(
            pd.Series([-100.0]),
            size=1.0,
            direction="shortonly",
            init_cash=10000.0,
            fees=0.0,
            slippage=0.1,
            fixed_fees=0.0,
        )
        orders = pf.order_records
        assert orders["price"][0] == pytest.approx(-110.0)

    def test_slippage_short_buy_negative_price(self):
        pf = vbt.Portfolio.from_orders(
            pd.Series([-100.0, -100.0]),
            size=[1.0, -1.0],
            direction="shortonly",
            init_cash=10000.0,
            fees=0.0,
            slippage=0.1,
            fixed_fees=0.0,
        )
        orders = pf.order_records
        assert orders["price"][1] == pytest.approx(-90.0)

    def test_multiplier_with_negative_price(self):
        pf = vbt.Portfolio.from_orders(
            pd.Series([-100.0, -100.0]),
            size=[1.0, -1.0],
            init_cash=100000.0,
            multiplier=50,
            fees=0.01,
            slippage=0.0,
            fixed_fees=0.0,
        )
        orders = pf.order_records
        np.testing.assert_allclose(orders["fees"][0], 50.0)
        np.testing.assert_allclose(pf.asset_value.values[0], -5000.0)

    def test_size_type_value_with_negative_price(self):
        pf = vbt.Portfolio.from_orders(
            neg_price,
            size=[50.0, 0, 0, 0, -50.0],
            size_type="value",
            init_cash=1000.0,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        orders = pf.order_records
        assert orders["size"][0] == pytest.approx(5.0)

    def test_size_type_target_value_with_negative_price(self):
        pf = vbt.Portfolio.from_orders(
            neg_price,
            size=[-50.0, 0, 0, 0, 0],
            size_type="targetvalue",
            init_cash=1000.0,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        orders = pf.order_records
        assert orders["size"][0] == pytest.approx(5.0)

    def test_cash_limited_buy_negative_price(self):
        pf = vbt.Portfolio.from_orders(
            pd.Series([-10.0]),
            size=100.0,
            init_cash=10.0,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        orders = pf.order_records
        assert orders["size"][0] == pytest.approx(1.0)

    def test_mixed_positive_and_negative_prices(self):
        mixed_price = pd.Series(
            [10.0, 5.0, -5.0, -10.0, 5.0],
            index=pd.date_range("2020", periods=5),
        )
        pf = vbt.Portfolio.from_orders(
            mixed_price,
            size=[1.0, 0, 0, 0, -1.0],
            init_cash=1000.0,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        orders = pf.order_records
        assert len(orders) == 2
        assert orders["price"][0] == pytest.approx(10.0)
        assert orders["price"][1] == pytest.approx(5.0)
        trades = pf.trades.records
        np.testing.assert_allclose(trades["pnl"].values[0], -5.0)
        np.testing.assert_allclose(
            pf.value.values,
            pf.cash.values + pf.asset_value.values,
        )


class TestNegativePriceFromSignals:
    def test_long_entry_exit_negative_price(self):
        pf = vbt.Portfolio.from_signals(
            neg_price,
            entries=[True, False, False, False, False],
            exits=[False, False, False, False, True],
            size=1.0,
            size_type="amount",
            init_cash=1000.0,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        trades = pf.trades.records
        np.testing.assert_allclose(trades["pnl"].values[0], -1.0)

    def test_short_entry_exit_negative_price(self):
        pf = vbt.Portfolio.from_signals(
            neg_price,
            entries=[True, False, False, False, False],
            exits=[False, False, False, False, True],
            size=1.0,
            size_type="amount",
            direction="shortonly",
            init_cash=1000.0,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        trades = pf.trades.records
        np.testing.assert_allclose(trades["pnl"].values[0], 1.0)

    def test_signals_value_consistency_negative_price(self):
        pf = vbt.Portfolio.from_signals(
            neg_price,
            entries=[True, False, False, False, False],
            exits=[False, False, False, False, False],
            size=2.0,
            size_type="amount",
            init_cash=1000.0,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        np.testing.assert_allclose(
            pf.value.values,
            pf.cash.values + pf.asset_value.values,
        )

    def test_signals_with_fees_and_slippage_negative_price(self):
        pf = vbt.Portfolio.from_signals(
            pd.Series([-100.0, -100.0]),
            entries=[True, False],
            exits=[False, True],
            size=1.0,
            size_type="amount",
            init_cash=10000.0,
            fees=0.01,
            slippage=0.1,
            fixed_fees=0.0,
        )
        orders = pf.orders.records
        np.testing.assert_allclose(orders["price"][0], -90.0)
        np.testing.assert_allclose(orders["fees"][0], 0.9)
        np.testing.assert_allclose(orders["price"][1], -110.0)
        np.testing.assert_allclose(orders["fees"][1], 1.1)


class TestNegativePriceAnalysisAndRecords:
    def test_free_cash_reconstruction_negative_short(self):
        from numba import njit
        from vectorbtpro.portfolio.nb.core import order_nb

        prices = pd.Series([-100.0, -90.0], index=pd.date_range("2020", periods=2))
        captured_free_cash = np.empty((2, 1), dtype=np.float64)

        @njit
        def order_func_nb(c):
            if c.i == 0:
                return order_nb(size=1.0, price=c.close[c.i, c.col], direction=Direction.ShortOnly)
            if c.i == 1:
                return order_nb(size=-1.0, price=c.close[c.i, c.col], direction=Direction.ShortOnly)
            return order_nb(size=0.0)

        @njit
        def post_order_func_nb(c, captured_free_cash):
            captured_free_cash[c.i, c.col] = c.free_cash_now

        pf = vbt.Portfolio.from_order_func(
            prices,
            order_func_nb=order_func_nb,
            post_order_func_nb=post_order_func_nb,
            post_order_args=(captured_free_cash,),
            init_cash=10000.0,
        )

        np.testing.assert_allclose(pf.get_cash(free=True).values, captured_free_cash[:, 0])

    def test_trade_mfe_mae_returns_negative_prices(self):
        long_mfe = vbt.pf_nb.trade_mfe_nb(1.0, TradeDirection.Long, -10.0, -8.0, use_returns=True)
        long_mae = vbt.pf_nb.trade_mae_nb(1.0, TradeDirection.Long, -10.0, -12.0, use_returns=True)
        short_mfe = vbt.pf_nb.trade_mfe_nb(1.0, TradeDirection.Short, -10.0, -12.0, use_returns=True)
        short_mae = vbt.pf_nb.trade_mae_nb(1.0, TradeDirection.Short, -10.0, -8.0, use_returns=True)

        np.testing.assert_allclose(long_mfe, 0.2)
        np.testing.assert_allclose(long_mae, -0.2)
        np.testing.assert_allclose(short_mfe, 2.0 / 12.0)
        np.testing.assert_allclose(short_mae, -2.0 / 8.0)


class TestWTICrashScenarios:
    def test_buy_negative_sell_less_negative(self):
        prices = pd.Series([-37.0, -20.0], index=pd.date_range("2020", periods=2))
        pf = vbt.Portfolio.from_orders(
            prices,
            size=[1.0, -1.0],
            init_cash=100_000.0,
            multiplier=1000.0,
            fees=0.0,
            slippage=0.0,
        )
        np.testing.assert_allclose(pf.trades.records["pnl"].values[0], 17000.0)

    def test_buy_negative_sell_positive(self):
        prices = pd.Series([-37.0, 5.0], index=pd.date_range("2020", periods=2))
        pf = vbt.Portfolio.from_orders(
            prices,
            size=[1.0, -1.0],
            init_cash=100_000.0,
            multiplier=1000.0,
            fees=0.0,
            slippage=0.0,
        )
        np.testing.assert_allclose(pf.trades.records["pnl"].values[0], 42000.0)

    def test_buy_positive_sell_negative(self):
        prices = pd.Series([1.0, -37.0], index=pd.date_range("2020", periods=2))
        pf = vbt.Portfolio.from_orders(
            prices,
            size=[1.0, -1.0],
            init_cash=100_000.0,
            multiplier=1000.0,
            fees=0.0,
            slippage=0.0,
        )
        np.testing.assert_allclose(pf.trades.records["pnl"].values[0], -38000.0)

    def test_short_crash_profit(self):
        prices = pd.Series([1.0, -37.0], index=pd.date_range("2020", periods=2))
        pf = vbt.Portfolio.from_orders(
            prices,
            size=[1.0, -1.0],
            direction="shortonly",
            init_cash=100_000.0,
            multiplier=1000.0,
            fees=0.0,
            slippage=0.0,
        )
        np.testing.assert_allclose(pf.trades.records["pnl"].values[0], 38000.0)

    def test_negative_price_with_fees_and_slippage(self):
        prices = pd.Series([-37.0, -30.0], index=pd.date_range("2020", periods=2))
        pf = vbt.Portfolio.from_orders(
            prices,
            size=[1.0, -1.0],
            init_cash=100_000.0,
            multiplier=1000.0,
            fees=0.01,
            slippage=0.0,
        )
        expected_pnl = 7000.0 - 370.0 - 300.0
        np.testing.assert_allclose(pf.trades.records["pnl"].values[0], expected_pnl, rtol=1e-6)

    def test_multi_bar_value_tracking(self):
        neg_series = pd.Series(
            [-37.0, -40.0, -45.0, -30.0, -20.0],
            index=pd.date_range("2020", periods=5),
        )
        pf = vbt.Portfolio.from_orders(
            neg_series,
            size=[1.0, 0, 0, 0, -1.0],
            init_cash=100_000.0,
            multiplier=1000.0,
            fees=0.0,
            slippage=0.0,
        )
        cash = pf.cash.values
        position = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
        expected_values = cash + position * neg_series.values * 1000.0
        np.testing.assert_allclose(pf.value.values, expected_values, rtol=1e-9)


class TestLeverageWithNegativePrice:
    def test_long_buy_leverage_eager(self):
        prices = pd.Series([-37.0, -37.0], index=pd.date_range("2020", periods=2))
        pf = vbt.Portfolio.from_orders(
            prices,
            size=[1.0, 0],
            init_cash=100_000.0,
            multiplier=1000.0,
            leverage=10.0,
            leverage_mode="eager",
            fees=0.0,
            slippage=0.0,
        )
        orders = pf.order_records
        assert len(orders) == 1
        assert not np.any(np.isnan(pf.value.values))
        assert not np.any(np.isinf(pf.value.values))

    def test_short_sell_leverage(self):
        prices = pd.Series([-37.0, -37.0], index=pd.date_range("2020", periods=2))
        pf = vbt.Portfolio.from_orders(
            prices,
            size=[1.0, 0],
            direction="shortonly",
            init_cash=100_000.0,
            multiplier=1000.0,
            leverage=10.0,
            fees=0.0,
            slippage=0.0,
        )
        assert not np.any(np.isnan(pf.value.values))
        assert not np.any(np.isinf(pf.value.values))


class TestZeroPriceEdgeCases:
    def test_price_crossing_through_zero_no_nan(self):
        prices = pd.Series([1.0, 0.01, -0.01, -1.0], index=pd.date_range("2020", periods=4))
        pf = vbt.Portfolio.from_orders(
            prices,
            size=[1.0, 0, 0, 0],
            init_cash=100_000.0,
            fees=0.0,
            slippage=0.0,
        )
        assert not np.any(np.isnan(pf.value.values))
        assert not np.any(np.isinf(pf.value.values))
        np.testing.assert_allclose(
            pf.value.values,
            pf.cash.values + pf.asset_value.values,
        )

    def test_zero_crossing_value_change(self):
        prices = pd.Series([5.0, -3.0], index=pd.date_range("2020", periods=2))
        pf = vbt.Portfolio.from_orders(
            prices,
            size=[1.0, 0],
            init_cash=100_000.0,
            multiplier=100.0,
            fees=0.0,
            slippage=0.0,
        )
        value_change = pf.value.values[-1] - pf.value.values[0]
        np.testing.assert_allclose(value_change, -800.0)


class TestStopLimitPriceResolution:
    def test_percent_delta_resolution_with_negative_prices(self):
        np.testing.assert_allclose(nb.resolve_limit_price_nb(-10.0, 0.1, DeltaFormat.Percent, True), -11.0)
        np.testing.assert_allclose(nb.resolve_limit_price_nb(-10.0, 0.1, DeltaFormat.Percent, False), -9.0)
        np.testing.assert_allclose(nb.resolve_stop_price_nb(-10.0, 0.1, DeltaFormat.Percent, True), -11.0)
        np.testing.assert_allclose(nb.resolve_stop_price_nb(-10.0, 0.1, DeltaFormat.Percent, False), -9.0)

    def test_negative_stop_exit_price_literal(self):
        result = nb.resolve_stop_exit_price_nb(
            stop_price=-40.0,
            close=-35.0,
            stop_exit_price=-37.0,
        )
        np.testing.assert_allclose(result, -37.0)

    def test_stop_exit_price_sentinels_with_negative_prices(self):
        result_stop = nb.resolve_stop_exit_price_nb(
            stop_price=-40.0,
            close=-35.0,
            stop_exit_price=float(StopExitPrice.Stop),
        )
        np.testing.assert_allclose(result_stop, -40.0)

        result_close = nb.resolve_stop_exit_price_nb(
            stop_price=-40.0,
            close=-35.0,
            stop_exit_price=float(StopExitPrice.Close),
        )
        np.testing.assert_allclose(result_close, -35.0)

    def test_negative_limit_order_price_literal(self):
        result = nb.resolve_limit_order_price_nb(
            limit_price=-40.0,
            close=-35.0,
            limit_order_price=-37.0,
        )
        np.testing.assert_allclose(result, -37.0)

    def test_limit_order_price_sentinels_with_negative_prices(self):
        result_limit = nb.resolve_limit_order_price_nb(
            limit_price=-40.0,
            close=-35.0,
            limit_order_price=float(LimitOrderPrice.Limit),
        )
        np.testing.assert_allclose(result_limit, -40.0)

        result_close = nb.resolve_limit_order_price_nb(
            limit_price=-40.0,
            close=-35.0,
            limit_order_price=float(LimitOrderPrice.Close),
        )
        np.testing.assert_allclose(result_close, -35.0)

    def test_percent_stops_with_negative_prices(self):
        falling_prices = pd.Series([-10.0, -11.0], index=pd.date_range("2020", periods=2))
        entries = pd.Series([True, False], index=falling_prices.index)

        sl_pf = vbt.Portfolio.from_signals(
            falling_prices,
            entries=entries,
            exits=False,
            sl_stop=0.1,
            delta_format="percent",
            direction="longonly",
            size=1.0,
            fees=0.0,
            fixed_fees=0.0,
            slippage=0.0,
        )
        np.testing.assert_allclose(sl_pf.asset_flow.values, np.array([1.0, -1.0]))
        np.testing.assert_allclose(sl_pf.orders.records["price"].values, np.array([-10.0, -11.0]))

        rising_prices = pd.Series([-10.0, -9.0], index=pd.date_range("2020", periods=2))
        entries = pd.Series([True, False], index=rising_prices.index)

        tp_pf = vbt.Portfolio.from_signals(
            rising_prices,
            entries=entries,
            exits=False,
            tp_stop=0.1,
            delta_format="percent",
            direction="longonly",
            size=1.0,
            fees=0.0,
            fixed_fees=0.0,
            slippage=0.0,
        )
        np.testing.assert_allclose(tp_pf.asset_flow.values, np.array([1.0, -1.0]))
        np.testing.assert_allclose(tp_pf.orders.records["price"].values, np.array([-10.0, -9.0]))
