import os

import numpy as np
import pandas as pd
import pytest

import vectorbtpro as vbt
from vectorbtpro.portfolio.enums import *


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True
    vbt.settings.portfolio["attach_call_seq"] = True


def teardown_module():
    vbt.settings.reset()


price = pd.Series(
    [100.0, 110.0, 105.0, 120.0, 115.0],
    index=pd.date_range("2020", periods=5),
)


class TestMultiplierFromOrders:
    def test_default_multiplier_no_change(self):
        pf1 = vbt.Portfolio.from_orders(
            price,
            size=1.0,
            init_cash=10000.0,
        )
        pf2 = vbt.Portfolio.from_orders(
            price,
            size=1.0,
            init_cash=10000.0,
            multiplier=1.0,
        )
        np.testing.assert_array_equal(pf1.order_records, pf2.order_records)
        np.testing.assert_allclose(pf1.value.values, pf2.value.values)
        np.testing.assert_allclose(pf1.returns.values, pf2.returns.values)

    def test_multiplier_scales_order_value(self):
        pf = vbt.Portfolio.from_orders(
            price,
            size=[1.0, 0, 0, 0, -1.0],
            init_cash=100000.0,
            multiplier=50,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        orders = pf.order_records
        assert len(orders) == 2
        assert orders["size"][0] == 1.0
        assert orders["price"][0] == 100.0
        assert orders["size"][1] == 1.0
        assert orders["price"][1] == 115.0

    def test_multiplier_pnl(self):
        pf = vbt.Portfolio.from_orders(
            price,
            size=[1.0, 0, 0, 0, -1.0],
            init_cash=100000.0,
            multiplier=50,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        trades = pf.trades.records
        expected_pnl = 15.0 * 1.0 * 50.0
        np.testing.assert_allclose(trades["pnl"].values[0], expected_pnl)

    def test_multiplier_asset_value(self):
        pf = vbt.Portfolio.from_orders(
            price,
            size=[1.0, 0, 0, 0, 0],
            init_cash=100000.0,
            multiplier=50,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        asset_val = pf.asset_value
        expected = np.array([5000.0, 5500.0, 5250.0, 6000.0, 5750.0])
        np.testing.assert_allclose(asset_val.values, expected)

    def test_multiplier_portfolio_value(self):
        pf = vbt.Portfolio.from_orders(
            price,
            size=[1.0, 0, 0, 0, 0],
            init_cash=100000.0,
            multiplier=50,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        value = pf.value
        cash = pf.cash
        asset_val = pf.asset_value
        np.testing.assert_allclose(value.values, cash.values + asset_val.values)

    def test_multiplier_broadcast_multi_column(self):
        price_wide = price.vbt.tile(2, keys=["ES", "NQ"])
        pf = vbt.Portfolio.from_orders(
            price_wide,
            size=1.0,
            init_cash=100000.0,
            multiplier=pd.Series([50, 20], index=["ES", "NQ"]),
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        asset_val = pf.asset_value
        np.testing.assert_allclose(asset_val.values[0, 0], 5000.0)
        np.testing.assert_allclose(asset_val.values[0, 1], 2000.0)

    def test_multiplier_fees_on_notional(self):
        pf = vbt.Portfolio.from_orders(
            pd.Series([100.0]),
            size=1.0,
            init_cash=100000.0,
            multiplier=50,
            fees=0.01,
            slippage=0.0,
            fixed_fees=0.0,
        )
        orders = pf.order_records
        np.testing.assert_allclose(orders["fees"][0], 50.0)

    def test_multiplier_with_fixed_fees(self):
        pf = vbt.Portfolio.from_orders(
            pd.Series([100.0]),
            size=1.0,
            init_cash=100000.0,
            multiplier=50,
            fees=0.01,
            slippage=0.0,
            fixed_fees=10.0,
        )
        orders = pf.order_records
        np.testing.assert_allclose(orders["fees"][0], 60.0)

    def test_multiplier_with_slippage(self):
        pf = vbt.Portfolio.from_orders(
            pd.Series([100.0]),
            size=1.0,
            init_cash=100000.0,
            multiplier=50,
            fees=0.0,
            slippage=0.01,
            fixed_fees=0.0,
        )
        orders = pf.order_records
        assert orders["price"][0] == pytest.approx(101.0)

    def test_multiplier_size_value_type(self):
        pf = vbt.Portfolio.from_orders(
            price,
            size=[5000.0, 0, 0, 0, -5000.0],
            size_type="value",
            init_cash=100000.0,
            multiplier=50,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        orders = pf.order_records
        assert orders["size"][0] == pytest.approx(1.0)


class TestMultiplierFromSignals:
    def test_from_signals_multiplier_pnl(self):
        pf = vbt.Portfolio.from_signals(
            price,
            entries=[True, False, False, False, False],
            exits=[False, False, False, False, True],
            size=1.0,
            size_type="amount",
            init_cash=100000.0,
            multiplier=50,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        trades = pf.trades.records
        expected_pnl = (115.0 - 100.0) * 1.0 * 50.0
        np.testing.assert_allclose(trades["pnl"].values[0], expected_pnl)

    def test_from_signals_default_multiplier(self):
        pf1 = vbt.Portfolio.from_signals(
            price,
            entries=[True, False, True, False, False],
            exits=[False, True, False, True, False],
            init_cash=10000.0,
        )
        pf2 = vbt.Portfolio.from_signals(
            price,
            entries=[True, False, True, False, False],
            exits=[False, True, False, True, False],
            init_cash=10000.0,
            multiplier=1.0,
        )
        np.testing.assert_allclose(pf1.value.values, pf2.value.values)

    def test_from_signals_short_multiplier(self):
        pf = vbt.Portfolio.from_signals(
            price,
            entries=[True, False, False, False, False],
            exits=[False, False, False, False, True],
            size=1.0,
            size_type="amount",
            direction="shortonly",
            init_cash=100000.0,
            multiplier=50,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        trades = pf.trades.records
        expected_pnl = (100.0 - 115.0) * 1.0 * 50.0
        np.testing.assert_allclose(trades["pnl"].values[0], expected_pnl)

    def test_from_signals_multiplier_asset_value(self):
        pf = vbt.Portfolio.from_signals(
            price,
            entries=[True, False, False, False, False],
            exits=[False, False, False, False, False],
            size=2.0,
            size_type="amount",
            init_cash=100000.0,
            multiplier=10,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        asset_val = pf.asset_value
        expected = price.values * 2.0 * 10.0
        np.testing.assert_allclose(asset_val.values, expected)


class TestMultiplierFromDefOrderFunc:
    def test_def_order_func_multiplier_pnl(self):
        pf = vbt.Portfolio.from_def_order_func(
            price,
            size=[1.0, 0, 0, 0, -1.0],
            init_cash=100000.0,
            multiplier=50,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        trades = pf.trades.records
        expected_pnl = (115.0 - 100.0) * 1.0 * 50.0
        np.testing.assert_allclose(trades["pnl"].values[0], expected_pnl)

    def test_def_order_func_default_multiplier(self):
        pf1 = vbt.Portfolio.from_def_order_func(
            price,
            size=[1.0, 0, 0, 0, -1.0],
            init_cash=10000.0,
        )
        pf2 = vbt.Portfolio.from_def_order_func(
            price,
            size=[1.0, 0, 0, 0, -1.0],
            init_cash=10000.0,
            multiplier=1.0,
        )
        np.testing.assert_allclose(pf1.value.values, pf2.value.values)

    def test_def_order_func_multiplier_asset_value(self):
        pf = vbt.Portfolio.from_def_order_func(
            price,
            size=[1.0, 0, 0, 0, 0],
            init_cash=100000.0,
            multiplier=50,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        asset_val = pf.asset_value
        expected = np.array([5000.0, 5500.0, 5250.0, 6000.0, 5750.0])
        np.testing.assert_allclose(asset_val.values, expected)


class TestMultiplierFromOrderFunc:
    def test_order_func_with_multiplier(self):
        from numba import njit
        from vectorbtpro.portfolio.nb.core import order_nb

        @njit
        def order_func_nb(c):
            if c.i == 0:
                return order_nb(size=1.0, price=c.close[c.i, c.col])
            if c.i == 4:
                return order_nb(size=-1.0, price=c.close[c.i, c.col])
            return order_nb(size=0.0)

        pf = vbt.Portfolio.from_order_func(
            price,
            order_func_nb=order_func_nb,
            init_cash=100000.0,
            multiplier=50,
        )
        trades = pf.trades.records
        expected_pnl = (115.0 - 100.0) * 1.0 * 50.0
        np.testing.assert_allclose(trades["pnl"].values[0], expected_pnl)

    def test_order_func_default_multiplier(self):
        from numba import njit
        from vectorbtpro.portfolio.nb.core import order_nb

        @njit
        def order_func_nb(c):
            if c.i == 0:
                return order_nb(size=1.0, price=c.close[c.i, c.col])
            return order_nb(size=0.0)

        pf1 = vbt.Portfolio.from_order_func(
            price,
            order_func_nb=order_func_nb,
            init_cash=10000.0,
        )
        pf2 = vbt.Portfolio.from_order_func(
            price,
            order_func_nb=order_func_nb,
            init_cash=10000.0,
            multiplier=1.0,
        )
        np.testing.assert_allclose(pf1.value.values, pf2.value.values)

    def test_order_func_multiplier_cash_deducted(self):
        from numba import njit
        from vectorbtpro.portfolio.nb.core import order_nb

        @njit
        def order_func_nb(c):
            if c.i == 0:
                return order_nb(size=1.0, price=c.close[c.i, c.col])
            return order_nb(size=0.0)

        pf = vbt.Portfolio.from_order_func(
            price,
            order_func_nb=order_func_nb,
            init_cash=100000.0,
            multiplier=50,
        )
        cash = pf.cash
        np.testing.assert_allclose(cash.values[0], 95000.0)


class TestMultiplierPostSimAnalysis:
    def test_multiplier_cash_flow(self):
        pf = vbt.Portfolio.from_orders(
            price,
            size=[1.0, 0, 0, 0, -1.0],
            init_cash=100000.0,
            multiplier=50,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        cf = pf.cash_flow
        np.testing.assert_allclose(cf.values[0], -5000.0)
        np.testing.assert_allclose(cf.values[1], 0.0)
        np.testing.assert_allclose(cf.values[2], 0.0)
        np.testing.assert_allclose(cf.values[3], 0.0)
        np.testing.assert_allclose(cf.values[4], 5750.0)

    def test_multiplier_cash(self):
        pf = vbt.Portfolio.from_orders(
            price,
            size=[1.0, 0, 0, 0, 0],
            init_cash=100000.0,
            multiplier=50,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        cash = pf.cash
        np.testing.assert_allclose(cash.values[0], 95000.0)
        for i in range(1, 5):
            np.testing.assert_allclose(cash.values[i], 95000.0)

    def test_multiplier_total_profit(self):
        pf = vbt.Portfolio.from_orders(
            price,
            size=[1.0, 0, 0, 0, -1.0],
            init_cash=100000.0,
            multiplier=50,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        tp_val = pf.total_profit
        np.testing.assert_allclose(float(tp_val), 750.0)

    def test_multiplier_total_return(self):
        pf = vbt.Portfolio.from_orders(
            price,
            size=[1.0, 0, 0, 0, -1.0],
            init_cash=100000.0,
            multiplier=50,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        tr = pf.total_return
        np.testing.assert_allclose(float(tr), 750.0 / 100000.0)

    def test_multiplier_value_series(self):
        pf = vbt.Portfolio.from_orders(
            price,
            size=[1.0, 0, 0, 0, 0],
            init_cash=100000.0,
            multiplier=50,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        value = pf.value
        cash = pf.cash
        asset_val = pf.asset_value
        np.testing.assert_allclose(value.values, cash.values + asset_val.values)
        np.testing.assert_allclose(value.values[0], 100000.0)
        np.testing.assert_allclose(value.values[1], 100500.0)

    def test_multiplier_returns(self):
        pf = vbt.Portfolio.from_orders(
            price,
            size=[1.0, 0, 0, 0, 0],
            init_cash=100000.0,
            multiplier=50,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        returns = pf.returns
        np.testing.assert_allclose(returns.values[0], 0.0)
        np.testing.assert_allclose(returns.values[1], 500.0 / 100000.0)


class TestMultiplierAssetValue:
    def test_asset_value_multiple_contracts(self):
        pf = vbt.Portfolio.from_orders(
            price,
            size=[2.0, 0, 0, 0, 0],
            init_cash=100000.0,
            multiplier=10,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        asset_val = pf.asset_value
        expected = price.values * 2.0 * 10.0
        np.testing.assert_allclose(asset_val.values, expected)

    def test_multiplier_stored_on_portfolio(self):
        pf = vbt.Portfolio.from_orders(
            price,
            size=1.0,
            init_cash=100000.0,
            multiplier=50,
        )
        mult = pf.multiplier
        assert mult is not None
        np.testing.assert_allclose(np.asarray(mult).flat[0], 50.0)

    def test_asset_value_zero_position(self):
        pf = vbt.Portfolio.from_orders(
            price,
            size=[0.0, 0.0, 0.0, 0.0, 0.0],
            init_cash=100000.0,
            multiplier=50,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        asset_val = pf.asset_value
        np.testing.assert_allclose(asset_val.values, 0.0)


class TestMultiplierEdgeCases:
    def test_multiplier_scalar(self):
        pf = vbt.Portfolio.from_orders(
            price,
            size=1.0,
            init_cash=100000.0,
            multiplier=50,
        )
        assert pf.multiplier is not None

    def test_multiplier_array(self):
        price_wide = price.vbt.tile(2, keys=["a", "b"])
        pf = vbt.Portfolio.from_orders(
            price_wide,
            size=1.0,
            init_cash=100000.0,
            multiplier=pd.Series([50, 20], index=["a", "b"]),
        )
        assert pf.multiplier is not None

    def test_multiplier_int_no_type_error(self):
        pf = vbt.Portfolio.from_orders(
            price,
            size=[1.0, 0, 0, 0, 0],
            init_cash=100000.0,
            multiplier=50,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        asset_val = pf.asset_value
        expected = np.array([5000.0, 5500.0, 5250.0, 6000.0, 5750.0])
        np.testing.assert_allclose(asset_val.values, expected)

    def test_short_with_multiplier(self):
        pf = vbt.Portfolio.from_orders(
            price,
            size=[-1.0, 0, 0, 0, 1.0],
            init_cash=100000.0,
            multiplier=50,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        trades = pf.trades.records
        expected_pnl = (100.0 - 115.0) * 1.0 * 50.0
        np.testing.assert_allclose(trades["pnl"].values[0], expected_pnl)

    def test_short_with_multiplier_profit(self):
        short_price = pd.Series(
            [100.0, 90.0, 80.0],
            index=pd.date_range("2020", periods=3),
        )
        pf = vbt.Portfolio.from_orders(
            short_price,
            size=[-1.0, 0, 1.0],
            init_cash=100000.0,
            multiplier=50,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        trades = pf.trades.records
        expected_pnl = (100.0 - 80.0) * 1.0 * 50.0
        np.testing.assert_allclose(trades["pnl"].values[0], expected_pnl)

    def test_equivalence_multiplier_vs_manual_scaling(self):
        M = 50
        pf1 = vbt.Portfolio.from_orders(
            price,
            size=[1.0, 0, 0, 0, -1.0],
            init_cash=1000000.0,
            multiplier=M,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        pf2 = vbt.Portfolio.from_orders(
            price * M,
            size=[1.0, 0, 0, 0, -1.0],
            init_cash=1000000.0,
            multiplier=1.0,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        t1 = pf1.trades.records
        t2 = pf2.trades.records
        np.testing.assert_allclose(t1["pnl"].values[0], t2["pnl"].values[0])

    def test_equivalence_asset_value_vs_manual_scaling(self):
        M = 50
        pf1 = vbt.Portfolio.from_orders(
            price,
            size=[1.0, 0, 0, 0, 0],
            init_cash=1000000.0,
            multiplier=M,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        pf2 = vbt.Portfolio.from_orders(
            price * M,
            size=[1.0, 0, 0, 0, 0],
            init_cash=1000000.0,
            multiplier=1.0,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        np.testing.assert_allclose(pf1.asset_value.values, pf2.asset_value.values)

    def test_multiplier_fractional(self):
        pf = vbt.Portfolio.from_orders(
            price,
            size=[1.0, 0, 0, 0, -1.0],
            init_cash=100000.0,
            multiplier=0.1,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        trades = pf.trades.records
        expected_pnl = 15.0 * 1.0 * 0.1
        np.testing.assert_allclose(trades["pnl"].values[0], expected_pnl)

    def test_multiplier_large(self):
        pf = vbt.Portfolio.from_orders(
            price,
            size=[1.0, 0, 0, 0, -1.0],
            init_cash=10_000_000.0,
            multiplier=1000,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        trades = pf.trades.records
        expected_pnl = 15.0 * 1.0 * 1000.0
        np.testing.assert_allclose(trades["pnl"].values[0], expected_pnl)


class TestMultiplierLogRecords:
    def test_log_records_contain_multiplier(self):
        pf = vbt.Portfolio.from_orders(
            price,
            size=1.0,
            init_cash=100000.0,
            multiplier=50,
            log=True,
        )
        logs = pf.log_records
        assert "asset_spec_multiplier" in logs.dtype.names
        np.testing.assert_allclose(logs["asset_spec_multiplier"][0], 50.0)

    def test_log_records_default_multiplier(self):
        pf = vbt.Portfolio.from_orders(
            price,
            size=1.0,
            init_cash=10000.0,
            log=True,
        )
        logs = pf.log_records
        assert "asset_spec_multiplier" in logs.dtype.names
        np.testing.assert_allclose(logs["asset_spec_multiplier"][0], 1.0)


class TestMultiplierContext:
    def test_multiplier_in_order_context(self):
        from numba import njit
        from vectorbtpro.portfolio.nb.core import order_nb
        from vectorbtpro.base.flex_indexing import flex_select_1d_pc_nb

        @njit
        def order_func_nb(c):
            m = flex_select_1d_pc_nb(c.multiplier, c.col)
            if c.i == 0:
                return order_nb(size=1.0, price=c.close[c.i, c.col])
            return order_nb(size=0.0)

        pf = vbt.Portfolio.from_order_func(
            price,
            order_func_nb=order_func_nb,
            init_cash=100000.0,
            multiplier=50,
        )
        assert len(pf.order_records) > 0

    def test_multiplier_in_segment_context(self):
        from numba import njit
        from vectorbtpro.portfolio.nb.core import order_nb
        from vectorbtpro.base.flex_indexing import flex_select_1d_pc_nb

        @njit
        def pre_segment_func_nb(c):
            m = flex_select_1d_pc_nb(c.multiplier, c.from_col)
            return ()

        @njit
        def order_func_nb(c):
            if c.i == 0:
                return order_nb(size=1.0, price=c.close[c.i, c.col])
            return order_nb(size=0.0)

        pf = vbt.Portfolio.from_order_func(
            price,
            order_func_nb=order_func_nb,
            pre_segment_func_nb=pre_segment_func_nb,
            init_cash=100000.0,
            multiplier=50,
        )
        assert len(pf.order_records) > 0


class TestMultiplierAssetSpec:
    def test_asset_spec_creation(self):
        spec = AssetSpec(multiplier=1.0)
        assert spec.multiplier == 1.0
        spec50 = AssetSpec(multiplier=50.0)
        assert spec50.multiplier == 50.0

    def test_no_asset_spec_default(self):
        assert NoAssetSpec.multiplier == 1.0


class TestMultiplierMultiColumn:
    def test_different_multipliers_per_column(self):
        price_wide = price.vbt.tile(3, keys=["ES", "NQ", "YM"])
        pf = vbt.Portfolio.from_orders(
            price_wide,
            size=1.0,
            init_cash=1000000.0,
            multiplier=pd.Series([50, 20, 5], index=["ES", "NQ", "YM"]),
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        asset_val = pf.asset_value
        np.testing.assert_allclose(asset_val.values[0, 0], 5000.0)
        np.testing.assert_allclose(asset_val.values[0, 1], 2000.0)
        np.testing.assert_allclose(asset_val.values[0, 2], 500.0)

    def test_different_multipliers_pnl(self):
        price_wide = price.vbt.tile(2, keys=["ES", "NQ"])
        pf = vbt.Portfolio.from_orders(
            price_wide,
            size=pd.DataFrame(
                [[1.0, 1.0], [0, 0], [0, 0], [0, 0], [-1.0, -1.0]],
                columns=["ES", "NQ"],
                index=price.index,
            ),
            init_cash=1000000.0,
            multiplier=pd.Series([50, 20], index=["ES", "NQ"]),
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        trades = pf.trades.records
        es_trades = trades[trades["col"] == 0]
        np.testing.assert_allclose(es_trades["pnl"].values[0], 750.0)
        nq_trades = trades[trades["col"] == 1]
        np.testing.assert_allclose(nq_trades["pnl"].values[0], 300.0)

    def test_multiplier_with_cash_sharing(self):
        price_wide = price.vbt.tile(2, keys=["ES", "NQ"])
        pf = vbt.Portfolio.from_orders(
            price_wide,
            size=1.0,
            init_cash=1000000.0,
            multiplier=pd.Series([50, 20], index=["ES", "NQ"]),
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
            group_by=True,
            cash_sharing=True,
        )
        value = pf.value
        assert not np.any(np.isnan(value.values))


class TestMultiplierTrades:
    def test_trade_entry_exit_prices_unchanged(self):
        pf = vbt.Portfolio.from_orders(
            price,
            size=[1.0, 0, 0, 0, -1.0],
            init_cash=100000.0,
            multiplier=50,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        trades = pf.trades.records
        assert trades["entry_price"].values[0] == pytest.approx(100.0)
        assert trades["exit_price"].values[0] == pytest.approx(115.0)

    def test_trade_return_with_multiplier(self):
        pf = vbt.Portfolio.from_orders(
            price,
            size=[1.0, 0, 0, 0, -1.0],
            init_cash=100000.0,
            multiplier=50,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        trades = pf.trades.records
        pnl = trades["pnl"].values[0]
        entry_val = 1.0 * 100.0 * 50.0
        expected_return = pnl / entry_val
        np.testing.assert_allclose(trades["return"].values[0], expected_return)

    def test_trade_pnl_with_fees_and_multiplier(self):
        pf = vbt.Portfolio.from_orders(
            price,
            size=[1.0, 0, 0, 0, -1.0],
            init_cash=100000.0,
            multiplier=50,
            fees=0.01,
            slippage=0.0,
            fixed_fees=0.0,
        )
        trades = pf.trades.records
        expected_pnl = (5750.0 - 5000.0) - 50.0 - 57.5
        np.testing.assert_allclose(trades["pnl"].values[0], expected_pnl)

    def test_multiple_trades_with_multiplier(self):
        pf = vbt.Portfolio.from_orders(
            price,
            size=[1.0, -1.0, 1.0, -1.0, 0],
            init_cash=100000.0,
            multiplier=50,
            fees=0.0,
            slippage=0.0,
            fixed_fees=0.0,
        )
        trades = pf.trades.records
        assert len(trades) == 2
        np.testing.assert_allclose(trades["pnl"].values[0], 500.0)
        np.testing.assert_allclose(trades["pnl"].values[1], 750.0)
