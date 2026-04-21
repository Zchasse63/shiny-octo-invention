import os

import vectorbtpro as vbt
from tests.utils import *
from vectorbtpro._dtypes import *
from vectorbtpro.utils import math_


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True


def teardown_module():
    vbt.settings.reset()


class TestMath:
    def test_is_close(self):
        a = 0.3
        b = 0.1 + 0.2

        # test scalar
        assert math_.is_close_nb(a, a)
        assert math_.is_close_nb(a, b)
        assert math_.is_close_nb(-a, -b)
        assert not math_.is_close_nb(-a, b)
        assert not math_.is_close_nb(a, -b)
        assert math_.is_close_nb(1e10 + a, 1e10 + b)

        # test np.nan
        assert not math_.is_close_nb(np.nan, b)
        assert not math_.is_close_nb(a, np.nan)

        # test np.inf
        assert not math_.is_close_nb(np.inf, b)
        assert not math_.is_close_nb(a, np.inf)
        assert not math_.is_close_nb(-np.inf, b)
        assert not math_.is_close_nb(a, -np.inf)
        assert not math_.is_close_nb(-np.inf, -np.inf)
        assert not math_.is_close_nb(np.inf, np.inf)
        assert not math_.is_close_nb(-np.inf, np.inf)

    def test_is_close_or_less(self):
        a = 0.3
        b = 0.1 + 0.2

        # test scalar
        assert math_.is_close_or_less_nb(a, a)
        assert math_.is_close_or_less_nb(a, b)
        assert math_.is_close_or_less_nb(-a, -b)
        assert math_.is_close_or_less_nb(-a, b)
        assert not math_.is_close_or_less_nb(a, -b)
        assert math_.is_close_or_less_nb(1e10 + a, 1e10 + b)

        # test np.nan
        assert not math_.is_close_or_less_nb(np.nan, b)
        assert not math_.is_close_or_less_nb(a, np.nan)

        # test np.inf
        assert not math_.is_close_or_less_nb(np.inf, b)
        assert math_.is_close_or_less_nb(a, np.inf)
        assert math_.is_close_or_less_nb(-np.inf, b)
        assert not math_.is_close_or_less_nb(a, -np.inf)
        assert not math_.is_close_or_less_nb(-np.inf, -np.inf)
        assert not math_.is_close_or_less_nb(np.inf, np.inf)
        assert math_.is_close_or_less_nb(-np.inf, np.inf)

    def test_is_less(self):
        a = 0.3
        b = 0.1 + 0.2

        # test scalar
        assert not math_.is_less_nb(a, a)
        assert not math_.is_less_nb(a, b)
        assert not math_.is_less_nb(-a, -b)
        assert math_.is_less_nb(-a, b)
        assert not math_.is_less_nb(a, -b)
        assert not math_.is_less_nb(1e10 + a, 1e10 + b)

        # test np.nan
        assert not math_.is_less_nb(np.nan, b)
        assert not math_.is_less_nb(a, np.nan)

        # test np.inf
        assert not math_.is_less_nb(np.inf, b)
        assert math_.is_less_nb(a, np.inf)
        assert math_.is_less_nb(-np.inf, b)
        assert not math_.is_less_nb(a, -np.inf)
        assert not math_.is_less_nb(-np.inf, -np.inf)
        assert not math_.is_less_nb(np.inf, np.inf)
        assert math_.is_less_nb(-np.inf, np.inf)

    def test_is_addition_zero(self):
        a = 0.3
        b = 0.1 + 0.2

        assert not math_.is_addition_zero_nb(a, b)
        assert math_.is_addition_zero_nb(-a, b)
        assert math_.is_addition_zero_nb(a, -b)
        assert not math_.is_addition_zero_nb(-a, -b)

    def test_add_nb(self):
        a = 0.3
        b = 0.1 + 0.2

        assert math_.add_nb(a, b) == a + b
        assert math_.add_nb(-a, b) == 0
        assert math_.add_nb(a, -b) == 0
        assert math_.add_nb(-a, -b) == -(a + b)
