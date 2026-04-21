import os

from numba import njit

import vectorbtpro as vbt
from tests.utils import *
from vectorbtpro._dtypes import *
from vectorbtpro.utils import array_

seed = 42


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True


def teardown_module():
    vbt.settings.reset()


class TestArray:
    def test_is_sorted(self):
        assert array_.is_sorted(np.array([0, 1, 2, 3, 4]))
        assert array_.is_sorted(np.array([0, 1]))
        assert array_.is_sorted(np.array([0]))
        assert not array_.is_sorted(np.array([1, 0]))
        assert not array_.is_sorted(np.array([0, 1, 2, 4, 3]))
        # nb
        assert array_.is_sorted_nb(np.array([0, 1, 2, 3, 4]))
        assert array_.is_sorted_nb(np.array([0, 1]))
        assert array_.is_sorted_nb(np.array([0]))
        assert not array_.is_sorted_nb(np.array([1, 0]))
        assert not array_.is_sorted_nb(np.array([0, 1, 2, 4, 3]))

    def test_insert_argsort_nb(self):
        a = np.random.uniform(size=1000)
        A = a.copy()
        I = np.arange(len(A))
        array_.insert_argsort_nb(A, I)
        np.testing.assert_array_equal(np.sort(a), A)
        np.testing.assert_array_equal(a[I], A)

    def test_get_ranges_arr(self):
        np.testing.assert_array_equal(array_.get_ranges_arr(0, 3), np.array([0, 1, 2]))
        np.testing.assert_array_equal(array_.get_ranges_arr(0, [1, 2, 3]), np.array([0, 0, 1, 0, 1, 2]))
        np.testing.assert_array_equal(array_.get_ranges_arr([0, 3], [3, 6]), np.array([0, 1, 2, 3, 4, 5]))

    def test_uniform_summing_to_one_nb(self):
        @njit
        def set_seed():
            np.random.seed(seed)

        set_seed()
        np.testing.assert_array_almost_equal(
            array_.uniform_summing_to_one_nb(10),
            np.array(
                [
                    5.808361e-02,
                    9.791091e-02,
                    2.412011e-05,
                    2.185215e-01,
                    2.241184e-01,
                    2.456528e-03,
                    1.308789e-01,
                    1.341822e-01,
                    8.453816e-02,
                    4.928569e-02,
                ]
            ),
        )
        assert np.sum(array_.uniform_summing_to_one_nb(10)) == 1

    def test_rescale(self):
        assert array_.rescale(0, (0, 10), (0, 1)) == 0
        assert array_.rescale(10, (0, 10), (0, 1)) == 1
        np.testing.assert_array_equal(
            array_.rescale(np.array([0, 2, 4, 6, 8, 10]), (0, 10), (0, 1)),
            np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        )
        np.testing.assert_array_equal(
            array_.rescale_nb(np.array([0, 2, 4, 6, 8, 10]), (0, 10), (0, 1)),
            np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        )

    def test_min_rel_rescale(self):
        np.testing.assert_array_equal(
            array_.min_rel_rescale(np.array([2, 4, 6]), (10, 20)),
            np.array([10.0, 15.0, 20.0]),
        )
        np.testing.assert_array_equal(
            array_.min_rel_rescale(np.array([5, 6, 7]), (10, 20)),
            np.array([10.0, 12.0, 14.0]),
        )
        np.testing.assert_array_equal(
            array_.min_rel_rescale(np.array([5, 5, 5]), (10, 20)),
            np.array([10.0, 10.0, 10.0]),
        )

    def test_max_rel_rescale(self):
        np.testing.assert_array_equal(
            array_.max_rel_rescale(np.array([2, 4, 6]), (10, 20)),
            np.array([10.0, 15.0, 20.0]),
        )
        np.testing.assert_array_equal(
            array_.max_rel_rescale(np.array([5, 6, 7]), (10, 20)),
            np.array([14.285714285714286, 17.142857142857142, 20.0]),
        )
        np.testing.assert_array_equal(
            array_.max_rel_rescale(np.array([5, 5, 5]), (10, 20)),
            np.array([20.0, 20.0, 20.0]),
        )

    def test_rescale_float_to_int_nb(self):
        @njit
        def set_seed():
            np.random.seed(seed)

        set_seed()
        np.testing.assert_array_equal(
            array_.rescale_float_to_int_nb(np.array([0.3, 0.3, 0.3, 0.1]), (10, 20), 70),
            np.array([17, 14, 22, 17]),
        )
        assert np.sum(array_.rescale_float_to_int_nb(np.array([0.3, 0.3, 0.3, 0.1]), (10, 20), 70)) == 70
