import os
from collections import namedtuple

import pytest
from numba import njit

import vectorbtpro as vbt
from tests.utils import *
from vectorbtpro._dtypes import *
from vectorbtpro.utils import checks


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True


def teardown_module():
    vbt.settings.reset()


class TestChecks:
    def test_is_np_array(self):
        assert not checks.is_np_array(0)
        assert checks.is_np_array(np.array([0]))
        assert not checks.is_np_array(pd.Series([1, 2, 3]))
        assert not checks.is_np_array(pd.DataFrame([1, 2, 3]))

    def test_is_pandas(self):
        assert not checks.is_pandas(0)
        assert not checks.is_pandas(np.array([0]))
        assert checks.is_pandas(pd.Series([1, 2, 3]))
        assert checks.is_pandas(pd.DataFrame([1, 2, 3]))

    def test_is_series(self):
        assert not checks.is_series(0)
        assert not checks.is_series(np.array([0]))
        assert checks.is_series(pd.Series([1, 2, 3]))
        assert not checks.is_series(pd.DataFrame([1, 2, 3]))

    def test_is_frame(self):
        assert not checks.is_frame(0)
        assert not checks.is_frame(np.array([0]))
        assert not checks.is_frame(pd.Series([1, 2, 3]))
        assert checks.is_frame(pd.DataFrame([1, 2, 3]))

    def test_is_array(self):
        assert not checks.is_any_array(0)
        assert checks.is_any_array(np.array([0]))
        assert checks.is_any_array(pd.Series([1, 2, 3]))
        assert checks.is_any_array(pd.DataFrame([1, 2, 3]))

    def test_is_sequence(self):
        assert checks.is_sequence([1, 2, 3])
        assert checks.is_sequence("123")
        assert not checks.is_sequence(0)
        assert not checks.is_sequence(dict(a=2).items())

    def test_is_iterable(self):
        assert checks.is_iterable([1, 2, 3])
        assert checks.is_iterable("123")
        assert not checks.is_iterable(0)
        assert checks.is_iterable(dict(a=2).items())

    def test_is_numba_func(self):
        def test_func(x):
            return x

        @njit
        def test_func_nb(x):
            return x

        assert not checks.is_numba_func(test_func)
        assert checks.is_numba_func(test_func_nb)

    def test_is_hashable(self):
        assert checks.is_hashable(2)
        assert not checks.is_hashable(np.asarray(2))

    def test_is_index_equal(self):
        assert checks.is_index_equal(pd.Index([0]), pd.Index([0]))
        assert not checks.is_index_equal(pd.Index([0]), pd.Index([1]))
        assert not checks.is_index_equal(pd.Index([0], name="name"), pd.Index([0]))
        assert checks.is_index_equal(pd.Index([0], name="name"), pd.Index([0]), check_names=False)
        assert not checks.is_index_equal(pd.MultiIndex.from_arrays([[0], [1]]), pd.Index([0]))
        assert checks.is_index_equal(pd.MultiIndex.from_arrays([[0], [1]]), pd.MultiIndex.from_arrays([[0], [1]]))
        assert checks.is_index_equal(
            pd.MultiIndex.from_arrays([[0], [1]], names=["name1", "name2"]),
            pd.MultiIndex.from_arrays([[0], [1]], names=["name1", "name2"]),
        )
        assert not checks.is_index_equal(
            pd.MultiIndex.from_arrays([[0], [1]], names=["name1", "name2"]),
            pd.MultiIndex.from_arrays([[0], [1]], names=["name3", "name4"]),
        )

    def test_is_default_index(self):
        assert checks.is_default_index(pd.DataFrame([[1, 2, 3]]).columns)
        assert checks.is_default_index(pd.Series([1, 2, 3]).to_frame().columns)
        assert checks.is_default_index(pd.Index([0, 1, 2]))
        assert not checks.is_default_index(pd.Index([0, 1, 2], name="name"))

    def test_is_equal(self):
        assert checks.is_equal(np.arange(3), np.arange(3), np.array_equal)
        assert not checks.is_equal(np.arange(3), None, np.array_equal)
        assert not checks.is_equal(None, np.arange(3), np.array_equal)
        assert checks.is_equal(None, None, np.array_equal)

    def test_is_namedtuple(self):
        assert checks.is_namedtuple(namedtuple("Hello", ["world"])(*range(1)))
        assert not checks.is_namedtuple((0,))

    def test_func_accepts_arg(self):
        def test(a, *args, b=2, **kwargs):
            pass

        assert checks.func_accepts_arg(test, "a")
        assert not checks.func_accepts_arg(test, "args")
        assert checks.func_accepts_arg(test, "*args")
        assert checks.func_accepts_arg(test, "b")
        assert not checks.func_accepts_arg(test, "kwargs")
        assert checks.func_accepts_arg(test, "**kwargs")
        assert not checks.func_accepts_arg(test, "c")

    def test_is_deep_equal(self):
        sr = pd.Series([1, 2, 3], index=pd.Index(["a", "b", "c"], name="index"), name="name")
        sr2 = pd.Series([1.0, 2.0, 3.0], index=sr.index, name=sr.name)
        sr3 = pd.Series([np.nan, 2.0, 3.0], index=sr.index, name=sr.name)
        sr4 = pd.Series([np.nan, 2.0, 3.0 + 1e-15], index=sr.index, name=sr.name)
        assert checks.is_deep_equal(sr, sr.copy())
        assert checks.is_deep_equal(sr2, sr2.copy())
        assert checks.is_deep_equal(sr3, sr3.copy())
        assert checks.is_deep_equal(sr4, sr4.copy())
        assert not checks.is_deep_equal(sr, sr2)
        assert checks.is_deep_equal(sr3, sr4)
        assert not checks.is_deep_equal(sr3, sr4, rtol=0, atol=1e-16)
        assert not checks.is_deep_equal(sr3, sr4, check_exact=True)
        assert not checks.is_deep_equal(sr, sr.rename("name2"))
        assert checks.is_deep_equal(sr.index, sr.copy().index)
        assert not checks.is_deep_equal(sr.index, sr.copy().index[:-1])
        assert not checks.is_deep_equal(sr.index, sr.copy().rename("indx2"))
        assert checks.is_deep_equal(sr.to_frame(), sr.to_frame().copy())
        assert not checks.is_deep_equal(sr, sr.to_frame().copy())
        assert not checks.is_deep_equal(sr.to_frame(), sr.copy())

        arr = np.array([1, 2, 3])
        arr2 = np.array([1.0, 2.0, 3.0])
        arr3 = np.array([np.nan, 2.0, 3.0])
        arr4 = np.array([np.nan, 2.0, 3 + 1e-15])
        assert checks.is_deep_equal(arr, arr.copy())
        assert checks.is_deep_equal(arr2, arr2.copy())
        assert checks.is_deep_equal(arr3, arr3.copy())
        assert checks.is_deep_equal(arr4, arr4.copy())
        assert not checks.is_deep_equal(arr, arr2)
        assert checks.is_deep_equal(arr3, arr4)
        assert not checks.is_deep_equal(arr3, arr4, rtol=0, atol=1e-16)
        assert not checks.is_deep_equal(arr3, arr4, check_exact=True)

        records_arr = np.asarray(
            [
                (1, 1.0),
                (2, 2.0),
                (3, 3.0),
            ],
            dtype=np.dtype([("a", np.int32), ("b", np.float64)]),
        )
        records_arr2 = np.asarray(
            [
                (1.0, 1.0),
                (2.0, 2.0),
                (3.0, 3.0),
            ],
            dtype=np.dtype([("a", np.float64), ("b", np.float64)]),
        )
        records_arr3 = np.asarray(
            [
                (np.nan, 1.0),
                (2.0, 2.0),
                (3.0, 3.0),
            ],
            dtype=np.dtype([("a", np.float64), ("b", np.float64)]),
        )
        records_arr4 = np.asarray(
            [
                (np.nan, 1.0),
                (2.0, 2.0),
                (3.0 + 1e-15, 3.0),
            ],
            dtype=np.dtype([("a", np.float64), ("b", np.float64)]),
        )
        assert checks.is_deep_equal(records_arr, records_arr.copy())
        assert checks.is_deep_equal(records_arr2, records_arr2.copy())
        assert checks.is_deep_equal(records_arr3, records_arr3.copy())
        assert checks.is_deep_equal(records_arr4, records_arr4.copy())
        assert not checks.is_deep_equal(records_arr, records_arr2)
        assert checks.is_deep_equal(records_arr3, records_arr4)
        assert not checks.is_deep_equal(records_arr3, records_arr4, rtol=0, atol=1e-16)
        assert not checks.is_deep_equal(records_arr3, records_arr4, check_exact=True)

        assert checks.is_deep_equal([sr, arr, records_arr], [sr, arr, records_arr])
        assert not checks.is_deep_equal([sr, arr, records_arr], [sr, arr, records_arr2])
        assert not checks.is_deep_equal([sr, arr, records_arr], [sr, records_arr, arr])
        assert checks.is_deep_equal(
            {"sr": sr, "arr": arr, "records_arr": records_arr},
            {"sr": sr, "arr": arr, "records_arr": records_arr},
        )
        assert not checks.is_deep_equal(
            {"sr": sr, "arr": arr, "records_arr": records_arr},
            {"sr": sr, "arr": arr, "records_arr2": records_arr},
        )
        assert not checks.is_deep_equal(
            {"sr": sr, "arr": arr, "records_arr": records_arr},
            {"sr": sr, "arr": arr, "records_arr": records_arr2},
        )

        assert checks.is_deep_equal(0, 0)
        assert not checks.is_deep_equal(0, False)
        assert not checks.is_deep_equal(0, 1)
        assert not checks.is_deep_equal(lambda x: x, lambda x: x)
        assert not checks.is_deep_equal(lambda x: x, lambda x: 2 * x)
        y = lambda x: x + 1
        assert checks.is_deep_equal(y, y)

    def test_is_instance_of(self):
        class _A:
            pass

        class A:
            pass

        class B:
            pass

        class C(B):
            pass

        class D(A, C):
            pass

        d = D()

        assert not checks.is_instance_of(d, _A)
        assert checks.is_instance_of(d, A)
        assert checks.is_instance_of(d, B)
        assert checks.is_instance_of(d, C)
        assert checks.is_instance_of(d, D)
        assert checks.is_instance_of(d, object)

        assert not checks.is_instance_of(d, "_A")
        assert checks.is_instance_of(d, "A")
        assert checks.is_instance_of(d, "B")
        assert checks.is_instance_of(d, "C")
        assert checks.is_instance_of(d, "D")
        assert checks.is_instance_of(d, "object")

    def test_is_subclass_of(self):
        class _A:
            pass

        class A:
            pass

        class B:
            pass

        class C(B):
            pass

        class D(A, C):
            pass

        assert not checks.is_subclass_of(D, _A)
        assert checks.is_subclass_of(D, A)
        assert checks.is_subclass_of(D, B)
        assert checks.is_subclass_of(D, C)
        assert checks.is_subclass_of(D, D)
        assert checks.is_subclass_of(D, object)

        assert not checks.is_subclass_of(D, "_A")
        assert checks.is_subclass_of(D, "A")
        assert checks.is_subclass_of(D, "B")
        assert checks.is_subclass_of(D, "C")
        assert checks.is_subclass_of(D, "D")
        assert checks.is_subclass_of(D, "object")

        assert not checks.is_subclass_of(D, vbt.Regex("_A"))
        assert checks.is_subclass_of(D, vbt.Regex("[A-D]"))
        assert not checks.is_subclass_of(D, vbt.Regex("[E-F]"))
        assert checks.is_subclass_of(D, vbt.Regex("object"))

    def test_assert_in(self):
        checks.assert_in(0, (0, 1))
        with pytest.raises(Exception):
            checks.assert_in(2, (0, 1))

    def test_assert_numba_func(self):
        def test_func(x):
            return x

        @njit
        def test_func_nb(x):
            return x

        checks.assert_numba_func(test_func_nb)
        with pytest.raises(Exception):
            checks.assert_numba_func(test_func)

    def test_assert_not_none(self):
        checks.assert_not_none(0)
        with pytest.raises(Exception):
            checks.assert_not_none(None)

    def test_assert_type(self):
        checks.assert_instance_of(0, int)
        checks.assert_instance_of(np.zeros(1), (np.ndarray, pd.Series))
        checks.assert_instance_of(pd.Series([1, 2, 3]), (np.ndarray, pd.Series))
        with pytest.raises(Exception):
            checks.assert_instance_of(pd.DataFrame([1, 2, 3]), (np.ndarray, pd.Series))

    def test_assert_subclass_of(self):
        class A:
            pass

        class B(A):
            pass

        class C(B):
            pass

        checks.assert_subclass_of(B, A)
        checks.assert_subclass_of(C, B)
        checks.assert_subclass_of(C, A)
        with pytest.raises(Exception):
            checks.assert_subclass_of(A, B)

    def test_assert_type_equal(self):
        checks.assert_type_equal(0, 1)
        checks.assert_type_equal(np.zeros(1), np.empty(1))
        with pytest.raises(Exception):
            checks.assert_instance_of(0, np.zeros(1))

    def test_assert_dtype(self):
        checks.assert_dtype(np.zeros(1), float_)
        checks.assert_dtype(pd.Series([1, 2, 3]), int_)
        checks.assert_dtype(pd.DataFrame({"a": [1, 2], "b": [3, 4]}), int_)
        with pytest.raises(Exception):
            checks.assert_dtype(pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]}), int_)

    def test_assert_subdtype(self):
        checks.assert_subdtype([0], np.number)
        checks.assert_subdtype(np.array([1, 2, 3]), np.number)
        checks.assert_subdtype(pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]}), np.number)
        with pytest.raises(Exception):
            checks.assert_subdtype(np.array([1, 2, 3]), np.floating)
        with pytest.raises(Exception):
            checks.assert_subdtype(pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]}), np.floating)

    def test_assert_dtype_equal(self):
        checks.assert_dtype_equal([1], [1, 1, 1])
        checks.assert_dtype_equal(pd.Series([1, 2, 3]), pd.DataFrame([[1, 2, 3]]))
        checks.assert_dtype_equal(pd.DataFrame([[1, 2, 3.0]]), pd.DataFrame([[1, 2, 3.0]]))
        with pytest.raises(Exception):
            checks.assert_dtype_equal(pd.DataFrame([[1, 2, 3]]), pd.DataFrame([[1, 2, 3.0]]))

    def test_assert_ndim(self):
        checks.assert_ndim(0, 0)
        checks.assert_ndim(np.zeros(1), 1)
        checks.assert_ndim(pd.Series([1, 2, 3]), (1, 2))
        checks.assert_ndim(pd.DataFrame([1, 2, 3]), (1, 2))
        with pytest.raises(Exception):
            checks.assert_ndim(np.zeros((3, 3, 3)), (1, 2))

    def test_assert_len_equal(self):
        checks.assert_len_equal([[1]], [[2]])
        checks.assert_len_equal([[1]], [[2, 3]])
        with pytest.raises(Exception):
            checks.assert_len_equal([[1]], [[2], [3]])

    def test_assert_shape_equal(self):
        checks.assert_shape_equal(0, 1)
        checks.assert_shape_equal([1, 2, 3], np.array([1, 2, 3]))
        checks.assert_shape_equal([1, 2, 3], pd.Series([1, 2, 3]))
        checks.assert_shape_equal(np.zeros((3, 3)), pd.Series([1, 2, 3]), axis=0)
        checks.assert_shape_equal(np.zeros((2, 3)), pd.Series([1, 2, 3]), axis=(1, 0))
        with pytest.raises(Exception):
            checks.assert_shape_equal(np.zeros((2, 3)), pd.Series([1, 2, 3]), axis=(0, 1))

    def test_assert_index_equal(self):
        checks.assert_index_equal(pd.Index([1, 2, 3]), pd.Index([1, 2, 3]))
        with pytest.raises(Exception):
            checks.assert_index_equal(pd.Index([1, 2, 3]), pd.Index([2, 3, 4]))

    def test_assert_meta_equal(self):
        index = ["x", "y", "z"]
        columns = ["a", "b", "c"]
        checks.assert_meta_equal(np.array([1, 2, 3]), np.array([1, 2, 3]))
        checks.assert_meta_equal(pd.Series([1, 2, 3], index=index), pd.Series([1, 2, 3], index=index))
        checks.assert_meta_equal(pd.DataFrame([[1, 2, 3]], columns=columns), pd.DataFrame([[1, 2, 3]], columns=columns))
        with pytest.raises(Exception):
            checks.assert_meta_equal(pd.Series([1, 2]), pd.DataFrame([1, 2]))

        with pytest.raises(Exception):
            checks.assert_meta_equal(pd.DataFrame([1, 2]), pd.DataFrame([1, 2, 3]))

        with pytest.raises(Exception):
            checks.assert_meta_equal(pd.DataFrame([1, 2, 3]), pd.DataFrame([1, 2, 3], index=index))

        with pytest.raises(Exception):
            checks.assert_meta_equal(pd.DataFrame([[1, 2, 3]]), pd.DataFrame([[1, 2, 3]], columns=columns))

    def test_assert_array_equal(self):
        index = ["x", "y", "z"]
        columns = ["a", "b", "c"]
        checks.assert_array_equal(np.array([1, 2, 3]), np.array([1, 2, 3]))
        checks.assert_array_equal(pd.Series([1, 2, 3], index=index), pd.Series([1, 2, 3], index=index))
        checks.assert_array_equal(
            pd.DataFrame([[1, 2, 3]], columns=columns),
            pd.DataFrame([[1, 2, 3]], columns=columns),
        )
        with pytest.raises(Exception):
            checks.assert_array_equal(np.array([1, 2]), np.array([1, 2, 3]))

    def test_assert_level_not_exists(self):
        i = pd.Index(["x", "y", "z"], name="i")
        multi_i = pd.MultiIndex.from_arrays([["x", "y", "z"], ["x2", "y2", "z2"]], names=["i", "i2"])
        checks.assert_level_not_exists(i, "i2")
        checks.assert_level_not_exists(multi_i, "i3")
        with pytest.raises(Exception):
            checks.assert_level_not_exists(i, "i")
            checks.assert_level_not_exists(multi_i, "i")

    def test_assert_equal(self):
        checks.assert_equal(0, 0)
        checks.assert_equal(False, False)
        with pytest.raises(Exception):
            checks.assert_equal(0, 1)

    def test_assert_dict_valid(self):
        checks.assert_dict_valid(dict(a=2, b=3), [["a", "b", "c"]])
        with pytest.raises(Exception):
            checks.assert_dict_valid(dict(a=2, b=3, d=4), [["a", "b", "c"]])
        checks.assert_dict_valid(dict(a=2, b=3, c=dict(d=4, e=5)), [["a", "b", "c"], ["d", "e"]])
        with pytest.raises(Exception):
            checks.assert_dict_valid(dict(a=2, b=3, c=dict(d=4, f=5)), [["a", "b", "c"], ["d", "e"]])
