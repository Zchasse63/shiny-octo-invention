import os
from collections import namedtuple

import pytest

import vectorbtpro as vbt
from tests.utils import *
from vectorbtpro._dtypes import *
from vectorbtpro.utils import enum_

Enum = namedtuple("Enum", ["Attr1", "Attr2"])(*range(2))


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True


def teardown_module():
    vbt.settings.reset()


class TestEnum:
    def test_map_enum_fields(self):
        assert enum_.map_enum_fields(0, Enum) == 0
        assert enum_.map_enum_fields(10, Enum) == 10
        with pytest.raises(Exception):
            enum_.map_enum_fields(10.0, Enum)
        assert enum_.map_enum_fields("Attr1", Enum) == 0
        assert enum_.map_enum_fields("attr1", Enum) == 0
        with pytest.raises(Exception):
            enum_.map_enum_fields("hello", Enum)
        assert enum_.map_enum_fields("attr1", Enum) == 0
        assert enum_.map_enum_fields(("attr1", "attr2"), Enum) == (0, 1)
        assert enum_.map_enum_fields([["attr1", "attr2"]], Enum) == [[0, 1]]
        np.testing.assert_array_equal(enum_.map_enum_fields(np.array([]), Enum), np.array([]))
        with pytest.raises(Exception):
            enum_.map_enum_fields(np.array([[0.0, 1.0]]), Enum)
        with pytest.raises(Exception):
            enum_.map_enum_fields(np.array([[False, True]]), Enum)
        np.testing.assert_array_equal(enum_.map_enum_fields(np.array([[0, 1]]), Enum), np.array([[0, 1]]))
        np.testing.assert_array_equal(enum_.map_enum_fields(np.array([["attr1", "attr2"]]), Enum), np.array([[0, 1]]))
        with pytest.raises(Exception):
            enum_.map_enum_fields(np.array([["attr1", 1]]), Enum)
        assert_series_equal(enum_.map_enum_fields(pd.Series([]), Enum), pd.Series([]))
        with pytest.raises(Exception):
            enum_.map_enum_fields(pd.Series([0.0, 1.0]), Enum)
        with pytest.raises(Exception):
            enum_.map_enum_fields(pd.Series([False, True]), Enum)
        assert_series_equal(enum_.map_enum_fields(pd.Series([0, 1]), Enum), pd.Series([0, 1]))
        assert_series_equal(enum_.map_enum_fields(pd.Series(["attr1", "attr2"]), Enum), pd.Series([0, 1]))
        with pytest.raises(Exception):
            enum_.map_enum_fields(pd.Series(["attr1", 0]), Enum)
        assert_frame_equal(enum_.map_enum_fields(pd.DataFrame([]), Enum), pd.DataFrame([]))
        with pytest.raises(Exception):
            enum_.map_enum_fields(pd.DataFrame([[0.0, 1.0]]), Enum)
        assert_frame_equal(enum_.map_enum_fields(pd.DataFrame([[0, 1]]), Enum), pd.DataFrame([[0, 1]]))
        assert_frame_equal(
            enum_.map_enum_fields(pd.DataFrame([["attr1", "attr2"]]), Enum),
            pd.DataFrame([[0, 1]]),
        )
        assert_frame_equal(enum_.map_enum_fields(pd.DataFrame([[0, "attr2"]]), Enum), pd.DataFrame([[0, 1]]))

    def test_map_enum_values(self):
        assert enum_.map_enum_values(0, Enum) == "Attr1"
        assert enum_.map_enum_values(-1, Enum) is None
        with pytest.raises(Exception):
            enum_.map_enum_values(-2, Enum)
        assert enum_.map_enum_values((0, 1, "Attr3"), Enum) == ("Attr1", "Attr2", "Attr3")
        assert enum_.map_enum_values([[0, 1, "Attr3"]], Enum) == [["Attr1", "Attr2", "Attr3"]]
        assert enum_.map_enum_values("hello", Enum) == "hello"
        np.testing.assert_array_equal(enum_.map_enum_values(np.array([]), Enum), np.array([]))
        np.testing.assert_array_equal(
            enum_.map_enum_values(np.array([[0.0, 1.0]]), Enum),
            np.array([["Attr1", "Attr2"]]),
        )
        np.testing.assert_array_equal(
            enum_.map_enum_values(np.array([["Attr1", "Attr2"]]), Enum),
            np.array([["Attr1", "Attr2"]]),
        )
        np.testing.assert_array_equal(enum_.map_enum_values(np.array([[0, "Attr2"]]), Enum), np.array([["0", "Attr2"]]))
        assert_series_equal(enum_.map_enum_values(pd.Series([]), Enum), pd.Series([]))
        assert_series_equal(
            enum_.map_enum_values(pd.Series([0.0, 1.0]), Enum),
            pd.Series(["Attr1", "Attr2"]),
        )
        assert_series_equal(enum_.map_enum_values(pd.Series([0, 1]), Enum), pd.Series(["Attr1", "Attr2"]))
        assert_series_equal(
            enum_.map_enum_values(pd.Series(["Attr1", "Attr2"]), Enum),
            pd.Series(["Attr1", "Attr2"]),
        )
        with pytest.raises(Exception):
            enum_.map_enum_values(pd.Series([0, "Attr2"]), Enum)
        assert_frame_equal(enum_.map_enum_values(pd.DataFrame([]), Enum), pd.DataFrame([]))
        assert_frame_equal(
            enum_.map_enum_values(pd.DataFrame([[0.0, 1.0]]), Enum),
            pd.DataFrame([["Attr1", "Attr2"]]),
        )
        assert_frame_equal(
            enum_.map_enum_values(pd.DataFrame([[0, 1]]), Enum),
            pd.DataFrame([["Attr1", "Attr2"]]),
        )
        assert_frame_equal(
            enum_.map_enum_values(pd.DataFrame([["Attr1", "Attr2"]]), Enum),
            pd.DataFrame([["Attr1", "Attr2"]]),
        )
        assert_frame_equal(
            enum_.map_enum_values(pd.DataFrame([[0, "Attr2"]]), Enum),
            pd.DataFrame([["Attr1", "Attr2"]]),
        )
