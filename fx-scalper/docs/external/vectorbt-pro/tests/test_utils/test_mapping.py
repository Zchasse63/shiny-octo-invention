import os
from collections import namedtuple

import pytest

import vectorbtpro as vbt
from tests.utils import *
from vectorbtpro._dtypes import *
from vectorbtpro.utils import mapping


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True


def teardown_module():
    vbt.settings.reset()


Enum = namedtuple("Enum", ["Attr1", "Attr2"])(*range(2))


class TestMapping:
    def test_to_value_mapping(self):
        assert mapping.to_value_mapping(Enum) == {0: "Attr1", 1: "Attr2", -1: None}
        assert mapping.to_value_mapping(Enum, reverse=True) == {"Attr1": 0, "Attr2": 1, None: -1}
        assert mapping.to_value_mapping({0: "Attr1", 1: "Attr2", -1: None}) == {0: "Attr1", 1: "Attr2", -1: None}
        assert mapping.to_value_mapping(["Attr1", "Attr2"]) == {0: "Attr1", 1: "Attr2"}
        assert mapping.to_value_mapping(pd.Index(["Attr1", "Attr2"])) == {0: "Attr1", 1: "Attr2"}
        assert mapping.to_value_mapping(pd.Series(["Attr1", "Attr2"])) == {0: "Attr1", 1: "Attr2"}

    def test_apply_mapping(self):
        assert mapping.apply_mapping("Attr1", mapping_like=Enum, reverse=True) == 0
        with pytest.raises(Exception):
            mapping.apply_mapping("Attr3", mapping_like=Enum, reverse=True)
        assert mapping.apply_mapping("attr1", mapping_like=Enum, reverse=True, ignore_case=True) == 0
        with pytest.raises(Exception):
            mapping.apply_mapping("attr1", mapping_like=Enum, reverse=True, ignore_case=False)
        assert mapping.apply_mapping("Attr_1", mapping_like=Enum, reverse=True, ignore_underscores=True) == 0
        with pytest.raises(Exception):
            mapping.apply_mapping("Attr_1", mapping_like=Enum, reverse=True, ignore_underscores=False)
        assert (
            mapping.apply_mapping("attr_1", mapping_like=Enum, reverse=True, ignore_case=True, ignore_underscores=True)
            == 0
        )
        with pytest.raises(Exception):
            mapping.apply_mapping("attr_1", mapping_like=Enum, reverse=True, ignore_case=True, ignore_underscores=False)
        assert mapping.apply_mapping(np.array([1]), mapping_like={1: "hello"})[0] == "hello"
        assert mapping.apply_mapping(np.array([1]), mapping_like={1.0: "hello"})[0] == "hello"
        assert mapping.apply_mapping(np.array([1.0]), mapping_like={1: "hello"})[0] == "hello"
        assert mapping.apply_mapping(np.array([True]), mapping_like={1: "hello"})[0] == "hello"
        assert mapping.apply_mapping(np.array([True]), mapping_like={True: "hello"})[0] == "hello"
        with pytest.raises(Exception):
            mapping.apply_mapping(np.array([True]), mapping_like={"world": "hello"})
        with pytest.raises(Exception):
            mapping.apply_mapping(np.array([1]), mapping_like={"world": "hello"})
        assert mapping.apply_mapping(np.array(["world"]), mapping_like={"world": "hello"})[0] == "hello"
