import os

import pytest

import vectorbtpro as vbt
from tests.utils import *
from vectorbtpro._dtypes import *
from vectorbtpro.utils import attr_


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True


def teardown_module():
    vbt.settings.reset()


class TestAttr:
    def test_deep_getattr(self):
        class A:
            def a(self, x, y=None):
                return x + y

        class B:
            def a(self):
                return A()

            def b(self, x):
                return x

            @property
            def b_prop(self):
                return 1

        class C:
            @property
            def b(self):
                return B()

            @property
            def c(self):
                return 0

        with pytest.raises(Exception):
            attr_.deep_getattr(A(), "a")
        with pytest.raises(Exception):
            attr_.deep_getattr(A(), ("a",))
        with pytest.raises(Exception):
            attr_.deep_getattr(A(), ("a", 1))
        with pytest.raises(Exception):
            attr_.deep_getattr(A(), ("a", (1,)))
        assert attr_.deep_getattr(A(), ("a", (1,), {"y": 1})) == 2
        assert attr_.deep_getattr(C(), "c") == 0
        assert attr_.deep_getattr(C(), ["c"]) == 0
        assert attr_.deep_getattr(C(), ["b", ("b", (1,))]) == 1
        assert attr_.deep_getattr(C(), "b.b(1)") == 1
        assert attr_.deep_getattr(C(), ["b", ("a",), ("a", (1,), {"y": 1})]) == 2
        assert attr_.deep_getattr(C(), "b.a().a(1, y=1)") == 2
        assert attr_.deep_getattr(C(), "b.b_prop") == 1
        assert callable(attr_.deep_getattr(C(), "b.a.a", call_last_attr=False))
