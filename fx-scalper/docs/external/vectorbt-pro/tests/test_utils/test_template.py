import os
from collections import namedtuple

import pytest

import vectorbtpro as vbt
from tests.utils import *
from vectorbtpro._dtypes import *
from vectorbtpro.utils import template


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True


def teardown_module():
    vbt.settings.reset()


class TestTemplate:
    def test_sub(self):
        assert template.Sub("$hello$world", {"hello": 100}).substitute({"world": 300}) == "100300"
        assert template.Sub("$hello$world", {"hello": 100}).substitute({"hello": 200, "world": 300}) == "200300"

    def test_rep(self):
        assert template.Rep("hello", {"hello": 100}).substitute() == 100
        assert template.Rep("hello", {"hello": 100}).substitute({"hello": 200}) == 200

    def test_repeval(self):
        assert template.RepEval("hello == 100", {"hello": 100}).substitute()
        assert not template.RepEval("hello == 100", {"hello": 100}).substitute({"hello": 200})

    def test_repfunc(self):
        assert template.RepFunc(lambda hello: hello == 100, {"hello": 100}).substitute()
        assert not template.RepFunc(lambda hello: hello == 100, {"hello": 100}).substitute({"hello": 200})

    def test_substitute_templates(self):
        assert template.substitute_templates(template.Rep("hello"), {"hello": 100}) == 100
        with pytest.raises(Exception):
            template.substitute_templates(template.Rep("hello2"), {"hello": 100})
        assert isinstance(
            template.substitute_templates(template.Rep("hello2"), {"hello": 100}, strict=False), template.Rep
        )
        assert template.substitute_templates(template.Sub("$hello"), {"hello": 100}) == "100"
        with pytest.raises(Exception):
            template.substitute_templates(template.Sub("$hello2"), {"hello": 100})
        assert template.substitute_templates([template.Rep("hello")], {"hello": 100}, excl_types=()) == [100]
        assert template.substitute_templates({template.Rep("hello")}, {"hello": 100}, excl_types=()) == {100}
        assert template.substitute_templates([template.Rep("hello")], {"hello": 100}, excl_types=False) == [100]
        assert template.substitute_templates({template.Rep("hello")}, {"hello": 100}, excl_types=False) == {100}
        assert template.substitute_templates([template.Rep("hello")], {"hello": 100}, incl_types=list) == [100]
        assert template.substitute_templates({template.Rep("hello")}, {"hello": 100}, incl_types=set) == {100}
        assert template.substitute_templates([template.Rep("hello")], {"hello": 100}, incl_types=True) == [100]
        assert template.substitute_templates({template.Rep("hello")}, {"hello": 100}, incl_types=True) == {100}
        assert template.substitute_templates({"test": template.Rep("hello")}, {"hello": 100}) == {"test": 100}
        Tup = namedtuple("Tup", ["a"])
        tup = Tup(template.Rep("hello"))
        assert template.substitute_templates(tup, {"hello": 100}) == Tup(100)
        assert template.substitute_templates(template.RepEval("100"), max_depth=0) == 100
        assert template.substitute_templates((template.RepEval("100"),), max_depth=0) == (template.RepEval("100"),)
        assert template.substitute_templates((template.RepEval("100"),), max_depth=1) == (100,)
        assert template.substitute_templates((template.RepEval("100"),), max_len=1) == (100,)
        assert template.substitute_templates((0, template.RepEval("100")), max_len=1) == (
            0,
            template.RepEval("100"),
        )
        assert template.substitute_templates((0, template.RepEval("100")), max_len=2) == (
            0,
            100,
        )
