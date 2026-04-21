import inspect
import os

import pytest

import vectorbtpro as vbt
from tests.utils import *
from vectorbtpro._dtypes import *
from vectorbtpro.utils import parsing


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True


def teardown_module():
    vbt.settings.reset()


class TestParsing:
    def test_get_func_kwargs(self):
        def f(a, *args, b=2, **kwargs):
            pass

        assert parsing.get_func_kwargs(f) == {"b": 2}

    def test_get_func_arg_names(self):
        def f(a, *args, b=2, **kwargs):
            pass

        assert parsing.get_func_arg_names(f) == ["a", "b"]

    def test_get_expr_var_names(self):
        assert parsing.get_expr_var_names("d = (a + b) / c") == ["d", "c", "a", "b"]

    def test_annotate_args(self):
        def f(a, *args, b=2, **kwargs):
            pass

        with pytest.raises(Exception):
            parsing.annotate_args(f, (), {})
        assert parsing.annotate_args(f, (1,), {}) == dict(
            a={"kind": inspect.Parameter.POSITIONAL_OR_KEYWORD, "value": 1},
            args={"kind": inspect.Parameter.VAR_POSITIONAL, "value": ()},
            b={"kind": inspect.Parameter.KEYWORD_ONLY, "value": 2},
            kwargs={"kind": inspect.Parameter.VAR_KEYWORD, "value": {}},
        )
        assert parsing.annotate_args(f, (1,), {}, only_passed=True) == dict(
            a={"kind": inspect.Parameter.POSITIONAL_OR_KEYWORD, "value": 1},
        )
        assert parsing.annotate_args(f, (1, 2, 3), {}) == dict(
            a={"kind": inspect.Parameter.POSITIONAL_OR_KEYWORD, "value": 1},
            args={"kind": inspect.Parameter.VAR_POSITIONAL, "value": (2, 3)},
            b={"kind": inspect.Parameter.KEYWORD_ONLY, "value": 2},
            kwargs={"kind": inspect.Parameter.VAR_KEYWORD, "value": {}},
        )

        def f2(a, b=2, **kwargs):
            pass

        assert parsing.annotate_args(f2, (1, 2), dict(c=3)) == dict(
            a={"kind": inspect.Parameter.POSITIONAL_OR_KEYWORD, "value": 1},
            b={"kind": inspect.Parameter.POSITIONAL_OR_KEYWORD, "value": 2},
            kwargs={"kind": inspect.Parameter.VAR_KEYWORD, "value": dict(c=3)},
        )

    def test_ann_args_to_args(self):
        def f(a, *args, b=2, **kwargs):
            pass

        assert parsing.ann_args_to_args(parsing.annotate_args(f, (1,), {})) == ((1,), {"b": 2})
        assert parsing.ann_args_to_args(parsing.annotate_args(f, (1,), {}, only_passed=True)) == ((1,), {})
        assert parsing.ann_args_to_args(parsing.annotate_args(f, (1, 2, 3), {})) == ((1, 2, 3), {"b": 2})

        def f2(a, b=2, **kwargs):
            pass

        assert parsing.ann_args_to_args(parsing.annotate_args(f2, (1, 2), dict(c=3))) == ((1, 2), {"c": 3})

    def test_match_ann_arg(self):
        def f(a, *args, b=2, **kwargs):
            pass

        with pytest.raises(Exception):
            parsing.annotate_args(f, (), {})

        ann_args = parsing.annotate_args(f, (0, 1), dict(c=3))

        assert parsing.match_ann_arg(ann_args, 0) == 0
        assert parsing.match_ann_arg(ann_args, "a") == 0
        assert parsing.match_ann_arg(ann_args, 1) == 1
        assert parsing.match_ann_arg(ann_args, 2) == 2
        assert parsing.match_ann_arg(ann_args, "b") == 2
        assert parsing.match_ann_arg(ann_args, parsing.Regex("(a|b)")) == 0
        assert parsing.match_ann_arg(ann_args, 3) == 3
        assert parsing.match_ann_arg(ann_args, "c") == 3
        with pytest.raises(Exception):
            parsing.match_ann_arg(ann_args, 4)
        with pytest.raises(Exception):
            parsing.match_ann_arg(ann_args, "d")

    def test_ignore_flat_ann_args(self):
        def f(a, *args, b=2, **kwargs):
            pass

        ann_args = parsing.annotate_args(f, (0, 1), dict(c=3))

        flat_ann_args = parsing.flatten_ann_args(ann_args)
        assert list(parsing.ignore_flat_ann_args(flat_ann_args, [0]).items()) == list(flat_ann_args.items())[1:]
        assert list(parsing.ignore_flat_ann_args(flat_ann_args, ["a"]).items()) == list(flat_ann_args.items())[1:]
        assert (
            list(parsing.ignore_flat_ann_args(flat_ann_args, [parsing.Regex("a")]).items())
            == list(flat_ann_args.items())[2:]
        )

    def test_get_context_vars(self):
        a = 1
        b = 2
        assert parsing.get_context_vars(["a", "b"]) == [1, 2]
        with pytest.raises(Exception):
            parsing.get_context_vars(["a", "b", "c"])
        assert parsing.get_context_vars(["c", "d", "e"], local_dict=dict(c=1, d=2, e=3)) == [1, 2, 3]
        assert parsing.get_context_vars(["c", "d", "e"], global_dict=dict(c=1, d=2, e=3)) == [1, 2, 3]
