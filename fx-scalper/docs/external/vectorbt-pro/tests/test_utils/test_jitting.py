import os
from functools import wraps

import pytest
from numba.core.registry import CPUDispatcher

import vectorbtpro as vbt
from tests.utils import *
from vectorbtpro._dtypes import *
from vectorbtpro.utils import checks, jitting


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True


def teardown_module():
    vbt.settings.reset()


class TestJitting:
    def test_jitters(self):
        py_func = lambda x: x

        assert jitting.NumPyJitter().decorate(py_func) is py_func
        if checks.is_numba_enabled():
            assert isinstance(jitting.NumbaJitter().decorate(py_func), CPUDispatcher)
            assert not jitting.NumbaJitter(parallel=True).decorate(py_func).targetoptions["parallel"]
            assert jitting.NumbaJitter(parallel=True).decorate(py_func, tags={"can_parallel"}).targetoptions["parallel"]
            assert (
                jitting.NumbaJitter(parallel=True, fix_cannot_parallel=False)
                .decorate(py_func)
                .targetoptions["parallel"]
            )

    def test_get_func_suffix(self):
        def py_func():
            pass

        def func_nb():
            pass

        assert jitting.get_func_suffix(lambda x: x) is None
        assert jitting.get_func_suffix(py_func) is None
        assert jitting.get_func_suffix(func_nb) == "nb"

    def test_resolve_jitter_type(self):
        def py_func():
            pass

        def func_nb():
            pass

        with pytest.raises(Exception):
            jitting.resolve_jitter_type()
        with pytest.raises(Exception):
            jitting.resolve_jitter_type(py_func=py_func)
        assert jitting.resolve_jitter_type(py_func=func_nb) is jitting.NumbaJitter
        assert jitting.resolve_jitter_type(jitter="numba", py_func=func_nb) is jitting.NumbaJitter
        with pytest.raises(Exception):
            jitting.resolve_jitter_type(jitter="numba2", py_func=func_nb)
        assert jitting.resolve_jitter_type(jitter=jitting.NumbaJitter, py_func=func_nb) is jitting.NumbaJitter
        assert jitting.resolve_jitter_type(jitter=jitting.NumbaJitter(), py_func=func_nb) is jitting.NumbaJitter
        with pytest.raises(Exception):
            jitting.resolve_jitter_type(jitter=object, py_func=func_nb)

    def test_get_id_of_jitter_type(self):
        assert jitting.get_id_of_jitter_type(jitting.NumbaJitter) == "nb"
        assert jitting.get_id_of_jitter_type(jitting.NumPyJitter) == "np"
        assert jitting.get_id_of_jitter_type(object) is None

    def test_resolve_jitted_kwargs(self):
        assert jitting.resolve_jitted_kwargs(option=True) == dict()
        assert jitting.resolve_jitted_kwargs(option=False) is None
        assert jitting.resolve_jitted_kwargs(option=dict(test="test")) == dict(test="test")
        assert jitting.resolve_jitted_kwargs(option="numba") == dict(jitter="numba")
        with pytest.raises(Exception):
            jitting.resolve_jitted_kwargs(option=10)
        assert jitting.resolve_jitted_kwargs(option="numba", jitter="numpy") == dict(jitter="numba")

    def test_jitted(self):
        class MyJitter(jitting.Jitter):
            def decorate(self, py_func, tags=None):
                @wraps(py_func)
                def wrapper(*args, **kwargs):
                    return py_func(*args, **kwargs)

                wrapper.config = self.config
                return wrapper

        vbt.settings.jitting.jitters["my"] = dict(cls=MyJitter)

        @jitting.jitted
        def func_my():
            pass

        assert dict(func_my.config) == dict()

        @jitting.jitted(test="test")
        def func_my():
            pass

        assert dict(func_my.config) == dict(test="test")
