import os

import pytest

import vectorbtpro as vbt
from tests.utils import *
from vectorbtpro._dtypes import *
from vectorbtpro.utils import hashing


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True


def teardown_module():
    vbt.settings.reset()


class TestHashing:
    def test_hash_args(self):
        def f(a, *args, b=2, **kwargs):
            pass

        with pytest.raises(Exception):
            hashing.hash_args(f, (0, 1), dict(c=np.array([1, 2, 3])))
        hashing.hash_args(f, (0, 1), dict(c=np.array([1, 2, 3])), ignore_args=["c"])
