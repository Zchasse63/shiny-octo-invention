import os

import vectorbtpro as vbt
from tests.utils import *
from vectorbtpro._dtypes import *
from vectorbtpro.utils import tagging


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True


def teardown_module():
    vbt.settings.reset()


class TestTags:
    def test_match_tags(self):
        assert tagging.match_tags("hello", "hello")
        assert not tagging.match_tags("hello", "world")
        assert tagging.match_tags(["hello", "world"], "world")
        assert tagging.match_tags("hello", ["hello", "world"])
        assert tagging.match_tags("hello and world", ["hello", "world"])
        assert not tagging.match_tags("hello and not world", ["hello", "world"])
