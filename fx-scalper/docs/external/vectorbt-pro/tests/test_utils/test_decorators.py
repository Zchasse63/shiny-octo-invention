import os

import vectorbtpro as vbt
from tests.utils import *
from vectorbtpro._dtypes import *
from vectorbtpro.utils import decorators


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True


def teardown_module():
    vbt.settings.reset()


class TestDecorators:
    def test_hybrid_method(self):
        class G:
            @decorators.hybrid_method
            def g(cls_or_self):
                if isinstance(cls_or_self, type):
                    return True  # class
                return False  # instance

        assert G.g()
        assert not G().g()

    def test_hybrid_property(self):
        class G:
            @decorators.hybrid_property
            def g(cls_or_self):
                if isinstance(cls_or_self, type):
                    return True  # class
                return False  # instance

        assert G.g
        assert not G().g

    def test_custom_property(self):
        class G:
            @decorators.custom_property(some="key")
            def cache_me(self):
                return np.random.uniform()

        assert "some" in G.cache_me.options
        assert G.cache_me.options["some"] == "key"

    def test_custom_function(self):
        @decorators.custom_function(some="key")
        def cache_me():
            return np.random.uniform()

        assert "some" in cache_me.options
        assert cache_me.options["some"] == "key"
