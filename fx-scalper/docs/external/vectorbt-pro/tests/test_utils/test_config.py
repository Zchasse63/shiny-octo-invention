import os
from copy import copy, deepcopy

import pytest

import vectorbtpro as vbt
from tests.utils import *
from vectorbtpro._dtypes import *
from vectorbtpro.utils import config

tomlkit_available = True
try:
    import tomlkit
except:
    tomlkit_available = False

requires_tomlkit = pytest.mark.skipif(not tomlkit_available, reason="tomlkit not installed")


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True


def teardown_module():
    vbt.settings.reset()


class TestConfig:
    def test_copy_dict(self):
        assert config.copy_dict(None) == {}

        def _init_dict():
            return dict(const=0, lst=[1, 2, 3], dct=dict(const=1, lst=[4, 5, 6]))

        dct = _init_dict()
        _dct = config.copy_dict(dct, "shallow", nested=False)
        _dct["const"] = 2
        _dct["dct"]["const"] = 3
        _dct["lst"][0] = 0
        _dct["dct"]["lst"][0] = 0
        assert dct == dict(const=0, lst=[0, 2, 3], dct=dict(const=3, lst=[0, 5, 6]))

        dct = _init_dict()
        _dct = config.copy_dict(dct, "shallow", nested=True)
        _dct["const"] = 2
        _dct["dct"]["const"] = 3
        _dct["lst"][0] = 0
        _dct["dct"]["lst"][0] = 0
        assert dct == dict(const=0, lst=[0, 2, 3], dct=dict(const=1, lst=[0, 5, 6]))

        dct = _init_dict()
        _dct = config.copy_dict(dct, "hybrid", nested=False)
        _dct["const"] = 2
        _dct["dct"]["const"] = 3
        _dct["lst"][0] = 0
        _dct["dct"]["lst"][0] = 0
        assert dct == dict(const=0, lst=[1, 2, 3], dct=dict(const=1, lst=[0, 5, 6]))

        dct = _init_dict()
        _dct = config.copy_dict(dct, "hybrid", nested=True)
        _dct["const"] = 2
        _dct["dct"]["const"] = 3
        _dct["lst"][0] = 0
        _dct["dct"]["lst"][0] = 0
        assert dct == dict(const=0, lst=[1, 2, 3], dct=dict(const=1, lst=[4, 5, 6]))

        def init_config_(**kwargs):
            return config.Config(dict(lst=[1, 2, 3], dct=config.Config(dict(lst=[4, 5, 6]), **kwargs)), **kwargs)

        cfg = init_config_(options_=dict(readonly=True))
        _cfg = config.copy_dict(cfg, "shallow", nested=False)
        assert isinstance(_cfg, config.Config)
        assert _cfg.get_option("readonly")
        assert isinstance(_cfg["dct"], config.Config)
        assert _cfg["dct"].get_option("readonly")
        _cfg["lst"][0] = 0
        _cfg["dct"]["lst"][0] = 0
        assert cfg["lst"] == [0, 2, 3]
        assert cfg["dct"]["lst"] == [0, 5, 6]

        cfg = init_config_(options_=dict(readonly=True))
        _cfg = config.copy_dict(cfg, "shallow", nested=True)
        assert isinstance(_cfg, config.Config)
        assert _cfg.get_option("readonly")
        assert isinstance(_cfg["dct"], config.Config)
        assert _cfg["dct"].get_option("readonly")
        _cfg["lst"][0] = 0
        _cfg["dct"]["lst"][0] = 0
        assert cfg["lst"] == [0, 2, 3]
        assert cfg["dct"]["lst"] == [0, 5, 6]

        cfg = init_config_(options_=dict(readonly=True))
        _cfg = config.copy_dict(cfg, "hybrid", nested=False)
        assert isinstance(_cfg, config.Config)
        assert _cfg.get_option("readonly")
        assert isinstance(_cfg["dct"], config.Config)
        assert _cfg["dct"].get_option("readonly")
        _cfg["lst"][0] = 0
        _cfg["dct"]["lst"][0] = 0
        assert cfg["lst"] == [1, 2, 3]
        assert cfg["dct"]["lst"] == [0, 5, 6]

        cfg = init_config_(options_=dict(readonly=True))
        _cfg = config.copy_dict(cfg, "hybrid", nested=True)
        assert isinstance(_cfg, config.Config)
        assert _cfg.get_option("readonly")
        assert isinstance(_cfg["dct"], config.Config)
        assert _cfg["dct"].get_option("readonly")
        _cfg["lst"][0] = 0
        _cfg["dct"]["lst"][0] = 0
        assert cfg["lst"] == [1, 2, 3]
        assert cfg["dct"]["lst"] == [4, 5, 6]

        cfg = init_config_(options_=dict(readonly=True))
        _cfg = config.copy_dict(cfg, "deep")
        assert isinstance(_cfg, config.Config)
        assert _cfg.get_option("readonly")
        assert isinstance(_cfg["dct"], config.Config)
        assert _cfg["dct"].get_option("readonly")
        _cfg["lst"][0] = 0
        _cfg["dct"]["lst"][0] = 0
        assert cfg["lst"] == [1, 2, 3]
        assert cfg["dct"]["lst"] == [4, 5, 6]

    def test_update_dict(self):
        dct = dict(a=1)
        config.update_dict(dct, None)
        assert dct == dct
        config.update_dict(None, dct)
        assert dct == dct

        def init_config_(**kwargs):
            return config.Config(dict(a=0, b=config.Config(dict(c=1), **kwargs)), **kwargs)

        cfg = init_config_()
        config.update_dict(cfg, dict(a=1), nested=False)
        assert cfg == config.Config(dict(a=1, b=config.Config(dict(c=1))))

        cfg = init_config_()
        config.update_dict(cfg, dict(b=dict(c=2)), nested=False)
        assert cfg == config.Config(dict(a=0, b=dict(c=2)))

        cfg = init_config_()
        config.update_dict(cfg, dict(b=dict(c=2)), nested=True)
        assert cfg == config.Config(dict(a=0, b=config.Config(dict(c=2))))

        cfg = init_config_(options_=dict(readonly=True))
        with pytest.raises(Exception):
            config.update_dict(cfg, dict(b=dict(c=2)), nested=True)

        cfg = init_config_(options_=dict(readonly=True))
        config.update_dict(cfg, dict(b=dict(c=2)), nested=True, force=True)
        assert cfg == config.Config(dict(a=0, b=config.Config(dict(c=2))))
        assert cfg.get_option("readonly")
        assert cfg["b"].get_option("readonly")

        cfg = init_config_(options_=dict(readonly=True))
        config.update_dict(
            cfg,
            config.Config(
                dict(b=config.Config(dict(c=2), options_=dict(readonly=False))),
                options_=dict(readonly=False),
            ),
            nested=True,
            force=True,
        )
        assert cfg == config.Config(dict(a=0, b=config.Config(dict(c=2))))
        assert cfg.get_option("readonly")
        assert cfg["b"].get_option("readonly")

    def test_merge_dicts(self):
        assert config.merge_dicts({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
        assert config.merge_dicts({"a": 1}, {"a": 2}) == {"a": 2}
        assert config.merge_dicts({"a": {"b": 2}}, {"a": {"c": 3}}) == {"a": {"b": 2, "c": 3}}
        assert config.merge_dicts({"a": {"b": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3}}

        def init_configs(**kwargs):
            lists = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
            return (
                lists,
                config.Config(dict(lst=lists[0], dct=dict(a=1, lst=lists[1])), **kwargs),
                dict(lst=lists[2], dct=config.Config(dict(b=2, lst=lists[3]), **kwargs)),
            )

        lists, cfg1, cfg2 = init_configs(options_=dict(readonly=True))
        _cfg = config.merge_dicts(cfg1, cfg2, to_dict=True, copy_mode="shallow", nested=False)
        assert _cfg == dict(lst=lists[2], dct=config.Config(dict(b=2, lst=lists[3])))
        lists[2][0] = 0
        lists[3][0] = 0
        assert _cfg["lst"] == [0, 8, 9]
        assert _cfg["dct"]["lst"] == [0, 11, 12]

        lists, cfg1, cfg2 = init_configs(options_=dict(readonly=True))
        _cfg = config.merge_dicts(cfg1, cfg2, to_dict=True, copy_mode="shallow", nested=True)
        assert _cfg == dict(lst=lists[2], dct=dict(a=1, b=2, lst=lists[3]))
        lists[2][0] = 0
        lists[3][0] = 0
        assert _cfg["lst"] == [0, 8, 9]
        assert _cfg["dct"]["lst"] == [0, 11, 12]

        lists, cfg1, cfg2 = init_configs(options_=dict(readonly=True))
        cfg2["dct"] = config.atomic_dict(cfg2["dct"])
        _cfg = config.merge_dicts(cfg1, cfg2, to_dict=True, copy_mode="shallow", nested=True)
        assert _cfg == dict(lst=lists[2], dct=config.atomic_dict(b=2, lst=lists[3]))
        lists[2][0] = 0
        lists[3][0] = 0
        assert _cfg["lst"] == [0, 8, 9]
        assert _cfg["dct"]["lst"] == [0, 11, 12]

        lists, cfg1, cfg2 = init_configs(options_=dict(readonly=True))
        _cfg = config.merge_dicts(cfg1, config.atomic_dict(cfg2), to_dict=True, copy_mode="shallow", nested=True)
        assert _cfg == config.atomic_dict(lst=lists[2], dct=dict(b=2, lst=lists[3]))
        lists[2][0] = 0
        lists[3][0] = 0
        assert _cfg["lst"] == [0, 8, 9]
        assert _cfg["dct"]["lst"] == [0, 11, 12]

        lists, cfg1, cfg2 = init_configs(options_=dict(readonly=True))
        _cfg = config.merge_dicts(cfg1, cfg2, to_dict=False, copy_mode="shallow", nested=False)
        assert _cfg == config.Config(dict(lst=lists[2], dct=config.Config(dict(b=2, lst=lists[3]))))
        assert _cfg.get_option("readonly")
        lists[2][0] = 0
        lists[3][0] = 0
        assert _cfg["lst"] == [0, 8, 9]
        assert _cfg["dct"]["lst"] == [0, 11, 12]

        lists, cfg1, cfg2 = init_configs(options_=dict(readonly=True))
        _cfg = config.merge_dicts(cfg1, cfg2, to_dict=False, copy_mode="hybrid", nested=False)
        assert _cfg == config.Config(dict(lst=lists[2], dct=config.Config(dict(b=2, lst=lists[3]))))
        assert _cfg.get_option("readonly")
        lists[2][0] = 0
        lists[3][0] = 0
        assert _cfg["lst"] == [7, 8, 9]
        assert _cfg["dct"]["lst"] == [0, 11, 12]

        lists, cfg1, cfg2 = init_configs(options_=dict(readonly=True))
        _cfg = config.merge_dicts(cfg1, cfg2, to_dict=False, copy_mode="hybrid", nested=True)
        assert _cfg == config.Config(dict(lst=lists[2], dct=dict(a=1, b=2, lst=lists[3])))
        assert _cfg.get_option("readonly")
        lists[2][0] = 0
        lists[3][0] = 0
        assert _cfg["lst"] == [7, 8, 9]
        assert _cfg["dct"]["lst"] == [10, 11, 12]

        lists, cfg1, cfg2 = init_configs(options_=dict(readonly=True))
        _cfg = config.merge_dicts(cfg1, cfg2, to_dict=False, copy_mode="deep", nested=False)
        assert _cfg == config.Config(dict(lst=lists[2], dct=config.Config(dict(b=2, lst=lists[3]))))
        assert _cfg.get_option("readonly")
        lists[2][0] = 0
        lists[3][0] = 0
        assert _cfg["lst"] == [7, 8, 9]
        assert _cfg["dct"]["lst"] == [10, 11, 12]

    def test_config_copy(self):
        def init_config(**kwargs):
            dct = dict(const=0, lst=[1, 2, 3], dct=config.Config(dict(const=1, lst=[4, 5, 6])))
            return dct, config.Config(dct, **kwargs)

        dct, cfg = init_config(options_=dict(copy_kwargs=dict(copy_mode="shallow"), nested=False))
        assert isinstance(cfg["dct"], config.Config)
        assert isinstance(cfg.get_option("reset_dct")["dct"], config.Config)
        dct["const"] = 2
        dct["dct"]["const"] = 3
        dct["lst"][0] = 0
        dct["dct"]["lst"][0] = 0
        assert cfg == config.Config(dict(const=0, lst=[0, 2, 3], dct=config.Config(dict(const=3, lst=[0, 5, 6]))))
        assert cfg.get_option("reset_dct") == dict(
            const=0, lst=[0, 2, 3], dct=config.Config(dict(const=3, lst=[0, 5, 6]))
        )

        dct, cfg = init_config(options_=dict(copy_kwargs=dict(copy_mode="shallow"), nested=True))
        assert isinstance(cfg["dct"], config.Config)
        assert isinstance(cfg.get_option("reset_dct")["dct"], config.Config)
        dct["const"] = 2
        dct["dct"]["const"] = 3
        dct["lst"][0] = 0
        dct["dct"]["lst"][0] = 0
        assert cfg == config.Config(dict(const=0, lst=[0, 2, 3], dct=config.Config(dict(const=1, lst=[0, 5, 6]))))
        assert cfg.get_option("reset_dct") == dict(
            const=0, lst=[0, 2, 3], dct=config.Config(dict(const=1, lst=[0, 5, 6]))
        )

        dct, cfg = init_config(options_=dict(copy_kwargs=dict(copy_mode="hybrid"), nested=True))
        assert isinstance(cfg["dct"], config.Config)
        assert isinstance(cfg.get_option("reset_dct")["dct"], config.Config)
        dct["const"] = 2
        dct["dct"]["const"] = 3
        dct["lst"][0] = 0
        dct["dct"]["lst"][0] = 0
        assert cfg == config.Config(dict(const=0, lst=[1, 2, 3], dct=config.Config(dict(const=1, lst=[4, 5, 6]))))
        assert cfg.get_option("reset_dct") == dict(
            const=0, lst=[1, 2, 3], dct=config.Config(dict(const=1, lst=[4, 5, 6]))
        )

        dct, cfg = init_config(
            options_=dict(
                copy_kwargs=dict(copy_mode="shallow"),
                reset_dct_copy_kwargs=dict(copy_mode="hybrid"),
                nested=True,
            )
        )
        assert isinstance(cfg["dct"], config.Config)
        assert isinstance(cfg.get_option("reset_dct")["dct"], config.Config)
        dct["const"] = 2
        dct["dct"]["const"] = 3
        dct["lst"][0] = 0
        dct["dct"]["lst"][0] = 0
        assert cfg == config.Config(dict(const=0, lst=[0, 2, 3], dct=config.Config(dict(const=1, lst=[0, 5, 6]))))
        assert cfg.get_option("reset_dct") == dict(
            const=0, lst=[1, 2, 3], dct=config.Config(dict(const=1, lst=[4, 5, 6]))
        )

        dct, cfg = init_config(options_=dict(copy_kwargs=dict(copy_mode="deep"), nested=True))
        assert isinstance(cfg["dct"], config.Config)
        assert isinstance(cfg.get_option("reset_dct")["dct"], config.Config)
        dct["const"] = 2
        dct["dct"]["const"] = 3
        dct["lst"][0] = 0
        dct["dct"]["lst"][0] = 0
        assert cfg == config.Config(dict(const=0, lst=[1, 2, 3], dct=config.Config(dict(const=1, lst=[4, 5, 6]))))
        assert cfg.get_option("reset_dct") == dict(
            const=0, lst=[1, 2, 3], dct=config.Config(dict(const=1, lst=[4, 5, 6]))
        )

        init_d, _ = init_config()
        init_d = config.copy_dict(init_d, "deep")
        dct, cfg = init_config(options_=dict(copy_kwargs=dict(copy_mode="hybrid"), reset_dct=init_d, nested=True))
        assert isinstance(cfg["dct"], config.Config)
        assert isinstance(cfg.get_option("reset_dct")["dct"], config.Config)
        dct["const"] = 2
        dct["dct"]["const"] = 3
        assert cfg == config.Config(dict(const=0, lst=[1, 2, 3], dct=config.Config(dict(const=1, lst=[4, 5, 6]))))
        init_d["lst"][0] = 0
        init_d["dct"]["lst"][0] = 0
        assert cfg == config.Config(dict(const=0, lst=[1, 2, 3], dct=config.Config(dict(const=1, lst=[4, 5, 6]))))
        assert cfg.get_option("reset_dct") == dict(
            const=0, lst=[1, 2, 3], dct=config.Config(dict(const=1, lst=[4, 5, 6]))
        )

        init_d, _ = init_config()
        init_d = config.copy_dict(init_d, "deep")
        dct, cfg = init_config(
            options_=dict(
                copy_kwargs=dict(copy_mode="hybrid"),
                reset_dct=init_d,
                reset_dct_copy_kwargs=dict(copy_mode="shallow"),
                nested=True,
            )
        )
        assert isinstance(cfg["dct"], config.Config)
        assert isinstance(cfg.get_option("reset_dct")["dct"], config.Config)
        dct["const"] = 2
        dct["dct"]["const"] = 3
        dct["lst"][0] = 0
        dct["dct"]["lst"][0] = 0
        init_d["const"] = 2
        init_d["dct"]["const"] = 3
        init_d["lst"][0] = 0
        init_d["dct"]["lst"][0] = 0
        assert cfg == config.Config(dict(const=0, lst=[1, 2, 3], dct=config.Config(dict(const=1, lst=[4, 5, 6]))))
        assert cfg.get_option("reset_dct") == dict(
            const=0, lst=[0, 2, 3], dct=config.Config(dict(const=1, lst=[0, 5, 6]))
        )

        _, cfg = init_config(options_=dict(nested=True))
        _cfg = copy(cfg)
        _cfg["const"] = 2
        _cfg["dct"]["const"] = 3
        _cfg["lst"][0] = 0
        _cfg["dct"]["lst"][0] = 0
        _cfg.get_option("reset_dct")["const"] = 2
        _cfg.get_option("reset_dct")["dct"]["const"] = 3
        _cfg.get_option("reset_dct")["lst"][0] = 0
        _cfg.get_option("reset_dct")["dct"]["lst"][0] = 0
        assert cfg == config.Config(dict(const=0, lst=[0, 2, 3], dct=config.Config(dict(const=3, lst=[0, 5, 6]))))
        assert cfg.get_option("reset_dct") == dict(
            const=2, lst=[0, 2, 3], dct=config.Config(dict(const=3, lst=[0, 5, 6]))
        )

        _, cfg = init_config(options_=dict(nested=True))
        _cfg = deepcopy(cfg)
        _cfg["const"] = 2
        _cfg["dct"]["const"] = 3
        _cfg["lst"][0] = 0
        _cfg["dct"]["lst"][0] = 0
        _cfg.get_option("reset_dct")["const"] = 2
        _cfg.get_option("reset_dct")["dct"]["const"] = 3
        _cfg.get_option("reset_dct")["lst"][0] = 0
        _cfg.get_option("reset_dct")["dct"]["lst"][0] = 0
        assert cfg == config.Config(dict(const=0, lst=[1, 2, 3], dct=config.Config(dict(const=1, lst=[4, 5, 6]))))
        assert cfg.get_option("reset_dct") == dict(
            const=0, lst=[1, 2, 3], dct=config.Config(dict(const=1, lst=[4, 5, 6]))
        )

        _, cfg = init_config(options_=dict(copy_kwargs=dict(copy_mode="hybrid"), nested=True))
        _cfg = cfg.copy()
        _cfg["const"] = 2
        _cfg["dct"]["const"] = 3
        _cfg["lst"][0] = 0
        _cfg["dct"]["lst"][0] = 0
        _cfg.get_option("reset_dct")["const"] = 2
        _cfg.get_option("reset_dct")["dct"]["const"] = 3
        _cfg.get_option("reset_dct")["lst"][0] = 0
        _cfg.get_option("reset_dct")["dct"]["lst"][0] = 0
        assert cfg == config.Config(dict(const=0, lst=[1, 2, 3], dct=config.Config(dict(const=1, lst=[4, 5, 6]))))
        assert cfg.get_option("reset_dct") == dict(
            const=0, lst=[1, 2, 3], dct=config.Config(dict(const=1, lst=[4, 5, 6]))
        )

        _, cfg = init_config(options_=dict(copy_kwargs=dict(copy_mode="hybrid"), nested=True))
        _cfg = cfg.copy(reset_dct_copy_kwargs=dict(copy_mode="shallow"))
        _cfg["const"] = 2
        _cfg["dct"]["const"] = 3
        _cfg["lst"][0] = 0
        _cfg["dct"]["lst"][0] = 0
        _cfg.get_option("reset_dct")["const"] = 2
        _cfg.get_option("reset_dct")["dct"]["const"] = 3
        _cfg.get_option("reset_dct")["lst"][0] = 0
        _cfg.get_option("reset_dct")["dct"]["lst"][0] = 0
        assert cfg == config.Config(dict(const=0, lst=[1, 2, 3], dct=config.Config(dict(const=1, lst=[4, 5, 6]))))
        assert cfg.get_option("reset_dct") == dict(
            const=0, lst=[0, 2, 3], dct=config.Config(dict(const=1, lst=[0, 5, 6]))
        )

        _, cfg = init_config(options_=dict(nested=True))
        _cfg = cfg.copy(copy_mode="deep")
        _cfg["const"] = 2
        _cfg["dct"]["const"] = 3
        _cfg["lst"][0] = 0
        _cfg["dct"]["lst"][0] = 0
        _cfg.get_option("reset_dct")["const"] = 2
        _cfg.get_option("reset_dct")["dct"]["const"] = 3
        _cfg.get_option("reset_dct")["lst"][0] = 0
        _cfg.get_option("reset_dct")["dct"]["lst"][0] = 0
        assert cfg == config.Config(dict(const=0, lst=[1, 2, 3], dct=config.Config(dict(const=1, lst=[4, 5, 6]))))
        assert cfg.get_option("reset_dct") == dict(
            const=0, lst=[1, 2, 3], dct=config.Config(dict(const=1, lst=[4, 5, 6]))
        )

    def test_config_convert_children(self):
        cfg = config.Config(
            dict(dct=config.child_dict(dct=config.Config(dict(), options_=dict(nested=False)))),
            options_=dict(nested=True, convert_children=True),
        )
        assert cfg.get_option("nested")
        assert cfg.get_option("convert_children")
        assert isinstance(cfg["dct"], config.Config)
        assert cfg["dct"].get_option("nested")
        assert cfg["dct"].get_option("convert_children")
        assert isinstance(cfg["dct"]["dct"], config.Config)
        assert not cfg["dct"]["dct"].get_option("nested")
        assert not cfg["dct"]["dct"].get_option("convert_children")

    def test_config_from_config(self):
        cfg = config.Config(
            config.Config(
                dict(a=0),
                options_=dict(
                    copy_kwargs=dict(copy_mode="deep", nested=True),
                    reset_dct=dict(b=0),
                    reset_dct_copy_kwargs=dict(copy_mode="deep", nested=True),
                    frozen_keys=True,
                    readonly=True,
                    nested=True,
                    convert_children=True,
                    as_attrs=True,
                ),
            )
        )
        assert dict(cfg) == dict(a=0)
        assert cfg.get_option("copy_kwargs") == dict(copy_mode="deep", nested=True)
        assert cfg.get_option("reset_dct") == dict(b=0)
        assert cfg.get_option("reset_dct_copy_kwargs") == dict(copy_mode="deep", nested=True)
        assert cfg.get_option("frozen_keys")
        assert cfg.get_option("readonly")
        assert cfg.get_option("nested")
        assert cfg.get_option("convert_children")
        assert cfg.get_option("as_attrs")

        c2 = config.Config(
            cfg,
            options_=dict(
                copy_kwargs=dict(copy_mode="hybrid"),
                reset_dct=dict(b=0),
                reset_dct_copy_kwargs=dict(nested=False),
                frozen_keys=False,
                readonly=False,
                nested=False,
                convert_children=False,
                as_attrs=False,
            ),
        )
        assert dict(c2) == dict(a=0)
        assert c2.get_option("copy_kwargs") == dict(copy_mode="hybrid", nested=True)
        assert c2.get_option("reset_dct") == dict(b=0)
        assert c2.get_option("reset_dct_copy_kwargs") == dict(copy_mode="hybrid", nested=False)
        assert not c2.get_option("frozen_keys")
        assert not c2.get_option("readonly")
        assert not c2.get_option("nested")
        assert not c2.get_option("convert_children")
        assert not c2.get_option("as_attrs")

    def test_config_defaults(self):
        cfg = config.Config(dict(a=0))
        assert dict(cfg) == dict(a=0)
        assert cfg.get_option("copy_kwargs") == dict(copy_mode="none", nested=True)
        assert cfg.get_option("reset_dct") == dict(a=0)
        assert cfg.get_option("reset_dct_copy_kwargs") == dict(copy_mode="hybrid", nested=True)
        assert not cfg.get_option("frozen_keys")
        assert not cfg.get_option("readonly")
        assert cfg.get_option("nested")
        assert not cfg.get_option("convert_children")
        assert not cfg.get_option("as_attrs")

        vbt.settings.config.options.reset()
        vbt.settings.config.options["copy_kwargs"] = dict(copy_mode="deep")
        vbt.settings.config.options["reset_dct_copy_kwargs"] = dict(copy_mode="deep")
        vbt.settings.config.options["frozen_keys"] = True
        vbt.settings.config.options["readonly"] = True
        vbt.settings.config.options["nested"] = False
        vbt.settings.config.options["convert_children"] = True
        vbt.settings.config.options["as_attrs"] = True

        cfg = config.Config(dict(a=0))
        assert dict(cfg) == dict(a=0)
        assert cfg.get_option("copy_kwargs") == dict(copy_mode="deep", nested=False)
        assert cfg.get_option("reset_dct") == dict(a=0)
        assert cfg.get_option("reset_dct_copy_kwargs") == dict(copy_mode="deep", nested=False)
        assert cfg.get_option("frozen_keys")
        assert cfg.get_option("readonly")
        assert not cfg.get_option("nested")
        assert cfg.get_option("convert_children")
        assert cfg.get_option("as_attrs")

        vbt.settings.config.reset()

    def test_config_as_attrs(self):
        cfg = config.Config(dict(a=0, b=0, dct=dict(d=0)), options_=dict(as_attrs=True))
        assert cfg.a == 0
        assert cfg.b == 0
        with pytest.raises(Exception):
            assert cfg.dct.d == 0

        cfg.e = 0
        assert cfg["e"] == 0
        cfg["f"] = 0
        assert cfg.f == 0
        with pytest.raises(Exception):
            assert cfg.g == 0
        del cfg["f"]
        with pytest.raises(Exception):
            assert cfg.f == 0
        del cfg.e
        with pytest.raises(Exception):
            assert cfg["e"] == 0
        cfg.clear()
        assert dict(cfg) == dict()
        assert not hasattr(cfg, "a")
        assert not hasattr(cfg, "b")
        cfg.a = 0
        cfg.b = 0
        cfg.pop("a")
        assert not hasattr(cfg, "a")
        cfg.popitem()
        assert not hasattr(cfg, "b")

        cfg = config.Config(
            config.child_dict(a=0, b=0, dct=config.child_dict(d=0)),
            options_=dict(as_attrs=True, nested=True, convert_children=True),
        )
        assert cfg.a == 0
        assert cfg.b == 0
        assert cfg.dct.d == 0

        with pytest.raises(Exception):
            config.Config(dict(options_=True), options_=dict(as_attrs=True))

    def test_config_frozen_keys(self):
        cfg = config.Config(dict(a=0), options_=dict(frozen_keys=False))
        cfg.pop("a")
        assert dict(cfg) == dict()

        cfg = config.Config(dict(a=0), options_=dict(frozen_keys=False))
        cfg.popitem()
        assert dict(cfg) == dict()

        cfg = config.Config(dict(a=0), options_=dict(frozen_keys=False))
        cfg.clear()
        assert dict(cfg) == dict()

        cfg = config.Config(dict(a=0), options_=dict(frozen_keys=False))
        cfg.update(dict(a=1))
        assert dict(cfg) == dict(a=1)

        cfg = config.Config(dict(a=0), options_=dict(frozen_keys=False))
        cfg.update(dict(b=0))
        assert dict(cfg) == dict(a=0, b=0)

        cfg = config.Config(dict(a=0), options_=dict(frozen_keys=False))
        del cfg["a"]
        assert dict(cfg) == dict()

        cfg = config.Config(dict(a=0), options_=dict(frozen_keys=False))
        cfg["a"] = 1
        assert dict(cfg) == dict(a=1)

        cfg = config.Config(dict(a=0), options_=dict(frozen_keys=False))
        cfg["b"] = 0
        assert dict(cfg) == dict(a=0, b=0)

        cfg = config.Config(dict(a=0), options_=dict(frozen_keys=True))
        with pytest.raises(Exception):
            cfg.pop("a")
        cfg.pop("a", force=True)
        assert dict(cfg) == dict()

        cfg = config.Config(dict(a=0), options_=dict(frozen_keys=True))
        with pytest.raises(Exception):
            cfg.popitem()
        cfg.popitem(force=True)
        assert dict(cfg) == dict()

        cfg = config.Config(dict(a=0), options_=dict(frozen_keys=True))
        with pytest.raises(Exception):
            cfg.clear()
        cfg.clear(force=True)
        assert dict(cfg) == dict()

        cfg = config.Config(dict(a=0), options_=dict(frozen_keys=True))
        cfg.update(dict(a=1))
        assert dict(cfg) == dict(a=1)

        cfg = config.Config(dict(a=0), options_=dict(frozen_keys=True))
        with pytest.raises(Exception):
            cfg.update(dict(b=0))
        cfg.update(dict(b=0), force=True)
        assert dict(cfg) == dict(a=0, b=0)

        cfg = config.Config(dict(a=0), options_=dict(frozen_keys=True))
        with pytest.raises(Exception):
            del cfg["a"]
        cfg.__delitem__("a", force=True)
        assert dict(cfg) == dict()

        cfg = config.Config(dict(a=0), options_=dict(frozen_keys=True))
        cfg["a"] = 1
        assert dict(cfg) == dict(a=1)

        cfg = config.Config(dict(a=0), options_=dict(frozen_keys=True))
        with pytest.raises(Exception):
            cfg["b"] = 0
        cfg.__setitem__("b", 0, force=True)
        assert dict(cfg) == dict(a=0, b=0)

    def test_config_readonly(self):
        cfg = config.Config(dict(a=0), options_=dict(readonly=False))
        cfg.pop("a")
        assert dict(cfg) == dict()

        cfg = config.Config(dict(a=0), options_=dict(readonly=False))
        cfg.popitem()
        assert dict(cfg) == dict()

        cfg = config.Config(dict(a=0), options_=dict(readonly=False))
        cfg.clear()
        assert dict(cfg) == dict()

        cfg = config.Config(dict(a=0), options_=dict(readonly=False))
        cfg.update(dict(a=1))
        assert dict(cfg) == dict(a=1)

        cfg = config.Config(dict(a=0), options_=dict(readonly=False))
        cfg.update(dict(b=0))
        assert dict(cfg) == dict(a=0, b=0)

        cfg = config.Config(dict(a=0), options_=dict(readonly=False))
        del cfg["a"]
        assert dict(cfg) == dict()

        cfg = config.Config(dict(a=0), options_=dict(readonly=False))
        cfg["a"] = 1
        assert dict(cfg) == dict(a=1)

        cfg = config.Config(dict(a=0), options_=dict(readonly=False))
        cfg["b"] = 0
        assert dict(cfg) == dict(a=0, b=0)

        cfg = config.Config(dict(a=0), options_=dict(readonly=True))
        with pytest.raises(Exception):
            cfg.pop("a")
        cfg.pop("a", force=True)
        assert dict(cfg) == dict()

        cfg = config.Config(dict(a=0), options_=dict(readonly=True))
        with pytest.raises(Exception):
            cfg.popitem()
        cfg.popitem(force=True)
        assert dict(cfg) == dict()

        cfg = config.Config(dict(a=0), options_=dict(readonly=True))
        with pytest.raises(Exception):
            cfg.clear()
        cfg.clear(force=True)
        assert dict(cfg) == dict()

        cfg = config.Config(dict(a=0), options_=dict(readonly=True))
        with pytest.raises(Exception):
            cfg.update(dict(a=1))
        cfg.update(dict(a=1), force=True)
        assert dict(cfg) == dict(a=1)

        cfg = config.Config(dict(a=0), options_=dict(readonly=True))
        with pytest.raises(Exception):
            cfg.update(dict(b=0))
        cfg.update(dict(b=0), force=True)
        assert dict(cfg) == dict(a=0, b=0)

        cfg = config.Config(dict(a=0), options_=dict(readonly=True))
        with pytest.raises(Exception):
            del cfg["a"]
        cfg.__delitem__("a", force=True)
        assert dict(cfg) == dict()

        cfg = config.Config(dict(a=0), options_=dict(readonly=True))
        with pytest.raises(Exception):
            cfg["a"] = 1
        cfg.__setitem__("a", 1, force=True)
        assert dict(cfg) == dict(a=1)

        cfg = config.Config(dict(a=0), options_=dict(readonly=True))
        with pytest.raises(Exception):
            cfg["b"] = 0
        cfg.__setitem__("b", 0, force=True)
        assert dict(cfg) == dict(a=0, b=0)

    def test_config_merge_with(self):
        cfg1 = config.Config(
            dict(a=0, dct=dict(b=1, dct=config.Config(dict(c=2), options_=dict(readonly=False)))),
            options_=dict(readonly=False, nested=False),
        )
        cfg2 = config.Config(
            dict(d=3, dct=config.Config(dict(e=4, dct=dict(f=5)), options_=dict(readonly=True))),
            options_=dict(readonly=True, nested=False),
        )
        _cfg = cfg1.merge_with(cfg2)
        assert _cfg == dict(a=0, d=3, dct=cfg2["dct"])
        assert not isinstance(_cfg, config.Config)
        assert isinstance(_cfg["dct"], config.Config)
        assert not isinstance(_cfg["dct"]["dct"], config.Config)

        _cfg = cfg1.merge_with(cfg2, to_dict=False, nested=False)
        assert _cfg == config.Config(dict(a=0, d=3, dct=cfg2["dct"]))
        assert not _cfg.get_option("readonly")
        assert isinstance(_cfg["dct"], config.Config)
        assert _cfg["dct"].get_option("readonly")
        assert not isinstance(_cfg["dct"]["dct"], config.Config)

        _cfg = cfg1.merge_with(cfg2, to_dict=False, nested=True)
        assert _cfg == config.Config(dict(a=0, d=3, dct=dict(b=1, e=4, dct=config.Config(dict(c=2, f=5)))))
        assert not _cfg.get_option("readonly")
        assert not isinstance(_cfg["dct"], config.Config)
        assert isinstance(_cfg["dct"]["dct"], config.Config)
        assert not _cfg["dct"]["dct"].get_option("readonly")

    def test_config_reset(self):
        cfg = config.Config(
            dict(a=0, dct=dict(b=0)),
            options_=dict(copy_kwargs=dict(copy_mode="shallow"), nested=False),
        )
        cfg["a"] = 1
        cfg["dct"]["b"] = 1
        cfg.reset()
        assert cfg == config.Config(dict(a=0, dct=dict(b=1)))

        cfg = config.Config(
            dict(a=0, dct=dict(b=0)),
            options_=dict(copy_kwargs=dict(copy_mode="hybrid"), nested=False),
        )
        cfg["a"] = 1
        cfg["dct"]["b"] = 1
        cfg.reset()
        assert cfg == config.Config(dict(a=0, dct=dict(b=0)))

        cfg = config.Config(
            dict(a=0, dct=dict(b=0)),
            options_=dict(copy_kwargs=dict(copy_mode="deep"), nested=False),
        )
        cfg["a"] = 1
        cfg["dct"]["b"] = 1
        cfg.reset()
        assert cfg == config.Config(dict(a=0, dct=dict(b=0)))

        cfg = config.Config(
            dict(a=0, dct=dict(b=0)),
            options_=dict(copy_kwargs=dict(copy_mode="shallow"), nested=True),
        )
        cfg["a"] = 1
        cfg["dct"]["b"] = 1
        cfg.reset()
        assert cfg == config.Config(dict(a=0, dct=dict(b=0)))

        cfg = config.Config(
            dict(a=0, dct=dict(b=0)),
            options_=dict(copy_kwargs=dict(copy_mode="hybrid"), nested=True),
        )
        cfg["a"] = 1
        cfg["dct"]["b"] = 1
        cfg.reset()
        assert cfg == config.Config(dict(a=0, dct=dict(b=0)))

        cfg = config.Config(
            dict(a=0, dct=dict(b=0)),
            options_=dict(copy_kwargs=dict(copy_mode="deep"), nested=True),
        )
        cfg["a"] = 1
        cfg["dct"]["b"] = 1
        cfg.reset()
        assert cfg == config.Config(dict(a=0, dct=dict(b=0)))

    @pytest.mark.parametrize("test_file_format", ["ini", "yml", pytest.param("toml", marks=requires_tomlkit)])
    def test_config_save_and_load(self, tmp_path, test_file_format):
        cfg = config.Config(
            dict(a=0, dct=dict(b=[1, 2, 3], dct=config.Config(options_=dict(readonly=False)))),
            options_=dict(
                copy_kwargs=dict(copy_mode="deep", nested=True),
                reset_dct=dict(b=0),
                reset_dct_copy_kwargs=dict(copy_mode="deep", nested=True),
                pickle_reset_dct=True,
                frozen_keys=True,
                readonly=True,
                nested=True,
                convert_children=True,
                as_attrs=True,
            ),
        )
        cfg.save(tmp_path / "config")
        new_cfg = config.Config.load(tmp_path / "config")
        assert new_cfg == deepcopy(cfg)
        assert new_cfg.__dict__ == deepcopy(cfg).__dict__
        cfg.save(tmp_path / "config", file_format=test_file_format)
        new_cfg = config.Config.load(tmp_path / "config", file_format=test_file_format)
        assert new_cfg == deepcopy(cfg)
        assert new_cfg.__dict__ == deepcopy(cfg).__dict__

    @pytest.mark.parametrize("test_file_format", ["ini", "yml", pytest.param("toml", marks=requires_tomlkit)])
    def test_config_load_update(self, tmp_path, test_file_format):
        cfg1 = config.Config(
            dict(a=0, dct=dict(b=[1, 2, 3], dct=config.Config(options_=dict(readonly=False)))),
            options_=dict(
                copy_kwargs=dict(copy_mode="deep", nested=True),
                reset_dct=dict(b=0),
                reset_dct_copy_kwargs=dict(copy_mode="deep", nested=True),
                pickle_reset_dct=True,
                frozen_keys=True,
                readonly=True,
                nested=True,
                convert_children=True,
                as_attrs=True,
            ),
        )
        cfg2 = cfg3 = cfg4 = cfg5 = config.Config(
            dct=dict(a=1, dct=dict(b=[4, 5, 6], dct=config.Config(options_=dict(readonly=True)))),
            options_=dict(
                copy_kwargs=dict(copy_mode="shallow", nested=False),
                reset_dct=dict(b=1),
                reset_dct_copy_kwargs=dict(copy_mode="shallow", nested=False),
                pickle_reset_dct=False,
                frozen_keys=False,
                readonly=False,
                nested=False,
                convert_children=False,
                as_attrs=False,
            ),
        )
        cfg2 = deepcopy(cfg2)
        cfg3 = deepcopy(cfg3)
        cfg4 = deepcopy(cfg4)
        cfg5 = deepcopy(cfg5)
        cfg1.save(tmp_path / "config")
        cfg2.load_update(tmp_path / "config")
        assert cfg2 == deepcopy(cfg1)
        assert cfg2.__dict__ != cfg1.__dict__
        cfg3.load_update(tmp_path / "config", update_options=True)
        assert cfg3 == deepcopy(cfg1)
        assert cfg3.__dict__ == cfg1.__dict__
        cfg1.save(tmp_path / "config", file_format=test_file_format)
        cfg4.load_update(tmp_path / "config", file_format=test_file_format)
        assert cfg4 == deepcopy(cfg1)
        assert cfg4.__dict__ != cfg1.__dict__
        cfg5.load_update(tmp_path / "config", file_format=test_file_format, update_options=True)
        assert cfg5 == deepcopy(cfg1)
        assert cfg5.__dict__ == cfg1.__dict__

    @pytest.mark.parametrize("test_file_format", ["ini", "yml", pytest.param("toml", marks=requires_tomlkit)])
    def test_configured(self, tmp_path, test_file_format):
        class H(config.Configured):
            _rec_id = "123456789"
            _writeable_attrs = {"my_attr", "my_cfg"}

            def __init__(self, a, b=2, **kwargs):
                super().__init__(a=a, b=b, **kwargs)
                self.my_attr = 100
                self.my_cfg = config.Config(dict(sr=pd.Series([1, 2, 3])))

        assert H(1).config == config.Config({"a": 1, "b": 2})
        assert H(1).replace(b=3).config == config.Config({"a": 1, "b": 3})
        assert H(pd.Series([1, 2, 3])) == H(pd.Series([1, 2, 3]))
        assert H(pd.Series([1, 2, 3])) != H(pd.Series([1, 2, 4]))
        assert H(pd.DataFrame([1, 2, 3])) == H(pd.DataFrame([1, 2, 3]))
        assert H(pd.DataFrame([1, 2, 3])) != H(pd.DataFrame([1, 2, 4]))
        assert H(pd.Index([1, 2, 3])) == H(pd.Index([1, 2, 3]))
        assert H(pd.Index([1, 2, 3])) != H(pd.Index([1, 2, 4]))
        assert H(np.array([1, 2, 3])) == H(np.array([1, 2, 3]))
        assert H(np.array([1, 2, 3])) != H(np.array([1, 2, 4]))
        assert H(None) == H(None)
        assert H(None) != H(10.0)

        vbt.RecInfo(H._rec_id, H).register()

        h = H(1)
        h.my_attr = 200
        h.my_cfg["df"] = pd.DataFrame([1, 2, 3])
        h2 = H(1)
        h2.my_attr = 200
        h2.my_cfg["df"] = pd.DataFrame([1, 2, 3])
        h.save(tmp_path / "configured")
        new_h = H.load(tmp_path / "configured")
        assert new_h == h2
        assert new_h != H(1)
        assert new_h.__dict__ == h2.__dict__
        assert new_h.__dict__ != H(1).__dict__
        assert new_h.my_attr == h.my_attr
        assert new_h.my_cfg == h.my_cfg
        h.save(tmp_path / "configured", file_format=test_file_format)
        new_h = H.load(tmp_path / "configured", file_format=test_file_format)
        assert new_h == h2
        assert new_h != H(1)
        assert new_h.__dict__ == h2.__dict__
        assert new_h.__dict__ != H(1).__dict__
        assert new_h.my_attr == h.my_attr
        assert new_h.my_cfg == h.my_cfg
