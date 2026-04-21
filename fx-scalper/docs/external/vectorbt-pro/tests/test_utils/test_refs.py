import os

import pytest

import vectorbtpro as vbt
from tests.utils import *
from vectorbtpro._dtypes import *
from vectorbtpro.utils import refs

networkx_available = True
try:
    import networkx
except:
    networkx_available = False

requires_networkx = pytest.mark.skipif(not networkx_available, reason="networkx not installed")


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True


def teardown_module():
    vbt.settings.reset()


class TestRefs:
    def test_get_api_ref(self):
        pf = vbt.PF.from_holding([1, 2, 3])

        assert vbt.get_refname(vbt) == "vectorbtpro"
        assert vbt.get_refname(vbt.utils) == "vectorbtpro.utils"
        assert vbt.get_refname(vbt.utils.datetime_) == "vectorbtpro.utils.datetime_"
        assert vbt.get_refname(vbt.dt) == "vectorbtpro.utils.datetime_"
        assert vbt.get_refname(vbt.nb) == "vectorbtpro.generic.nb"
        assert vbt.get_refname(vbt.nb.rolling_mean_nb) == "vectorbtpro.generic.nb.rolling.rolling_mean_nb"
        assert vbt.get_refname(vbt.settings) == "vectorbtpro._settings.settings"
        assert vbt.get_refname(vbt.Portfolio) == "vectorbtpro.portfolio.base.Portfolio"
        assert vbt.get_refname(vbt.PF) == "vectorbtpro.portfolio.base.Portfolio"
        assert vbt.get_refname(vbt.Data.fetch_symbol) == "vectorbtpro.data.base.Data.fetch_symbol"
        assert vbt.get_refname(vbt.YFData.fetch_symbol) == "vectorbtpro.data.custom.yf.YFData.fetch_symbol"
        assert vbt.get_refname(vbt.Data.run) == "vectorbtpro.data.base.Data.run"
        assert vbt.get_refname(vbt.YFData.run) == "vectorbtpro.data.base.Data.run"
        assert vbt.get_refname(vbt.PF.cash_sharing) == "vectorbtpro.portfolio.base.Portfolio.cash_sharing"
        assert vbt.get_refname(vbt.PF.close) == "vectorbtpro.portfolio.base.Portfolio.close"
        assert vbt.get_refname(vbt.PF.plot_value) == "vectorbtpro.portfolio.base.Portfolio.plot_value"
        assert vbt.get_refname(vbt.PF.get_filled_close) == "vectorbtpro.portfolio.base.Portfolio.get_filled_close"
        assert vbt.get_refname(vbt.PF.get_sharpe_ratio) == "vectorbtpro.portfolio.base.Portfolio.get_sharpe_ratio"
        assert vbt.get_refname(vbt.PF.sharpe_ratio) == "vectorbtpro.portfolio.base.Portfolio.sharpe_ratio"
        assert vbt.get_refname(vbt.ADX) == "vectorbtpro.indicators.custom.adx.ADX"
        assert vbt.get_refname(vbt.ADX.adx) == "vectorbtpro.indicators.custom.adx.ADX.adx"
        assert vbt.get_refname(vbt.ADX.adx_above) == "vectorbtpro.indicators.custom.adx.ADX.adx_above"
        assert vbt.get_refname(pf.plot_value) == "vectorbtpro.portfolio.base.Portfolio.plot_value"
        assert vbt.get_refname(pf.get_filled_close) == "vectorbtpro.portfolio.base.Portfolio.get_filled_close"
        assert vbt.get_refname(pf.get_sharpe_ratio) == "vectorbtpro.portfolio.base.Portfolio.get_sharpe_ratio"

        assert vbt.get_refname((vbt, "utils")) == "vectorbtpro.utils"
        assert vbt.get_refname((vbt, "utils", "datetime_")) == "vectorbtpro.utils.datetime_"
        assert vbt.get_refname((vbt, "utils.datetime_")) == "vectorbtpro.utils.datetime_"
        assert vbt.get_refname((vbt.utils, "datetime_")) == "vectorbtpro.utils.datetime_"
        assert vbt.get_refname((vbt, "dt")) == "vectorbtpro.utils.datetime_"
        assert vbt.get_refname((vbt, "nb")) == "vectorbtpro.generic.nb"
        assert vbt.get_refname((vbt, "nb", "rolling_mean_nb")) == "vectorbtpro.generic.nb.rolling.rolling_mean_nb"
        assert vbt.get_refname((vbt, "settings")) == "vectorbtpro._settings.settings"
        assert vbt.get_refname((vbt, "Portfolio")) == "vectorbtpro.portfolio.base.Portfolio"
        assert vbt.get_refname((vbt, "PF")) == "vectorbtpro.portfolio.base.Portfolio"
        assert vbt.get_refname((vbt, "Data", "fetch_symbol")) == "vectorbtpro.data.base.Data.fetch_symbol"
        assert vbt.get_refname((vbt, "YFData", "fetch_symbol")) == "vectorbtpro.data.custom.yf.YFData.fetch_symbol"
        assert vbt.get_refname((vbt, "Data", "run")) == "vectorbtpro.data.base.Data.run"
        assert vbt.get_refname((vbt, "YFData", "run")) == "vectorbtpro.data.base.Data.run"
        assert vbt.get_refname((pf, "cash_sharing")) == "vectorbtpro.portfolio.base.Portfolio.cash_sharing"
        assert vbt.get_refname((pf, "close")) == "vectorbtpro.portfolio.base.Portfolio.close"
        assert vbt.get_refname((pf, "plot_value")) == "vectorbtpro.portfolio.base.Portfolio.plot_value"
        assert vbt.get_refname((pf, "get_filled_close")) == "vectorbtpro.portfolio.base.Portfolio.get_filled_close"
        assert vbt.get_refname((pf, "get_sharpe_ratio")) == "vectorbtpro.portfolio.base.Portfolio.get_sharpe_ratio"
        assert vbt.get_refname((pf, "sharpe_ratio")) == "vectorbtpro.portfolio.base.Portfolio.sharpe_ratio"

        assert vbt.get_refname("vbt") == "vectorbtpro"
        assert vbt.get_refname("vectorbtpro") == "vectorbtpro"
        assert vbt.get_refname("vbt.utils") == "vectorbtpro.utils"
        assert vbt.get_refname("utils") == "vectorbtpro.utils"
        assert vbt.get_refname("vbt.utils.datetime_") == "vectorbtpro.utils.datetime_"
        assert vbt.get_refname("datetime_") == "vectorbtpro.utils.datetime_"
        assert vbt.get_refname("vbt.dt") == "vectorbtpro.utils.datetime_"
        assert vbt.get_refname("vbt.nb") == "vectorbtpro.generic.nb"
        assert vbt.get_refname("vbt.nb.rolling_mean_nb") == "vectorbtpro.generic.nb.rolling.rolling_mean_nb"
        assert vbt.get_refname("rolling_mean_nb") == "vectorbtpro.generic.nb.rolling.rolling_mean_nb"
        assert vbt.get_refname("vbt._settings.settings") == "vectorbtpro._settings.settings"
        assert vbt.get_refname("vbt.settings") == "vectorbtpro._settings.settings"
        assert vbt.get_refname("settings") == "vectorbtpro._settings.settings"
        assert vbt.get_refname("vbt.Portfolio") == "vectorbtpro.portfolio.base.Portfolio"
        assert vbt.get_refname("vbt.PF") == "vectorbtpro.portfolio.base.Portfolio"
        assert vbt.get_refname("Portfolio") == "vectorbtpro.portfolio.base.Portfolio"
        assert vbt.get_refname("PF") == "vectorbtpro.portfolio.base.Portfolio"
        assert vbt.get_refname("Data.fetch_symbol") == "vectorbtpro.data.base.Data.fetch_symbol"
        assert vbt.get_refname("YFData.fetch_symbol") == "vectorbtpro.data.custom.yf.YFData.fetch_symbol"
        assert vbt.get_refname("Data.run") == "vectorbtpro.data.base.Data.run"
        assert vbt.get_refname("YFData.run") == "vectorbtpro.data.base.Data.run"
        assert vbt.get_refname("PF.cash_sharing") == "vectorbtpro.portfolio.base.Portfolio.cash_sharing"
        assert vbt.get_refname("PF.close") == "vectorbtpro.portfolio.base.Portfolio.close"
        assert vbt.get_refname("PF.plot_value") == "vectorbtpro.portfolio.base.Portfolio.plot_value"
        assert vbt.get_refname("PF.get_filled_close") == "vectorbtpro.portfolio.base.Portfolio.get_filled_close"
        assert vbt.get_refname("PF.get_sharpe_ratio") == "vectorbtpro.portfolio.base.Portfolio.get_sharpe_ratio"
        assert vbt.get_refname("PF.sharpe_ratio") == "vectorbtpro.portfolio.base.Portfolio.sharpe_ratio"
        assert vbt.get_refname("ADX") == "vectorbtpro.indicators.custom.adx.ADX"
        assert vbt.get_refname("ADX.adx") == "vectorbtpro.indicators.custom.adx.ADX.adx"
        assert vbt.get_refname("ADX.adx_above") == "vectorbtpro.indicators.custom.adx.ADX.adx_above"

    def test_get_obj(self):
        assert vbt.get_obj(vbt.Portfolio) is vbt.Portfolio
        assert vbt.get_obj(vbt) is vbt
        assert vbt.get_obj("Portfolio") is vbt.Portfolio

    def test_RefIndex(self):
        ref_index = vbt.RefIndex()
        ref_info = ref_index.get_info("DHitMeta")
        assert isinstance(ref_info, refs.RefInfo)
        assert ref_info.refname == "vectorbtpro.utils.refs.DHitMeta"
        assert ref_info.container == "vectorbtpro.utils.refs"
        assert ref_info.direct_members == [
            "vectorbtpro.utils.refs.DHitMeta.is_builtin",
            "vectorbtpro.utils.refs.DHitMeta.is_private",
            "vectorbtpro.utils.refs.DHitMeta.is_unreachable",
        ]
        assert ref_info.nested_members == [
            "vectorbtpro.utils.attr_.DefineMixin.asdict",
            "vectorbtpro.utils.attr_.DefineMixin.assert_field_not_missing",
            "vectorbtpro.utils.refs.DHitMeta.block",
            "vectorbtpro.utils.base.Base.chat",
            "vectorbtpro.utils.refs.DHitMeta.col_offset",
            "vectorbtpro.utils.refs.DHitMeta.end_col_offset",
            "vectorbtpro.utils.refs.DHitMeta.end_lineno",
            "vectorbtpro.utils.attr_.DefineMixin.fields",
            "vectorbtpro.utils.attr_.DefineMixin.fields_dict",
            "vectorbtpro.utils.base.Base.find_api",
            "vectorbtpro.utils.base.Base.find_assets",
            "vectorbtpro.utils.base.Base.find_docs",
            "vectorbtpro.utils.base.Base.find_examples",
            "vectorbtpro.utils.base.Base.find_messages",
            "vectorbtpro.utils.attr_.DefineMixin.get_field",
            "vectorbtpro.utils.hashing.Hashable.get_hash",
            "vectorbtpro.utils.hashing.Hashable.hash",
            "vectorbtpro.utils.attr_.DefineMixin.hash_key",
            "vectorbtpro.utils.attr_.DefineMixin.is_field_missing",
            "vectorbtpro.utils.attr_.DefineMixin.is_field_optional",
            "vectorbtpro.utils.attr_.DefineMixin.is_field_required",
            "vectorbtpro.utils.refs.DHitMeta.lineno",
            "vectorbtpro.utils.attr_.DefineMixin.merge_over",
            "vectorbtpro.utils.attr_.DefineMixin.merge_with",
            "vectorbtpro.utils.refs.DHitMeta.name",
            "vectorbtpro.utils.refs.DHitMeta.refname",
            "vectorbtpro.utils.attr_.DefineMixin.replace",
            "vectorbtpro.utils.attr_.DefineMixin.resolve",
            "vectorbtpro.utils.attr_.DefineMixin.resolve_field",
            "vectorbtpro.utils.refs.DHitMeta.role",
            "vectorbtpro.utils.refs.DHitMeta.scope_refname",
            "vectorbtpro.utils.refs.DHitMeta.source_line",
        ]
        assert ref_info.direct_bases == ["vectorbtpro.utils.attr_.DefineMixin"]
        assert ref_info.nested_bases == ["vectorbtpro.utils.hashing.Hashable", "vectorbtpro.utils.base.Base"]
        assert ref_info.direct_dependencies == [
            "vectorbtpro.utils.attr_.define",
            "vectorbtpro.utils.attr_.DefineMixin",
            "builtins.str",
            "vectorbtpro.utils.attr_.define.field",
            "builtins.int",
            "typing.Optional",
            "vectorbtpro.utils.refs.DBlock",
            "vectorbtpro.utils.refs.DRole",
        ]
        assert ref_info.nested_dependencies == ["builtins.property", "builtins.bool"]

    @requires_networkx
    def test_RefGraph(self):
        ref_graph = vbt.RefIndex(container_kinds=["module", "class"]).build_graph("DHitMeta")
        assert list(ref_graph.G.nodes()) == [
            "vectorbtpro.utils.refs.DHitMeta",
            "vectorbtpro.utils.refs",
            "vectorbtpro.utils.refs.DHitMeta.is_builtin",
            "vectorbtpro.utils.refs.DHitMeta.is_private",
            "vectorbtpro.utils.refs.DHitMeta.is_unreachable",
            "vectorbtpro.utils.attr_.DefineMixin.asdict",
            "vectorbtpro.utils.attr_.DefineMixin.assert_field_not_missing",
            "vectorbtpro.utils.refs.DHitMeta.block",
            "vectorbtpro.utils.base.Base.chat",
            "vectorbtpro.utils.refs.DHitMeta.col_offset",
            "vectorbtpro.utils.refs.DHitMeta.end_col_offset",
            "vectorbtpro.utils.refs.DHitMeta.end_lineno",
            "vectorbtpro.utils.attr_.DefineMixin.fields",
            "vectorbtpro.utils.attr_.DefineMixin.fields_dict",
            "vectorbtpro.utils.base.Base.find_api",
            "vectorbtpro.utils.base.Base.find_assets",
            "vectorbtpro.utils.base.Base.find_docs",
            "vectorbtpro.utils.base.Base.find_examples",
            "vectorbtpro.utils.base.Base.find_messages",
            "vectorbtpro.utils.attr_.DefineMixin.get_field",
            "vectorbtpro.utils.hashing.Hashable.get_hash",
            "vectorbtpro.utils.hashing.Hashable.hash",
            "vectorbtpro.utils.attr_.DefineMixin.hash_key",
            "vectorbtpro.utils.attr_.DefineMixin.is_field_missing",
            "vectorbtpro.utils.attr_.DefineMixin.is_field_optional",
            "vectorbtpro.utils.attr_.DefineMixin.is_field_required",
            "vectorbtpro.utils.refs.DHitMeta.lineno",
            "vectorbtpro.utils.attr_.DefineMixin.merge_over",
            "vectorbtpro.utils.attr_.DefineMixin.merge_with",
            "vectorbtpro.utils.refs.DHitMeta.name",
            "vectorbtpro.utils.refs.DHitMeta.refname",
            "vectorbtpro.utils.attr_.DefineMixin.replace",
            "vectorbtpro.utils.attr_.DefineMixin.resolve",
            "vectorbtpro.utils.attr_.DefineMixin.resolve_field",
            "vectorbtpro.utils.refs.DHitMeta.role",
            "vectorbtpro.utils.refs.DHitMeta.scope_refname",
            "vectorbtpro.utils.refs.DHitMeta.source_line",
            "vectorbtpro.utils.attr_.DefineMixin",
            "vectorbtpro.utils.hashing.Hashable",
            "vectorbtpro.utils.base.Base",
            "vectorbtpro.utils.attr_.define",
            "vectorbtpro.utils.attr_.define.field",
            "typing.Optional",
            "vectorbtpro.utils.refs.DBlock",
            "vectorbtpro.utils.refs.DRole",
            "vectorbtpro.utils",
            "vectorbtpro.utils.attr_",
            "vectorbtpro.utils.hashing",
            "vectorbtpro.utils.base",
            "typing",
            "vectorbtpro",
        ]
        assert list(ref_graph.G.edges()) == [
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.refs.DHitMeta.is_builtin"),
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.refs.DHitMeta.is_private"),
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.refs.DHitMeta.is_unreachable"),
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.attr_.DefineMixin.asdict"),
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.attr_.DefineMixin.assert_field_not_missing"),
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.refs.DHitMeta.block"),
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.base.Base.chat"),
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.refs.DHitMeta.col_offset"),
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.refs.DHitMeta.end_col_offset"),
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.refs.DHitMeta.end_lineno"),
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.attr_.DefineMixin.fields"),
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.attr_.DefineMixin.fields_dict"),
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.base.Base.find_api"),
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.base.Base.find_assets"),
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.base.Base.find_docs"),
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.base.Base.find_examples"),
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.base.Base.find_messages"),
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.attr_.DefineMixin.get_field"),
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.hashing.Hashable.get_hash"),
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.hashing.Hashable.hash"),
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.attr_.DefineMixin.hash_key"),
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.attr_.DefineMixin.is_field_missing"),
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.attr_.DefineMixin.is_field_optional"),
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.attr_.DefineMixin.is_field_required"),
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.refs.DHitMeta.lineno"),
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.attr_.DefineMixin.merge_over"),
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.attr_.DefineMixin.merge_with"),
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.refs.DHitMeta.name"),
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.refs.DHitMeta.refname"),
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.attr_.DefineMixin.replace"),
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.attr_.DefineMixin.resolve"),
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.attr_.DefineMixin.resolve_field"),
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.refs.DHitMeta.role"),
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.refs.DHitMeta.scope_refname"),
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.refs.DHitMeta.source_line"),
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.attr_.DefineMixin"),
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.hashing.Hashable"),
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.base.Base"),
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.attr_.define"),
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.attr_.define.field"),
            ("vectorbtpro.utils.refs.DHitMeta", "typing.Optional"),
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.refs.DBlock"),
            ("vectorbtpro.utils.refs.DHitMeta", "vectorbtpro.utils.refs.DRole"),
            ("vectorbtpro.utils.refs", "vectorbtpro.utils.refs.DHitMeta"),
            ("vectorbtpro.utils.refs", "vectorbtpro.utils.refs.DBlock"),
            ("vectorbtpro.utils.refs", "vectorbtpro.utils.refs.DRole"),
            ("vectorbtpro.utils.attr_.DefineMixin", "vectorbtpro.utils.attr_.DefineMixin.asdict"),
            ("vectorbtpro.utils.attr_.DefineMixin", "vectorbtpro.utils.attr_.DefineMixin.assert_field_not_missing"),
            ("vectorbtpro.utils.attr_.DefineMixin", "vectorbtpro.utils.attr_.DefineMixin.fields"),
            ("vectorbtpro.utils.attr_.DefineMixin", "vectorbtpro.utils.attr_.DefineMixin.fields_dict"),
            ("vectorbtpro.utils.attr_.DefineMixin", "vectorbtpro.utils.attr_.DefineMixin.get_field"),
            ("vectorbtpro.utils.attr_.DefineMixin", "vectorbtpro.utils.attr_.DefineMixin.hash_key"),
            ("vectorbtpro.utils.attr_.DefineMixin", "vectorbtpro.utils.attr_.DefineMixin.is_field_missing"),
            ("vectorbtpro.utils.attr_.DefineMixin", "vectorbtpro.utils.attr_.DefineMixin.is_field_optional"),
            ("vectorbtpro.utils.attr_.DefineMixin", "vectorbtpro.utils.attr_.DefineMixin.is_field_required"),
            ("vectorbtpro.utils.attr_.DefineMixin", "vectorbtpro.utils.attr_.DefineMixin.merge_over"),
            ("vectorbtpro.utils.attr_.DefineMixin", "vectorbtpro.utils.attr_.DefineMixin.merge_with"),
            ("vectorbtpro.utils.attr_.DefineMixin", "vectorbtpro.utils.attr_.DefineMixin.replace"),
            ("vectorbtpro.utils.attr_.DefineMixin", "vectorbtpro.utils.attr_.DefineMixin.resolve"),
            ("vectorbtpro.utils.attr_.DefineMixin", "vectorbtpro.utils.attr_.DefineMixin.resolve_field"),
            ("vectorbtpro.utils.hashing.Hashable", "vectorbtpro.utils.hashing.Hashable.get_hash"),
            ("vectorbtpro.utils.hashing.Hashable", "vectorbtpro.utils.hashing.Hashable.hash"),
            ("vectorbtpro.utils.base.Base", "vectorbtpro.utils.base.Base.chat"),
            ("vectorbtpro.utils.base.Base", "vectorbtpro.utils.base.Base.find_api"),
            ("vectorbtpro.utils.base.Base", "vectorbtpro.utils.base.Base.find_assets"),
            ("vectorbtpro.utils.base.Base", "vectorbtpro.utils.base.Base.find_docs"),
            ("vectorbtpro.utils.base.Base", "vectorbtpro.utils.base.Base.find_examples"),
            ("vectorbtpro.utils.base.Base", "vectorbtpro.utils.base.Base.find_messages"),
            ("vectorbtpro.utils.attr_.define", "vectorbtpro.utils.attr_.define.field"),
            ("vectorbtpro.utils", "vectorbtpro.utils.refs"),
            ("vectorbtpro.utils", "vectorbtpro.utils.attr_"),
            ("vectorbtpro.utils", "vectorbtpro.utils.hashing"),
            ("vectorbtpro.utils", "vectorbtpro.utils.base"),
            ("vectorbtpro.utils.attr_", "vectorbtpro.utils.attr_.DefineMixin"),
            ("vectorbtpro.utils.attr_", "vectorbtpro.utils.attr_.define"),
            ("vectorbtpro.utils.hashing", "vectorbtpro.utils.hashing.Hashable"),
            ("vectorbtpro.utils.base", "vectorbtpro.utils.base.Base"),
            ("typing", "typing.Optional"),
            ("vectorbtpro", "vectorbtpro.utils"),
        ]
