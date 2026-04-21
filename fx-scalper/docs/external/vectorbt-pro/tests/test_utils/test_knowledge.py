import os

import pytest

import vectorbtpro as vbt
from tests.utils import *
from vectorbtpro._dtypes import *


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True


def teardown_module():
    vbt.settings.reset()


dataset = [
    {"s": "ABC", "b": True, "d2": {"c": "red", "l": [1, 2]}},
    {"s": "BCD", "b": True, "d2": {"c": "blue", "l": [3, 4]}},
    {"s": "CDE", "b": False, "d2": {"c": "green", "l": [5, 6]}},
    {"s": "DEF", "b": False, "d2": {"c": "yellow", "l": [7, 8]}},
    {"s": "EFG", "b": False, "d2": {"c": "black", "l": [9, 10]}, "xyz": 123},
]
asset = vbt.KnowledgeAsset(dataset)


class TestKnowledge:
    def test_combine(self):
        assert vbt.KnowledgeAsset.combine((asset[:3], asset[3:])) == asset
        assert asset[:3].combine(asset[3:]) == asset

    def test_merge(self):
        assert vbt.KnowledgeAsset.merge((asset[:2], asset[3:])) == asset[3:]
        assert vbt.KnowledgeAsset.merge((asset[:2], asset[3:])) == asset[:2].merge(asset[3:])
        assert asset.merge(wrap=False) == dataset[-1]

    def test_item_methods(self):
        assert asset.get_items(0) == dataset[0]
        assert asset[0] == dataset[0]
        assert asset[:2] == vbt.KnowledgeAsset(data=dataset[:2])
        assert asset[[0, 2]] == vbt.KnowledgeAsset(data=[dataset[0], dataset[2]])
        assert asset[[True, False, True, False, False]] == vbt.KnowledgeAsset(data=[dataset[0], dataset[2]])
        with pytest.raises(Exception):
            asset[[True, False, True, False, False, True]]
        with pytest.raises(Exception):
            asset[[0, False, 2, False, False]]

        asset2 = asset.set_items(0, dataset[1])
        assert asset2[0] == dataset[1]
        asset2 = asset.copy()
        asset2[0] = dataset[1]
        assert asset2[0] == dataset[1]
        asset2 = asset.copy()
        asset2[:2] = [dataset[1], dataset[0]]
        assert asset2[:2] == vbt.KnowledgeAsset(data=[dataset[1], dataset[0]])
        asset2 = asset.copy()
        asset2[[0, 2]] = [dataset[2], dataset[0]]
        assert asset2[[0, 2]] == vbt.KnowledgeAsset(data=[dataset[2], dataset[0]])
        with pytest.raises(Exception):
            asset2[[0, 2, 3]] = [dataset[2], dataset[0]]
        with pytest.raises(Exception):
            asset2[[0, 2]] = [dataset[2], dataset[0], dataset[1]]
        asset2 = asset.copy()
        asset2[[True, False, True, False, False]] = [dataset[2], dataset[0]]
        assert asset2[[True, False, True, False, False]] == vbt.KnowledgeAsset(data=[dataset[2], dataset[0]])
        with pytest.raises(Exception):
            asset2[[True, False, True, False, False, True]] = [dataset[2], dataset[0]]
        asset2[[True, False, True, False, False]] = dataset[::-1]
        assert asset2[[True, False, True, False, False]] == vbt.KnowledgeAsset(data=[dataset[4], dataset[2]])
        with pytest.raises(Exception):
            asset2[[True, False, True, False, False, False]] = dataset[::-1]
        with pytest.raises(Exception):
            asset2[[0, False, 2, False, False]] = dataset[::-1]

        asset3 = asset.delete_items(0)
        assert asset3 == asset[1:]
        asset3 = asset.copy()
        del asset3[0]
        assert asset3 == asset[1:]
        asset3 = asset.copy()
        del asset3[:2]
        assert asset3 == asset[2:]
        asset3 = asset.copy()
        del asset3[[0, 2]]
        assert asset3 == asset[[1, 3, 4]]
        asset3 = asset.copy()
        with pytest.raises(Exception):
            del asset3[[0, 2, 5]]
        asset3 = asset.copy()
        del asset3[[True, False, True, False, False]]
        assert asset3 == asset[[1, 3, 4]]
        asset3 = asset.copy()
        with pytest.raises(Exception):
            del asset3[[True, False, True, False, False, True]]
        with pytest.raises(Exception):
            del asset3[[0, False, 2, False, False]]

        asset4 = asset.append_item(dataset[-1])
        assert len(asset4) == 6
        asset4 = asset.copy()
        asset4.append(dataset[-1])
        assert len(asset4) == 6

        asset5 = asset.extend_items([dataset[-1]])
        assert len(asset5) == 6
        asset5 = asset.copy()
        asset5.extend([dataset[-1]])
        assert len(asset5) == 6

        assert asset.unique("b", keep="first") == asset[[0, 2]]
        assert asset.unique("b", keep="last") == asset[[1, 4]]

        assert asset.sort("d2.c", ascending=True) == asset[[4, 1, 2, 0, 3]]
        assert asset.sort("d2.c", ascending=False) == asset[[3, 0, 2, 1, 4]]

        assert asset.shuffle(seed=42) == asset[[3, 1, 2, 4, 0]]

        assert asset.sample(seed=42, wrap=False) == dataset[0]

    def test_apply(self):
        assert asset.apply(["flatten", ("query", len)]) == [5, 5, 5, 5, 6]
        assert asset.apply("query(flatten(d), len)") == [5, 5, 5, 5, 6]

    def test_get(self):
        assert asset.get() == dataset
        assert asset.get("d2.l[0]") == [1, 3, 5, 7, 9]
        assert asset.get("d2.l", source=lambda x: sum(x)) == [3, 7, 11, 15, 19]
        assert asset.get("d2.l[0]", keep_path=True) == [
            {"d2": {"l": {0: 1}}},
            {"d2": {"l": {0: 3}}},
            {"d2": {"l": {0: 5}}},
            {"d2": {"l": {0: 7}}},
            {"d2": {"l": {0: 9}}},
        ]
        assert asset.get(["d2.l[0]", "d2.l[1]"]) == [
            {"d2": {"l": {0: 1, 1: 2}}},
            {"d2": {"l": {0: 3, 1: 4}}},
            {"d2": {"l": {0: 5, 1: 6}}},
            {"d2": {"l": {0: 7, 1: 8}}},
            {"d2": {"l": {0: 9, 1: 10}}},
        ]
        assert asset.get("xyz", skip_missing=True) == [123]

    def test_set(self):
        assert asset.set(lambda d: sum(d["d2"]["l"])).get() == [3, 7, 11, 15, 19]
        assert (
            asset.set(lambda d: sum(d["d2"]["l"]), path="d2.sum").get()
            == asset.set(lambda x: sum(x["l"]), path="d2.sum").get()
            == asset.set(lambda l: sum(l), path="d2.sum").get()
            == [
                {"s": "ABC", "b": True, "d2": {"c": "red", "l": [1, 2], "sum": 3}},
                {"s": "BCD", "b": True, "d2": {"c": "blue", "l": [3, 4], "sum": 7}},
                {"s": "CDE", "b": False, "d2": {"c": "green", "l": [5, 6], "sum": 11}},
                {"s": "DEF", "b": False, "d2": {"c": "yellow", "l": [7, 8], "sum": 15}},
                {"s": "EFG", "b": False, "d2": {"c": "black", "l": [9, 10], "sum": 19}, "xyz": 123},
            ]
        )
        assert asset.set(lambda l: sum(l), path="d2.l").get() == [
            {"s": "ABC", "b": True, "d2": {"c": "red", "l": 3}},
            {"s": "BCD", "b": True, "d2": {"c": "blue", "l": 7}},
            {"s": "CDE", "b": False, "d2": {"c": "green", "l": 11}},
            {"s": "DEF", "b": False, "d2": {"c": "yellow", "l": 15}},
            {"s": "EFG", "b": False, "d2": {"c": "black", "l": 19}, "xyz": 123},
        ]

    def test_remove(self):
        assert asset.remove("d2.l[0]").get() == [
            {"s": "ABC", "b": True, "d2": {"c": "red", "l": [2]}},
            {"s": "BCD", "b": True, "d2": {"c": "blue", "l": [4]}},
            {"s": "CDE", "b": False, "d2": {"c": "green", "l": [6]}},
            {"s": "DEF", "b": False, "d2": {"c": "yellow", "l": [8]}},
            {"s": "EFG", "b": False, "d2": {"c": "black", "l": [10]}, "xyz": 123},
        ]
        assert asset.remove("xyz", skip_missing=True).get() == [
            {"s": "ABC", "b": True, "d2": {"c": "red", "l": [1, 2]}},
            {"s": "BCD", "b": True, "d2": {"c": "blue", "l": [3, 4]}},
            {"s": "CDE", "b": False, "d2": {"c": "green", "l": [5, 6]}},
            {"s": "DEF", "b": False, "d2": {"c": "yellow", "l": [7, 8]}},
            {"s": "EFG", "b": False, "d2": {"c": "black", "l": [9, 10]}},
        ]

    def test_move(self):
        assert asset.move("d2.l", "l").get() == [
            {"s": "ABC", "b": True, "d2": {"c": "red"}, "l": [1, 2]},
            {"s": "BCD", "b": True, "d2": {"c": "blue"}, "l": [3, 4]},
            {"s": "CDE", "b": False, "d2": {"c": "green"}, "l": [5, 6]},
            {"s": "DEF", "b": False, "d2": {"c": "yellow"}, "l": [7, 8]},
            {"s": "EFG", "b": False, "d2": {"c": "black"}, "xyz": 123, "l": [9, 10]},
        ]
        assert (
            asset.move({"d2.c": "c", "b": "d2.b"}).get()
            == asset.move(["d2.c", "b"], ["c", "d2.b"]).get()
            == [
                {"s": "ABC", "d2": {"l": [1, 2], "b": True}, "c": "red"},
                {"s": "BCD", "d2": {"l": [3, 4], "b": True}, "c": "blue"},
                {"s": "CDE", "d2": {"l": [5, 6], "b": False}, "c": "green"},
                {"s": "DEF", "d2": {"l": [7, 8], "b": False}, "c": "yellow"},
                {"s": "EFG", "d2": {"l": [9, 10], "b": False}, "xyz": 123, "c": "black"},
            ]
        )

    def test_rename(self):
        assert asset.rename("d2.l", "x").get() == [
            {"s": "ABC", "b": True, "d2": {"c": "red", "x": [1, 2]}},
            {"s": "BCD", "b": True, "d2": {"c": "blue", "x": [3, 4]}},
            {"s": "CDE", "b": False, "d2": {"c": "green", "x": [5, 6]}},
            {"s": "DEF", "b": False, "d2": {"c": "yellow", "x": [7, 8]}},
            {"s": "EFG", "b": False, "d2": {"c": "black", "x": [9, 10]}, "xyz": 123},
        ]
        assert asset.rename("xyz", "zyx", skip_missing=True, changed_only=True).get() == [
            {"s": "EFG", "b": False, "d2": {"c": "black", "l": [9, 10]}, "zyx": 123},
        ]

    def test_reorder(self):
        assert (
            asset.reorder(["xyz", ...], skip_missing=True).get()
            == asset.reorder(lambda x: ["xyz", ...] if "xyz" in x else [...]).get()
            == [
                {"s": "ABC", "b": True, "d2": {"c": "red", "l": [1, 2]}},
                {"s": "BCD", "b": True, "d2": {"c": "blue", "l": [3, 4]}},
                {"s": "CDE", "b": False, "d2": {"c": "green", "l": [5, 6]}},
                {"s": "DEF", "b": False, "d2": {"c": "yellow", "l": [7, 8]}},
                {"xyz": 123, "s": "EFG", "b": False, "d2": {"c": "black", "l": [9, 10]}},
            ]
        )

        assert asset.reorder("descending", path="d2.l").get() == [
            {"s": "ABC", "b": True, "d2": {"c": "red", "l": [2, 1]}},
            {"s": "BCD", "b": True, "d2": {"c": "blue", "l": [4, 3]}},
            {"s": "CDE", "b": False, "d2": {"c": "green", "l": [6, 5]}},
            {"s": "DEF", "b": False, "d2": {"c": "yellow", "l": [8, 7]}},
            {"s": "EFG", "b": False, "d2": {"c": "black", "l": [10, 9]}, "xyz": 123},
        ]

    def test_query(self):
        assert (
            asset.query("d['s'] == 'ABC'")
            == asset.query("x['s'] == 'ABC'")
            == asset.query("s == 'ABC'")
            == [
                {"s": "ABC", "b": True, "d2": {"c": "red", "l": [1, 2]}},
            ]
        )
        assert asset.query("x['s'] == 'ABC'", return_type="bool") == [True, False, False, False, False]
        assert (
            asset.query("find('BC', s)")
            == asset.query(lambda s: "BC" in s)
            == [
                {"s": "ABC", "b": True, "d2": {"c": "red", "l": [1, 2]}},
                {"s": "BCD", "b": True, "d2": {"c": "blue", "l": [3, 4]}},
            ]
        )
        try:
            import jmespath

            assert asset.query("[?contains(s, 'BC')].s", query_engine="jmespath") == ["ABC", "BCD"]
            assert asset.query("[].d2.c", query_engine="jmespath") == ["red", "blue", "green", "yellow", "black"]
            assert asset.query("[?d2.c != `blue`].d2.l", query_engine="jmespath") == [[1, 2], [5, 6], [7, 8], [9, 10]]
        except ImportError:
            pass
        try:
            import jsonpath.ext

            assert asset.query("$[*].d2.c", query_engine="jsonpath.ext") == ["red", "blue", "green", "yellow", "black"]
            assert asset.query("$[?(@.b == true)].s", query_engine="jsonpath.ext") == ["ABC", "BCD"]
        except ImportError:
            pass
        assert asset.query("s[b]", query_engine="pandas") == ["ABC", "BCD"]

    def test_find(self):
        assert asset.find("BC").get() == [
            {"s": "ABC", "b": True, "d2": {"c": "red", "l": [1, 2]}},
            {"s": "BCD", "b": True, "d2": {"c": "blue", "l": [3, 4]}},
        ]
        assert asset.find("BC", return_type="bool").get() == [True, True, False, False, False]
        assert asset.find(vbt.Not("BC")).get() == [
            {"s": "CDE", "b": False, "d2": {"c": "green", "l": [5, 6]}},
            {"s": "DEF", "b": False, "d2": {"c": "yellow", "l": [7, 8]}},
            {"s": "EFG", "b": False, "d2": {"c": "black", "l": [9, 10]}, "xyz": 123},
        ]
        assert asset.find("bc", ignore_case=True).get() == [
            {"s": "ABC", "b": True, "d2": {"c": "red", "l": [1, 2]}},
            {"s": "BCD", "b": True, "d2": {"c": "blue", "l": [3, 4]}},
        ]
        assert asset.find("bl", path="d2.c").get() == [
            {"s": "BCD", "b": True, "d2": {"c": "blue", "l": [3, 4]}},
            {"s": "EFG", "b": False, "d2": {"c": "black", "l": [9, 10]}, "xyz": 123},
        ]
        assert asset.find(5, path="d2.l[0]").get() == [{"s": "CDE", "b": False, "d2": {"c": "green", "l": [5, 6]}}]
        assert asset.find(True, path="d2.l", source=lambda x: sum(x) >= 10).get() == [
            {"s": "CDE", "b": False, "d2": {"c": "green", "l": [5, 6]}},
            {"s": "DEF", "b": False, "d2": {"c": "yellow", "l": [7, 8]}},
            {"s": "EFG", "b": False, "d2": {"c": "black", "l": [9, 10]}, "xyz": 123},
        ]
        assert asset.find(["A", "B", "C"]).get() == [
            {"s": "ABC", "b": True, "d2": {"c": "red", "l": [1, 2]}},
            {"s": "BCD", "b": True, "d2": {"c": "blue", "l": [3, 4]}},
            {"s": "CDE", "b": False, "d2": {"c": "green", "l": [5, 6]}},
        ]
        assert asset.find(["A", "B", "C"], find_all=True).get() == [
            {"s": "ABC", "b": True, "d2": {"c": "red", "l": [1, 2]}},
        ]
        assert asset.find(r"[ABC]+", mode="regex").get() == [
            {"s": "ABC", "b": True, "d2": {"c": "red", "l": [1, 2]}},
            {"s": "BCD", "b": True, "d2": {"c": "blue", "l": [3, 4]}},
            {"s": "CDE", "b": False, "d2": {"c": "green", "l": [5, 6]}},
        ]
        try:
            import fuzzysearch

            assert asset.find("yenlow", mode="fuzzy").get() == [
                {"s": "DEF", "b": False, "d2": {"c": "yellow", "l": [7, 8]}},
            ]
            assert asset.find("yenlow", mode="fuzzy", return_type="match").get() == "yellow"
            assert asset.find("yenlow", mode="fuzzy", return_type="match", merge_matches=False).get() == [
                [],
                [],
                [],
                ["yellow"],
                [],
            ]
            assert asset.find("yenlow", mode="fuzzy", return_type="match", return_path=True).get() == [
                {},
                {},
                {},
                {("d2", "c"): ["yellow"]},
                {},
            ]
        except ImportError:
            pass
        assert asset.find("xyz", in_dumps=True).get() == [
            {"s": "EFG", "b": False, "d2": {"c": "black", "l": [9, 10]}, "xyz": 123},
        ]

    def test_find_replace(self):
        assert asset.find_replace("BC", "XY").get() == [
            {"s": "AXY", "b": True, "d2": {"c": "red", "l": [1, 2]}},
            {"s": "XYD", "b": True, "d2": {"c": "blue", "l": [3, 4]}},
            {"s": "CDE", "b": False, "d2": {"c": "green", "l": [5, 6]}},
            {"s": "DEF", "b": False, "d2": {"c": "yellow", "l": [7, 8]}},
            {"s": "EFG", "b": False, "d2": {"c": "black", "l": [9, 10]}, "xyz": 123},
        ]
        assert asset.find_replace("BC", "XY", changed_only=True).get() == [
            {"s": "AXY", "b": True, "d2": {"c": "red", "l": [1, 2]}},
            {"s": "XYD", "b": True, "d2": {"c": "blue", "l": [3, 4]}},
        ]
        assert asset.find_replace(r"(D)E(F)", r"\1X\2", mode="regex", changed_only=True).get() == [
            {"s": "DXF", "b": False, "d2": {"c": "yellow", "l": [7, 8]}},
        ]
        assert asset.find_replace(True, False, changed_only=True).get() == [
            {"s": "ABC", "b": False, "d2": {"c": "red", "l": [1, 2]}},
            {"s": "BCD", "b": False, "d2": {"c": "blue", "l": [3, 4]}},
        ]
        assert asset.find_replace(3, 30, path="d2.l", changed_only=True).get() == [
            {"s": "BCD", "b": True, "d2": {"c": "blue", "l": [30, 4]}},
        ]
        assert (
            asset.find_replace({1: 10, 4: 40}, path="d2.l", changed_only=True).get()
            == asset.find_replace({1: 10, 4: 40}, path=["d2.l[0]", "d2.l[1]"], changed_only=True).get()
            == [
                {"s": "ABC", "b": True, "d2": {"c": "red", "l": [10, 2]}},
                {"s": "BCD", "b": True, "d2": {"c": "blue", "l": [3, 40]}},
            ]
        )
        assert asset.find_replace({1: 10, 4: 40}, find_all=True, changed_only=True).get() == []
        assert asset.find_replace({1: 10, 2: 20}, find_all=True, changed_only=True).get() == [
            {"s": "ABC", "b": True, "d2": {"c": "red", "l": [10, 20]}},
        ]
        assert asset.find_replace("a", "X", path=["s", "d2.c"], ignore_case=True, changed_only=True).get() == [
            {"s": "XBC", "b": True, "d2": {"c": "red", "l": [1, 2]}},
            {"s": "EFG", "b": False, "d2": {"c": "blXck", "l": [9, 10]}, "xyz": 123},
        ]
        assert asset.find_replace(123, 456, path="xyz", skip_missing=True, changed_only=True).get() == [
            {"s": "EFG", "b": False, "d2": {"c": "black", "l": [9, 10]}, "xyz": 456}
        ]

    def test_flatten(self):
        assert asset.flatten().get() == [
            {"s": "ABC", "b": True, ("d2", "c"): "red", ("d2", "l", 0): 1, ("d2", "l", 1): 2},
            {"s": "BCD", "b": True, ("d2", "c"): "blue", ("d2", "l", 0): 3, ("d2", "l", 1): 4},
            {"s": "CDE", "b": False, ("d2", "c"): "green", ("d2", "l", 0): 5, ("d2", "l", 1): 6},
            {"s": "DEF", "b": False, ("d2", "c"): "yellow", ("d2", "l", 0): 7, ("d2", "l", 1): 8},
            {"s": "EFG", "b": False, ("d2", "c"): "black", ("d2", "l", 0): 9, ("d2", "l", 1): 10, "xyz": 123},
        ]

    def test_unflatten(self):
        assert asset.flatten().unflatten().get() == dataset

    def test_dump(self):
        assert (
            asset.dump(source="{i: d}", default_flow_style=True).join()
            == """{0: {s: ABC, b: true, d2: {c: red, l: [1, 2]}}}
{1: {s: BCD, b: true, d2: {c: blue, l: [3, 4]}}}
{2: {s: CDE, b: false, d2: {c: green, l: [5, 6]}}}
{3: {s: DEF, b: false, d2: {c: yellow, l: [7, 8]}}}
{4: {s: EFG, b: false, d2: {c: black, l: [9, 10]}, xyz: 123}}
"""
        )

    def test_reduce(self):
        assert (
            asset.reduce(lambda d1, d2: vbt.merge_dicts(d1, d2))
            == asset.reduce(vbt.merge_dicts)
            == asset.reduce("{**d1, **d2}")
            == {"s": "EFG", "b": False, "d2": {"c": "black", "l": [9, 10]}, "xyz": 123}
        )
        assert asset.reduce("{**d1, **d2}", by="b") == [
            {"s": "BCD", "b": True, "d2": {"c": "blue", "l": [3, 4]}},
            {"s": "EFG", "b": False, "d2": {"c": "black", "l": [9, 10]}, "xyz": 123},
        ]
