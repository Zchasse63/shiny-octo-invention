import os
from itertools import combinations, product

import pytest

import vectorbtpro as vbt
from tests.utils import *
from vectorbtpro._dtypes import *
from vectorbtpro.utils import params, template

seed = 42


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True


def teardown_module():
    vbt.settings.reset()


class TestParams:
    def test_create_param_combs(self):
        assert params.generate_param_combs((combinations, [0, 1, 2, 3], 2)) == [[0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3]]
        assert params.generate_param_combs((product, (combinations, [0, 1, 2, 3], 2), [4, 5])) == [
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2],
            [1, 1, 2, 2, 3, 3, 2, 2, 3, 3, 3, 3],
            [4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5],
        ]
        assert params.generate_param_combs((product, (combinations, [0, 1, 2], 2), (combinations, [3, 4, 5], 2))) == [
            [0, 0, 0, 0, 0, 0, 1, 1, 1],
            [1, 1, 1, 2, 2, 2, 2, 2, 2],
            [3, 3, 4, 3, 3, 4, 3, 3, 4],
            [4, 5, 5, 4, 5, 5, 4, 5, 5],
        ]

    def test_find_params(self):
        assert vbt.Parameterizer.find_params_in_obj(
            {
                "a": 1,
                "b": vbt.Param([1, 2, 3]),
                "c": {"d": 2, "e": vbt.Param([1, 2, 3]), "f": (3, vbt.Param([1, 2, 3]))},
            }
        ) == {
            "b": vbt.Param([1, 2, 3]),
            ("c", "e"): vbt.Param([1, 2, 3]),
            ("c", "f", 1): vbt.Param([1, 2, 3]),
        }

    def test_combine_params(self):
        param_product = vbt.combine_params(
            param_dct={"a": vbt.Param(np.arange(2)), "b": vbt.Param(np.arange(2)), "c": vbt.Param(np.arange(2))},
            build_grid=False,
        )[0]
        assert param_product["a"] == [0, 0, 0, 0, 1, 1, 1, 1]
        assert param_product["b"] == [0, 0, 1, 1, 0, 0, 1, 1]
        assert param_product["c"] == [0, 1, 0, 1, 0, 1, 0, 1]
        param_product = vbt.combine_params(
            param_dct={
                "a": vbt.Param(np.arange(1000)),
                "b": vbt.Param(np.arange(1000)),
                "c": vbt.Param(np.arange(1000)),
            },
            build_grid=False,
            seed=seed,
            random_subset=10,
        )[0]
        assert param_product["a"] == [85, 94, 201, 433, 438, 526, 654, 697, 773, 858]
        assert param_product["b"] == [945, 177, 469, 15, 878, 478, 571, 368, 956, 597]
        assert param_product["c"] == [638, 347, 535, 233, 436, 978, 513, 26, 41, 915]
        param_product = vbt.combine_params(
            param_dct={
                "a": vbt.Param(np.arange(1000)),
                "b": vbt.Param(np.arange(1000)),
                "c": vbt.Param(np.arange(1000)),
            },
            build_grid=False,
            seed=seed,
            random_subset=10,
            random_sort=False,
        )[0]
        assert param_product["a"] == [654, 433, 201, 773, 438, 858, 85, 94, 697, 526]
        assert param_product["b"] == [571, 15, 469, 956, 878, 597, 945, 177, 368, 478]
        assert param_product["c"] == [513, 233, 535, 41, 436, 915, 638, 347, 26, 978]
        param_product = vbt.combine_params(
            param_dct={
                "a": vbt.Param(np.arange(1000)),
                "b": vbt.Param(np.arange(1000)),
                "c": vbt.Param(np.arange(1000)),
            },
            build_grid=False,
            seed=seed,
            random_subset=10,
            grid_indices=slice(None, None, 1000),
        )[0]
        assert param_product["a"] == [85, 89, 94, 201, 433, 438, 654, 697, 773, 858]
        assert param_product["b"] == [945, 250, 177, 469, 13, 875, 566, 366, 949, 594]
        assert param_product["c"] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        param_product = vbt.combine_params(
            param_dct={
                "a": vbt.Param(np.arange(1000), condition="a > b"),
                "b": vbt.Param(np.arange(1000), condition="b > c"),
                "c": vbt.Param(np.arange(1000)),
            },
            build_grid=False,
            seed=seed,
            random_subset=10,
        )[0]
        assert param_product["a"] == [450, 475, 700, 781, 786, 822, 858, 905, 922, 967]
        assert param_product["b"] == [227, 330, 354, 643, 513, 545, 827, 370, 744, 410]
        assert param_product["c"] == [92, 226, 67, 402, 128, 443, 276, 76, 366, 325]
        param_product = vbt.combine_params(
            param_dct={
                "a": vbt.Param(np.arange(1000), condition=vbt.RepFunc(lambda a, b: a > b)),
                "b": vbt.Param(np.arange(1000), condition=vbt.RepFunc(lambda b, c: b > c)),
                "c": vbt.Param(np.arange(1000)),
            },
            build_grid=False,
            seed=seed,
            random_subset=10,
        )[0]
        assert param_product["a"] == [450, 475, 700, 781, 786, 822, 858, 905, 922, 967]
        assert param_product["b"] == [227, 330, 354, 643, 513, 545, 827, 370, 744, 410]
        assert param_product["c"] == [92, 226, 67, 402, 128, 443, 276, 76, 366, 325]
        param_product = vbt.combine_params(
            param_dct={
                "a": vbt.Param(np.arange(1000), condition="__a__ > __b__"),
                "b": vbt.Param(np.arange(1000), condition="__b__ > __c__"),
                "c": vbt.Param(np.arange(1000)),
            },
            build_grid=False,
            seed=seed,
            random_subset=10,
        )[0]
        assert param_product["a"] == [450, 475, 700, 781, 786, 822, 858, 905, 922, 967]
        assert param_product["b"] == [227, 330, 354, 643, 513, 545, 827, 370, 744, 410]
        assert param_product["c"] == [92, 226, 67, 402, 128, 443, 276, 76, 366, 325]
        with pytest.raises(Exception):
            vbt.combine_params(
                param_dct={
                    "a": vbt.Param(np.arange(1000), hide=True, condition="__a__ > __b__"),
                    "b": vbt.Param(np.arange(1000), hide=True, condition="__b__ > __c__"),
                    "c": vbt.Param(np.arange(1000)),
                },
                build_grid=False,
                seed=seed,
                random_subset=10,
            )
        param_product = vbt.combine_params(
            param_dct={
                "a": vbt.Param(np.arange(1000), hide=True, keys=np.arange(1000), condition="__a__ > __b__"),
                "b": vbt.Param(np.arange(1000), hide=True, keys=np.arange(1000), condition="__b__ > __c__"),
                "c": vbt.Param(np.arange(1000)),
            },
            build_grid=False,
            seed=seed,
            random_subset=10,
        )[0]
        assert param_product["a"] == [450, 475, 700, 781, 786, 822, 858, 905, 922, 967]
        assert param_product["b"] == [227, 330, 354, 643, 513, 545, 827, 370, 744, 410]
        assert param_product["c"] == [92, 226, 67, 402, 128, 443, 276, 76, 366, 325]
        param_product = vbt.combine_params(
            param_dct={
                "a": vbt.Param(np.arange(1000), condition="a > b"),
                "b": vbt.Param(np.arange(1000), condition="b > c"),
                "c": vbt.Param(np.arange(1000)),
            },
            build_grid=False,
            seed=seed,
            random_subset=10,
            random_sort=False,
        )[0]
        assert param_product["a"] == [786, 781, 822, 450, 858, 700, 922, 967, 905, 475]
        assert param_product["b"] == [513, 643, 545, 227, 827, 354, 744, 410, 370, 330]
        assert param_product["c"] == [128, 402, 443, 92, 276, 67, 366, 325, 76, 226]
        param_product = vbt.combine_params(
            param_dct={
                "a": vbt.Param(np.arange(5), condition="a > b"),
                "b": vbt.Param(np.arange(5), condition="b > c"),
                "c": vbt.Param(np.arange(5)),
            },
            build_grid=False,
            seed=seed,
            random_subset=10,
            random_replace=False,
        )[0]
        assert param_product["a"] == [2, 3, 3, 3, 4, 4, 4, 4, 4, 4]
        assert param_product["b"] == [1, 1, 2, 2, 1, 2, 2, 3, 3, 3]
        assert param_product["c"] == [0, 0, 0, 1, 0, 0, 1, 0, 1, 2]
        param_product = vbt.combine_params(
            param_dct={
                "a": vbt.Param(np.arange(5), condition="a > b"),
                "b": vbt.Param(np.arange(5), condition="b > c"),
                "c": vbt.Param(np.arange(5)),
            },
            build_grid=False,
            seed=seed,
            random_subset=10,
            random_replace=True,
        )[0]
        assert param_product["a"] == [2, 2, 3, 3, 4, 4, 4, 4, 4, 4]
        assert param_product["b"] == [1, 1, 1, 2, 1, 1, 2, 2, 3, 3]
        assert param_product["c"] == [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
        param_product = vbt.combine_params(
            param_dct={
                "a": vbt.Param(np.arange(5), condition="a > b"),
                "b": vbt.Param(np.arange(5), condition="b > c"),
                "c": vbt.Param(np.arange(5)),
            },
            build_grid=False,
            seed=seed,
            random_subset=10,
            random_replace=True,
            max_guesses=2.0,
        )[0]
        assert param_product["a"] == [2, 3, 3]
        assert param_product["b"] == [1, 1, 2]
        assert param_product["c"] == [0, 0, 0]
        param_product = vbt.combine_params(
            param_dct={
                "a": vbt.Param(np.arange(5), condition="a > b"),
                "b": vbt.Param(np.arange(5), condition="b > c"),
                "c": vbt.Param(np.arange(5)),
            },
            build_grid=False,
            seed=seed,
            random_subset=10,
            random_replace=True,
            max_misses=2.0,
        )[0]
        assert param_product["a"] == [2, 3, 3, 4, 4, 4]
        assert param_product["b"] == [1, 1, 2, 1, 2, 3]
        assert param_product["c"] == [0, 0, 0, 0, 1, 1]
        param_product = vbt.combine_params(
            param_dct={
                "a": vbt.Param(np.arange(1000), condition="a > b"),
                "b": vbt.Param(np.arange(1000), condition="b > c"),
                "c": vbt.Param(np.arange(1000)),
            },
            build_grid=False,
            seed=seed,
            random_subset=10,
            grid_indices=slice(None, None, 1000),
        )[0]
        assert param_product["a"] == [228, 368, 777, 780, 784, 966, 970, 982, 982, 995]
        assert param_product["b"] == [106, 65, 281, 379, 709, 161, 98, 561, 863, 122]
        assert param_product["c"] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        param_product = vbt.combine_params(
            param_dct={"a": vbt.Param(np.arange(2)), "b": vbt.Param(np.arange(2)), "c": vbt.Param(np.arange(2))},
            build_grid=True,
        )[0]
        assert param_product["a"] == [0, 0, 0, 0, 1, 1, 1, 1]
        assert param_product["b"] == [0, 0, 1, 1, 0, 0, 1, 1]
        assert param_product["c"] == [0, 1, 0, 1, 0, 1, 0, 1]
        param_product = vbt.combine_params(
            param_dct={"a": vbt.Param(np.arange(2)), "b": vbt.Param(np.arange(2)), "c": vbt.Param(np.arange(2))},
            build_grid=True,
            grid_indices=slice(None, None, 2),
        )[0]
        assert param_product["a"] == [0, 0, 1, 1]
        assert param_product["b"] == [0, 1, 0, 1]
        assert param_product["c"] == [0, 0, 0, 0]
        param_product = vbt.combine_params(
            param_dct={
                "a": vbt.Param(np.arange(5), condition="a > b"),
                "b": vbt.Param(np.arange(5), condition="b > c"),
                "c": vbt.Param(np.arange(5)),
            },
            build_grid=True,
            seed=seed,
            random_subset=10,
            random_replace=False,
        )[0]
        assert param_product["a"] == [2, 3, 3, 3, 4, 4, 4, 4, 4, 4]
        assert param_product["b"] == [1, 1, 2, 2, 1, 2, 2, 3, 3, 3]
        assert param_product["c"] == [0, 0, 0, 1, 0, 0, 1, 0, 1, 2]
        param_product = vbt.combine_params(
            param_dct={
                "a": vbt.Param(np.arange(5), condition=vbt.RepFunc(lambda a, b: a > b)),
                "b": vbt.Param(np.arange(5), condition=vbt.RepFunc(lambda b, c: b > c)),
                "c": vbt.Param(np.arange(5)),
            },
            build_grid=True,
            seed=seed,
            random_subset=10,
            random_replace=False,
        )[0]
        assert param_product["a"] == [2, 3, 3, 3, 4, 4, 4, 4, 4, 4]
        assert param_product["b"] == [1, 1, 2, 2, 1, 2, 2, 3, 3, 3]
        assert param_product["c"] == [0, 0, 0, 1, 0, 0, 1, 0, 1, 2]
        param_product = vbt.combine_params(
            param_dct={
                "a": vbt.Param(np.arange(5), condition="__a__ > __b__"),
                "b": vbt.Param(np.arange(5), condition="__b__ > __c__"),
                "c": vbt.Param(np.arange(5)),
            },
            build_grid=True,
            seed=seed,
            random_subset=10,
            random_replace=False,
        )[0]
        assert param_product["a"] == [2, 3, 3, 3, 4, 4, 4, 4, 4, 4]
        assert param_product["b"] == [1, 1, 2, 2, 1, 2, 2, 3, 3, 3]
        assert param_product["c"] == [0, 0, 0, 1, 0, 0, 1, 0, 1, 2]
        with pytest.raises(Exception):
            vbt.combine_params(
                param_dct={
                    "a": vbt.Param(np.arange(5), hide=True, condition="__a__ > __b__"),
                    "b": vbt.Param(np.arange(5), hide=True, condition="__b__ > __c__"),
                    "c": vbt.Param(np.arange(5)),
                },
                build_grid=True,
                seed=seed,
                random_subset=10,
                random_replace=False,
            )
        param_product = vbt.combine_params(
            param_dct={
                "a": vbt.Param(np.arange(5), hide=True, keys=np.arange(5), condition="__a__ > __b__"),
                "b": vbt.Param(np.arange(5), hide=True, keys=np.arange(5), condition="__b__ > __c__"),
                "c": vbt.Param(np.arange(5)),
            },
            build_grid=True,
            seed=seed,
            random_subset=10,
            random_replace=False,
        )[0]
        assert param_product["a"] == [2, 3, 3, 3, 4, 4, 4, 4, 4, 4]
        assert param_product["b"] == [1, 1, 2, 2, 1, 2, 2, 3, 3, 3]
        assert param_product["c"] == [0, 0, 0, 1, 0, 0, 1, 0, 1, 2]
        param_product = vbt.combine_params(
            param_dct={
                "a": vbt.Param(np.arange(5), condition="a > b"),
                "b": vbt.Param(np.arange(5), condition="b > c"),
                "c": vbt.Param(np.arange(5)),
            },
            build_grid=True,
            seed=seed,
            random_subset=10,
            random_replace=True,
        )[0]
        assert param_product["a"] == [2, 2, 2, 3, 4, 4, 4, 4, 4, 4]
        assert param_product["b"] == [1, 1, 1, 2, 1, 1, 2, 2, 3, 3]
        assert param_product["c"] == [0, 0, 0, 0, 0, 0, 1, 1, 0, 1]

    def test_parameterized(self):
        def f(a, *my_args, b=2, **my_kwargs):
            return a, my_args, b, my_kwargs

        def merge_func(results, param_index):
            return results, param_index

        fp = vbt.parameterized(f, merge_func=merge_func, merge_kwargs=dict(param_index=template.Rep("param_index")))
        assert fp(1) == (1, (), 2, {})
        assert fp(1, 2) == (1, (2,), 2, {})
        assert fp(1, 2, 3) == (1, (2, 3), 2, {})
        assert fp(1, 2, 3, b=4) == (1, (2, 3), 4, {})
        assert fp(1, 2, 3, b=4, c=5) == (1, (2, 3), 4, {"c": 5})

        assert fp(vbt.Param([1]))[0] == [(1, (), 2, {})]
        assert_index_equal(fp(vbt.Param([1]))[1], pd.Index([1], dtype="int64", name="a"))
        assert fp(vbt.Param([1, 2]))[0] == [(1, (), 2, {}), (2, (), 2, {})]
        assert_index_equal(fp(vbt.Param([1, 2]))[1], pd.Index([1, 2], dtype="int64", name="a"))
        assert fp(1, vbt.Param([2, 3]))[0] == [(1, (2,), 2, {}), (1, (3,), 2, {})]
        assert_index_equal(fp(1, vbt.Param([2, 3]))[1], pd.Index([2, 3], dtype="int64", name="my_args_0"))
        assert fp(1, b=vbt.Param([2, 3]))[0] == [(1, (), 2, {}), (1, (), 3, {})]
        assert_index_equal(fp(1, b=vbt.Param([2, 3]))[1], pd.Index([2, 3], dtype="int64", name="b"))
        assert fp(1, c=vbt.Param([2, 3]))[0] == [(1, (), 2, {"c": 2}), (1, (), 2, {"c": 3})]
        assert_index_equal(fp(1, c=vbt.Param([2, 3]))[1], pd.Index([2, 3], dtype="int64", name="c"))
        kwargs = dict(c=dict(d=(2, dict(e=vbt.Param([3, 4])))), f=(5, vbt.Param([6, 7])))
        assert fp(1, **kwargs)[0] == [
            (1, (), 2, {"c": dict(d=(2, dict(e=3))), "f": (5, 6)}),
            (1, (), 2, {"c": dict(d=(2, dict(e=3))), "f": (5, 7)}),
            (1, (), 2, {"c": dict(d=(2, dict(e=4))), "f": (5, 6)}),
            (1, (), 2, {"c": dict(d=(2, dict(e=4))), "f": (5, 7)}),
        ]
        assert_index_equal(
            fp(1, **kwargs)[1],
            pd.MultiIndex.from_tuples(
                [
                    (3, 6),
                    (3, 7),
                    (4, 6),
                    (4, 7),
                ],
                names=["c_d_1_e", "f_1"],
            ),
        )

        assert fp(1, **kwargs, _selection=1) == (1, (), 2, {"c": dict(d=(2, dict(e=3))), "f": (5, 7)})
        assert fp(1, **kwargs, _selection=1, _skip_single_comb=False)[0] == [
            (1, (), 2, {"c": dict(d=(2, dict(e=3))), "f": (5, 7)}),
        ]
        assert_index_equal(
            fp(1, **kwargs, _selection=1, _skip_single_comb=False)[1],
            pd.MultiIndex.from_tuples(
                [
                    (3, 7),
                ],
                names=["c_d_1_e", "f_1"],
            ),
        )
        assert fp(1, **kwargs, _selection=(3, 7)) == (1, (), 2, {"c": dict(d=(2, dict(e=3))), "f": (5, 7)})
        assert fp(1, **kwargs, _selection=(3, 7), _skip_single_comb=False)[0] == [
            (1, (), 2, {"c": dict(d=(2, dict(e=3))), "f": (5, 7)}),
        ]
        assert_index_equal(
            fp(1, **kwargs, _selection=(3, 7), _skip_single_comb=False)[1],
            pd.MultiIndex.from_tuples(
                [
                    (3, 7),
                ],
                names=["c_d_1_e", "f_1"],
            ),
        )
        assert fp(1, **kwargs, _selection=[1])[0] == [
            (1, (), 2, {"c": dict(d=(2, dict(e=3))), "f": (5, 7)}),
        ]
        assert_index_equal(
            fp(1, **kwargs, _selection=[1])[1],
            pd.MultiIndex.from_tuples(
                [
                    (3, 7),
                ],
                names=["c_d_1_e", "f_1"],
            ),
        )
        assert fp(1, **kwargs, _selection=[1, (4, 7)])[0] == [
            (1, (), 2, {"c": dict(d=(2, dict(e=3))), "f": (5, 7)}),
            (1, (), 2, {"c": dict(d=(2, dict(e=4))), "f": (5, 7)}),
        ]
        assert_index_equal(
            fp(1, **kwargs, _selection=[1, (4, 7)])[1],
            pd.MultiIndex.from_tuples(
                [
                    (3, 7),
                    (4, 7),
                ],
                names=["c_d_1_e", "f_1"],
            ),
        )
        assert fp(1, **kwargs, _selection=vbt.RepFunc(lambda param_index: param_index[[1, 3]]))[0] == [
            (1, (), 2, {"c": dict(d=(2, dict(e=3))), "f": (5, 7)}),
            (1, (), 2, {"c": dict(d=(2, dict(e=4))), "f": (5, 7)}),
        ]
        assert_index_equal(
            fp(1, **kwargs, _selection=vbt.RepFunc(lambda param_index: param_index[[1, 3]]))[1],
            pd.MultiIndex.from_tuples(
                [
                    (3, 7),
                    (4, 7),
                ],
                names=["c_d_1_e", "f_1"],
            ),
        )

        param_configs = [dict(a=1)]
        assert fp(param_configs=param_configs)[0] == [(1, (), 2, {})]
        assert fp(param_configs=param_configs)[1] is None
        param_configs = [dict(a=1, my_args=(2, 3))]
        assert fp(param_configs=param_configs)[0] == [(1, (2, 3), 2, {})]
        assert fp(param_configs=param_configs)[1] is None
        param_configs = [dict(a=1, my_args_0=2, my_args_1=3)]
        assert fp(param_configs=param_configs)[0] == [(1, (2, 3), 2, {})]
        assert fp(param_configs=param_configs)[1] is None
        param_configs = [dict(a=1, b=3)]
        assert fp(param_configs=param_configs)[0] == [(1, (), 3, {})]
        assert fp(param_configs=param_configs)[1] is None
        param_configs = [dict(a=1, my_kwargs=dict(c=3))]
        assert fp(param_configs=param_configs)[0] == [(1, (), 2, {"c": 3})]
        assert fp(param_configs=param_configs)[1] is None
        param_configs = [dict(a=1, c=3)]
        assert fp(param_configs=param_configs)[0] == [(1, (), 2, {"c": 3})]
        assert fp(param_configs=param_configs)[1] is None
        param_configs = [dict(a=2, my_args=(3, 4)), dict(b=5, my_kwargs=dict(c=6))]
        assert fp(1, 1, 1, param_configs=param_configs)[0] == [(2, (3, 4), 2, {}), (1, (1, 1), 5, {"c": 6})]
        assert_index_equal(
            fp(1, 1, 1, param_configs=param_configs)[1],
            pd.Index([0, 1], dtype="int64", name="param_config"),
        )

        param_configs = [dict(b=3)]
        assert fp(vbt.Param([2]), param_configs=param_configs)[0] == [
            (2, (), 3, {}),
        ]
        assert_index_equal(
            fp(vbt.Param([2]), param_configs=param_configs)[1],
            pd.Index([2], dtype="int64", name="a"),
        )
        param_configs = [dict(b=3, _name="my_config")]
        assert fp(vbt.Param([2]), param_configs=param_configs)[0] == [
            (2, (), 3, {}),
        ]
        assert_index_equal(
            fp(vbt.Param([2]), param_configs=param_configs)[1],
            pd.MultiIndex.from_tuples(
                [
                    (2, "my_config"),
                ],
                names=["a", "param_config"],
            ),
        )
        param_configs = [dict(b=3), dict(b=4)]
        assert fp(vbt.Param([1, 2]), param_configs=param_configs)[0] == [
            (1, (), 3, {}),
            (1, (), 4, {}),
            (2, (), 3, {}),
            (2, (), 4, {}),
        ]
        assert_index_equal(
            fp(vbt.Param([1, 2]), param_configs=param_configs)[1],
            pd.MultiIndex.from_tuples([(1, 0), (1, 1), (2, 0), (2, 1)], names=["a", "param_config"]),
        )
        assert fp(vbt.Param([1, 2]), param_configs=param_configs, _mono_chunk_len=2)[0] == [
            ([1, 1], (), [3, 4], {}),
            ([2, 2], (), [3, 4], {}),
        ]
        assert_index_equal(
            fp(vbt.Param([1, 2]), param_configs=param_configs, _mono_chunk_len=2)[1][0].index,
            pd.MultiIndex.from_tuples([(1, 0), (1, 1)], names=["a", "param_config"]),
        )
        assert_index_equal(
            fp(vbt.Param([1, 2]), param_configs=param_configs, _mono_chunk_len=2)[1][1].index,
            pd.MultiIndex.from_tuples([(2, 0), (2, 1)], names=["a", "param_config"]),
        )
        assert fp(vbt.Param([1, 2], mono_reduce=True), param_configs=param_configs, _mono_chunk_len=2)[0] == [
            (1, (), [3, 4], {}),
            (2, (), [3, 4], {}),
        ]
        assert_index_equal(
            fp(vbt.Param([1, 2]), param_configs=param_configs, _mono_chunk_len=2)[1][0].index,
            pd.MultiIndex.from_tuples([(1, 0), (1, 1)], names=["a", "param_config"]),
        )
        assert_index_equal(
            fp(vbt.Param([1, 2]), param_configs=param_configs, _mono_chunk_len=2)[1][1].index,
            pd.MultiIndex.from_tuples([(2, 0), (2, 1)], names=["a", "param_config"]),
        )
        assert fp(vbt.Param([1, 2]), hello="world", param_configs=param_configs, _mono_chunk_len=2)[0] == [
            ([1, 1], (), [3, 4], {"hello": "world"}),
            ([2, 2], (), [3, 4], {"hello": "world"}),
        ]
        assert_index_equal(
            fp(vbt.Param([1, 2]), param_configs=param_configs, _mono_chunk_len=2)[1][0].index,
            pd.MultiIndex.from_tuples([(1, 0), (1, 1)], names=["a", "param_config"]),
        )
        assert_index_equal(
            fp(vbt.Param([1, 2]), param_configs=param_configs, _mono_chunk_len=2)[1][1].index,
            pd.MultiIndex.from_tuples([(2, 0), (2, 1)], names=["a", "param_config"]),
        )
        assert fp(
            vbt.Param([1, 2]),
            hello="world",
            param_configs=param_configs,
            _mono_chunk_len=2,
            _mono_reduce=dict(hello=False),
        )[0] == [
            ([1, 1], (), [3, 4], {"hello": ["world", "world"]}),
            ([2, 2], (), [3, 4], {"hello": ["world", "world"]}),
        ]
        assert_index_equal(
            fp(vbt.Param([1, 2]), param_configs=param_configs, _mono_chunk_len=2)[1][0].index,
            pd.MultiIndex.from_tuples([(1, 0), (1, 1)], names=["a", "param_config"]),
        )
        assert_index_equal(
            fp(vbt.Param([1, 2]), param_configs=param_configs, _mono_chunk_len=2)[1][1].index,
            pd.MultiIndex.from_tuples([(2, 0), (2, 1)], names=["a", "param_config"]),
        )
        assert fp(
            vbt.Param([1, 2]),
            hello="world",
            param_configs=param_configs,
            _mono_chunk_len=2,
            _mono_reduce=True,
        )[0] == [
            (1, (), [3, 4], {"hello": "world"}),
            (2, (), [3, 4], {"hello": "world"}),
        ]
        assert_index_equal(
            fp(vbt.Param([1, 2]), param_configs=param_configs, _mono_chunk_len=2)[1][0].index,
            pd.MultiIndex.from_tuples([(1, 0), (1, 1)], names=["a", "param_config"]),
        )
        assert_index_equal(
            fp(vbt.Param([1, 2]), param_configs=param_configs, _mono_chunk_len=2)[1][1].index,
            pd.MultiIndex.from_tuples([(2, 0), (2, 1)], names=["a", "param_config"]),
        )
        assert fp(
            vbt.Param([1, 2], mono_merge_func=sum),
            param_configs=param_configs,
            _mono_chunk_len=2,
            _mono_merge_func=dict(b=sum),
        )[0] == [
            (2, (), 7, {}),
            (4, (), 7, {}),
        ]
        assert_index_equal(
            fp(vbt.Param([1, 2]), param_configs=param_configs, _mono_chunk_len=2)[1][0].index,
            pd.MultiIndex.from_tuples([(1, 0), (1, 1)], names=["a", "param_config"]),
        )
        assert_index_equal(
            fp(vbt.Param([1, 2]), param_configs=param_configs, _mono_chunk_len=2)[1][1].index,
            pd.MultiIndex.from_tuples([(2, 0), (2, 1)], names=["a", "param_config"]),
        )
        assert (
            fp(
                vbt.Param([1, 2]),
                param_configs=param_configs,
                _mono_chunk_len=2,
                _mono_merge_func=sum,
            )[0]
            == fp(
                vbt.Param([1, 2], mono_merge_func=sum),
                param_configs=param_configs,
                _mono_chunk_len=2,
                _mono_merge_func=dict(b=sum),
            )[0]
        )
