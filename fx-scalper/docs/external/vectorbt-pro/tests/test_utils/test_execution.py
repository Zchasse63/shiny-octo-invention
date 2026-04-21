import os

import numpy as np
import pandas as pd
import pytest

import vectorbtpro as vbt
from tests.utils import *
from vectorbtpro._dtypes import *
from vectorbtpro.utils import chunking, execution

pathos_available = True
try:
    import pathos
except:
    pathos_available = False

mpire_available = False
# try:
#     import mpire
# except:
#     mpire_available = False

dask_available = True
try:
    import dask
except:
    dask_available = False

ray_available = False
# try:
#     import ray
# except:
#     ray_available = False

requires_ray = pytest.mark.skipif(not ray_available, reason="ray not installed")


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True
    if dask_available:
        dask.config.set(scheduler="synchronous")


def teardown_module():
    vbt.settings.reset()
    if ray_available:
        ray.shutdown()


def execute_func(a, *args, b=None, **kwargs):
    return a + sum(args) + b + sum(kwargs.values())


class TestExecution:
    @requires_ray
    def test_get_ray_refs(self):
        def f1(*args, **kwargs):
            pass

        def f2(*args, **kwargs):
            pass

        lst1 = [1, 2, 3]
        lst2 = [1, 2, 3]
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([1, 2, 3])
        tasks = [
            (f1, (1, lst1, arr1), dict(a=1, b=lst1, c=arr1)),
            (f1, (2, lst2, arr2), dict(a=2, b=lst2, c=arr2)),
            (f2, (1, lst1, arr1), dict(a=1, b=lst1, c=arr1)),
        ]

        task_refs = execution.RayEngine.get_ray_refs(tasks, reuse_refs=False)
        func_refs = list(zip(*task_refs))[0]
        assert func_refs[0] is not func_refs[1]
        assert func_refs[0] is not func_refs[2]
        args_refs = list(zip(*task_refs))[1]
        assert args_refs[0][0] is not args_refs[1][0]
        assert args_refs[0][0] is not args_refs[2][0]
        assert args_refs[0][1] is not args_refs[1][1]
        assert args_refs[0][1] is not args_refs[2][1]
        assert args_refs[0][2] is not args_refs[1][2]
        assert args_refs[0][2] is not args_refs[2][2]
        kwargs_refs = list(zip(*task_refs))[2]
        assert kwargs_refs[0]["a"] is not kwargs_refs[1]["a"]
        assert kwargs_refs[0]["a"] is not kwargs_refs[2]["a"]
        assert kwargs_refs[0]["b"] is not kwargs_refs[1]["b"]
        assert kwargs_refs[0]["b"] is not kwargs_refs[2]["b"]
        assert kwargs_refs[0]["c"] is not kwargs_refs[1]["c"]
        assert kwargs_refs[0]["c"] is not kwargs_refs[2]["c"]

        task_refs = execution.RayEngine.get_ray_refs(tasks, reuse_refs=True)
        func_refs = list(zip(*task_refs))[0]
        assert func_refs[0] is func_refs[1]
        assert func_refs[0] is not func_refs[2]
        args_refs = list(zip(*task_refs))[1]
        assert args_refs[0][0] is not args_refs[1][0]
        assert args_refs[0][0] is args_refs[2][0]
        assert args_refs[0][1] is not args_refs[1][1]
        assert args_refs[0][1] is args_refs[2][1]
        assert args_refs[0][2] is not args_refs[1][2]
        assert args_refs[0][2] is args_refs[2][2]
        kwargs_refs = list(zip(*task_refs))[2]
        assert kwargs_refs[0]["a"] is not kwargs_refs[1]["a"]
        assert kwargs_refs[0]["a"] is kwargs_refs[2]["a"]
        assert kwargs_refs[0]["b"] is not kwargs_refs[1]["b"]
        assert kwargs_refs[0]["b"] is kwargs_refs[2]["b"]
        assert kwargs_refs[0]["c"] is not kwargs_refs[1]["c"]
        assert kwargs_refs[0]["c"] is kwargs_refs[2]["c"]

        assert task_refs == execution.RayEngine.get_ray_refs(task_refs)

    def test_execute(self):
        tasks = [
            (execute_func, (0, 1, 2), dict(b=3, c=4)),
            (execute_func, (5, 6, 7), dict(b=8, c=9)),
            (execute_func, (10, 11, 12), dict(b=13, c=14)),
        ]
        assert execution.execute(tasks, show_progress=True) == [10, 35, 60]
        assert execution.execute(tasks, engine="serial", show_progress=True) == [10, 35, 60]
        assert execution.execute(tasks, engine=execution.SerialEngine, show_progress=True) == [10, 35, 60]
        assert execution.execute(tasks, engine=execution.SerialEngine(show_progress=True)) == [10, 35, 60]
        assert execution.execute(
            tasks,
            engine=lambda tasks, my_arg: [func(*args, **kwargs) * my_arg for func, args, kwargs in tasks],
            my_arg=100,
        ) == [1000, 3500, 6000]
        with pytest.raises(Exception):
            execution.execute(tasks, engine=object)
        if dask_available:
            assert execution.execute(tasks, engine="dask") == [10, 35, 60]
        if ray_available:
            assert execution.execute(tasks, engine="ray") == [10, 35, 60]
        assert execution.execute(tasks, engine="threadpool") == [10, 35, 60]
        assert execution.execute(tasks, engine="processpool") == [10, 35, 60]
        if mpire_available:
            assert execution.execute(tasks, engine="mpire") == [10, 35, 60]
        if pathos_available:
            assert execution.execute(tasks, engine="pathos", pool_type="thread") == [10, 35, 60]
            assert execution.execute(tasks, engine="pathos", pool_type="process") == [10, 35, 60]
            assert execution.execute(tasks, engine="pathos", pool_type="parallel") == [10, 35, 60]

    def test_execute_chunks(self):
        def f(a, *args, b=None, **kwargs):
            return a + sum(args) + b + sum(kwargs.values())

        tasks = [
            (f, (0, 1, 2), dict(b=3, c=4)),
            (f, (5, 6, 7), dict(b=8, c=9)),
            (f, (10, 11, 12), dict(b=13, c=14)),
        ]

        assert execution.execute(
            tasks,
            chunk_meta=[
                chunking.ChunkMeta(uuid="", idx=0, start=0, end=1, indices=None),
                chunking.ChunkMeta(uuid="", idx=1, start=1, end=3, indices=None),
            ],
        ) == [10, 35, 60]
        assert execution.execute(
            tasks,
            chunk_meta=[
                chunking.ChunkMeta(uuid="", idx=1, start=1, end=3, indices=None),
                chunking.ChunkMeta(uuid="", idx=0, start=0, end=1, indices=None),
            ],
        ) == [10, 35, 60]
        assert execution.execute(
            tasks,
            chunk_meta=[
                chunking.ChunkMeta(uuid="", idx=1, start=1, end=3, indices=None),
                chunking.ChunkMeta(uuid="", idx=0, start=0, end=1, indices=None),
            ],
            in_chunk_order=True,
        ) == [35, 60, 10]
        assert execution.execute(
            tasks,
            chunk_meta=[
                chunking.ChunkMeta(uuid="", idx=1, start=None, end=None, indices=[0]),
                chunking.ChunkMeta(uuid="", idx=0, start=None, end=None, indices=[1, 2]),
            ],
        ) == [10, 35, 60]
        assert execution.execute(
            tasks,
            chunk_meta=[
                chunking.ChunkMeta(uuid="", idx=1, start=None, end=None, indices=[2, 1]),
                chunking.ChunkMeta(uuid="", idx=0, start=None, end=None, indices=[0]),
            ],
            in_chunk_order=True,
        ) == [60, 35, 10]

        assert execution.execute(
            iter(tasks),
            chunk_meta=[
                chunking.ChunkMeta(uuid="", idx=0, start=0, end=1, indices=None),
                chunking.ChunkMeta(uuid="", idx=1, start=1, end=3, indices=None),
            ],
        ) == [10, 35, 60]
        assert execution.execute(
            iter(tasks),
            chunk_meta=[
                chunking.ChunkMeta(uuid="", idx=1, start=1, end=3, indices=None),
                chunking.ChunkMeta(uuid="", idx=0, start=0, end=1, indices=None),
            ],
        ) == [10, 35, 60]
        assert execution.execute(
            iter(tasks),
            chunk_meta=[
                chunking.ChunkMeta(uuid="", idx=1, start=1, end=3, indices=None),
                chunking.ChunkMeta(uuid="", idx=0, start=0, end=1, indices=None),
            ],
            in_chunk_order=True,
        ) == [35, 60, 10]
        assert execution.execute(
            iter(tasks),
            chunk_meta=[
                chunking.ChunkMeta(uuid="", idx=1, start=None, end=None, indices=[0]),
                chunking.ChunkMeta(uuid="", idx=0, start=None, end=None, indices=[1, 2]),
            ],
        ) == [10, 35, 60]
        assert execution.execute(
            iter(tasks),
            chunk_meta=[
                chunking.ChunkMeta(uuid="", idx=1, start=None, end=None, indices=[2, 1]),
                chunking.ChunkMeta(uuid="", idx=0, start=None, end=None, indices=[0]),
            ],
            in_chunk_order=True,
        ) == [60, 35, 10]

        assert execution.execute(iter(tasks), n_chunks=1) == [10, 35, 60]
        assert execution.execute(iter(tasks), n_chunks=2) == [10, 35, 60]
        assert execution.execute(iter(tasks), n_chunks=3) == [10, 35, 60]
        assert execution.execute(iter(tasks), chunk_len=1) == [10, 35, 60]
        assert execution.execute(iter(tasks), chunk_len=2) == [10, 35, 60]
        assert execution.execute(iter(tasks), chunk_len=3) == [10, 35, 60]
        assert execution.execute(iter(tasks), chunk_len="auto") == [10, 35, 60]

        pre_execute_arg_lst = []
        pre_chunk_idx_lst = []
        pre_call_indices_lst = []
        pre_chunk_arg_lst = []
        post_chunk_idx_lst = []
        post_call_indices_lst = []
        post_call_results_lst = []
        post_chunk_arg_lst = []
        post_results_lst = []
        post_execute_arg_lst = []

        def pre_execute_func(pre_execute_arg):
            pre_execute_arg_lst.append(pre_execute_arg)

        def pre_chunk_func(chunk_idx, call_indices, pre_chunk_arg):
            pre_chunk_idx_lst.append(chunk_idx)
            pre_call_indices_lst.append(call_indices)
            pre_chunk_arg_lst.append(pre_chunk_arg)

        def post_chunk_func(chunk_idx, call_indices, call_results, post_chunk_arg):
            post_chunk_idx_lst.append(chunk_idx)
            post_call_indices_lst.append(call_indices)
            post_call_results_lst.append(call_results)
            post_chunk_arg_lst.append(post_chunk_arg)

        def post_execute_func(results, post_execute_arg):
            post_results_lst.append(results)
            post_execute_arg_lst.append(post_execute_arg)
            return [result + 1 for result in results]

        results = execution.execute(
            [(lambda _i=i: _i, (), {}) for i in range(10)],
            chunk_len=2,
            pre_execute_func=pre_execute_func,
            pre_execute_kwargs=dict(
                pre_execute_arg=100,
            ),
            pre_chunk_func=pre_chunk_func,
            pre_chunk_kwargs=dict(
                chunk_idx=vbt.Rep("chunk_idx"),
                call_indices=vbt.Rep("call_indices"),
                pre_chunk_arg=101,
            ),
            post_chunk_func=post_chunk_func,
            post_chunk_kwargs=dict(
                chunk_idx=vbt.Rep("chunk_idx"),
                call_indices=vbt.Rep("call_indices"),
                call_results=vbt.Rep("call_results"),
                post_chunk_arg=102,
            ),
            post_execute_func=post_execute_func,
            post_execute_kwargs=dict(
                results=vbt.Rep("results"),
                post_execute_arg=103,
            ),
        )
        assert pre_execute_arg_lst == [100]
        assert pre_chunk_idx_lst == [0, 1, 2, 3, 4]
        assert pre_call_indices_lst == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
        assert pre_chunk_arg_lst == [101, 101, 101, 101, 101]
        assert post_chunk_idx_lst == [0, 1, 2, 3, 4]
        assert post_call_indices_lst == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
        assert post_call_results_lst == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
        assert post_chunk_arg_lst == [102, 102, 102, 102, 102]
        assert post_results_lst == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
        assert post_execute_arg_lst == [103]
        assert results == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def test_iterated(self):
        def f(a, *args, b=None, **kwargs):
            return a + sum(args) + b + sum(kwargs.values())

        f_iterated = vbt.iterated(f)
        assert f_iterated([0, 1, 2], 3, b=4, c=5) == [12, 13, 14]
        f_iterated = vbt.iterated(f, merge_func="concat")
        assert_series_equal(
            f_iterated([0, 1, 2], 3, b=4, c=5),
            pd.Series([12, 13, 14], index=pd.Index([0, 1, 2], name="a")),
        )
        assert_series_equal(
            f_iterated(3, 3, b=4, c=5),
            pd.Series([12, 13, 14], index=pd.Index([0, 1, 2], name="a")),
        )
        assert_series_equal(
            f_iterated(dict(a0=0, a1=1, a2=2), 3, b=4, c=5),
            pd.Series([12, 13, 14], index=pd.Index(["a0", "a1", "a2"], name="a")),
        )
        assert_series_equal(
            f_iterated(pd.Index([0, 1, 2]), 3, b=4, c=5),
            pd.Series([12, 13, 14], index=pd.Index([0, 1, 2], name="a")),
        )
        assert_series_equal(
            f_iterated(pd.Index([0, 1, 2], name="custom_a"), 3, b=4, c=5),
            pd.Series([12, 13, 14], index=pd.Index([0, 1, 2], name="custom_a")),
        )
        assert_series_equal(
            f_iterated(pd.Series([0, 1, 2], index=pd.Index(["a0", "a1", "a2"], name="a")), 3, b=4, c=5),
            pd.Series([12, 13, 14], index=pd.Index(["a0", "a1", "a2"], name="a")),
        )
        np.testing.assert_array_equal(
            f_iterated(iter([0, 1, 2]), 3, b=4, c=5),
            np.array([12, 13, 14]),
        )

        @vbt.iterated
        def f_iterated2(a, *args, b=None, **kwargs):
            return vbt.NoResult

        assert f_iterated2([0, 1, 2], 3, b=4, c=5) == [vbt.NoResult, vbt.NoResult, vbt.NoResult]
        with pytest.raises(vbt.NoResultsException):
            f_iterated2([0, 1, 2], 3, b=4, c=5, _filter_results=True)
        assert f_iterated2([0, 1, 2], 3, b=4, c=5, _filter_results=True, _raise_no_results=False) == vbt.NoResult
