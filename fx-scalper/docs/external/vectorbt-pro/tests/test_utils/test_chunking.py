import os

import pytest

import vectorbtpro as vbt
from tests.utils import *
from vectorbtpro._dtypes import *
from vectorbtpro.utils import chunking, parsing, template


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True


def teardown_module():
    vbt.settings.reset()


class TestChunking:
    def test_arg_getter_mixin(self):
        def f(a, *args, b=None, **kwargs):
            pass

        ann_args = parsing.annotate_args(f, (0, 1), dict(c=2))

        assert chunking.ArgGetter(0).get_arg(ann_args) == 0
        assert chunking.ArgGetter("a").get_arg(ann_args) == 0
        assert chunking.ArgGetter(1).get_arg(ann_args) == 1
        assert chunking.ArgGetter(2).get_arg(ann_args) is None
        assert chunking.ArgGetter("b").get_arg(ann_args) is None
        assert chunking.ArgGetter(3).get_arg(ann_args) == 2
        assert chunking.ArgGetter("c").get_arg(ann_args) == 2
        with pytest.raises(Exception):
            chunking.ArgGetter(4).get_arg(ann_args)
        with pytest.raises(Exception):
            chunking.ArgGetter("d").get_arg(ann_args)

    def test_sizers(self):
        def f(a):
            pass

        ann_args = parsing.annotate_args(f, (10,), {})
        assert chunking.ArgSizer(arg_query="a").apply(ann_args) == 10

        ann_args = parsing.annotate_args(f, ([1, 2, 3],), {})
        assert chunking.LenSizer(arg_query="a").apply(ann_args) == 3

        ann_args = parsing.annotate_args(f, (3,), {})
        assert chunking.LenSizer(arg_query="a", single_type=int).apply(ann_args) == 1
        with pytest.raises(Exception):
            chunking.LenSizer().apply(ann_args)
        with pytest.raises(Exception):
            chunking.LenSizer(arg_query="a").apply(ann_args)

        ann_args = parsing.annotate_args(f, ((2, 3),), {})
        with pytest.raises(Exception):
            chunking.ShapeSizer().apply(ann_args)
        with pytest.raises(Exception):
            chunking.ShapeSizer(arg_query="a").apply(ann_args)
        assert chunking.ShapeSizer(arg_query="a", axis=0).apply(ann_args) == 2
        assert chunking.ShapeSizer(arg_query="a", axis=1).apply(ann_args) == 3
        assert chunking.ShapeSizer(arg_query="a", axis=2).apply(ann_args) == 0

        ann_args = parsing.annotate_args(f, (np.empty((2, 3)),), {})
        with pytest.raises(Exception):
            chunking.ArraySizer().apply(ann_args)
        with pytest.raises(Exception):
            chunking.ArraySizer(arg_query="a").apply(ann_args)
        assert chunking.ArraySizer(arg_query="a", axis=0).apply(ann_args) == 2
        assert chunking.ArraySizer(arg_query="a", axis=1).apply(ann_args) == 3
        assert chunking.ArraySizer(arg_query="a", axis=2).apply(ann_args) == 0

    def test_iter_chunk_meta(self):
        with pytest.raises(Exception):
            list(vbt.iter_chunk_meta(n_chunks=0))

        chunk_meta_equal(
            list(vbt.iter_chunk_meta(n_chunks=4)),
            [
                chunking.ChunkMeta(uuid="", idx=0, start=None, end=None, indices=None),
                chunking.ChunkMeta(uuid="", idx=1, start=None, end=None, indices=None),
                chunking.ChunkMeta(uuid="", idx=2, start=None, end=None, indices=None),
                chunking.ChunkMeta(uuid="", idx=3, start=None, end=None, indices=None),
            ],
        )
        chunk_meta_equal(
            list(vbt.iter_chunk_meta(n_chunks=1, size=4)),
            [chunking.ChunkMeta(uuid="", idx=0, start=0, end=4, indices=None)],
        )
        chunk_meta_equal(
            list(vbt.iter_chunk_meta(n_chunks=2, size=4)),
            [
                chunking.ChunkMeta(uuid="", idx=0, start=0, end=2, indices=None),
                chunking.ChunkMeta(uuid="", idx=1, start=2, end=4, indices=None),
            ],
        )
        chunk_meta_equal(
            list(vbt.iter_chunk_meta(n_chunks=3, size=4)),
            [
                chunking.ChunkMeta(uuid="", idx=0, start=0, end=2, indices=None),
                chunking.ChunkMeta(uuid="", idx=1, start=2, end=3, indices=None),
                chunking.ChunkMeta(uuid="", idx=2, start=3, end=4, indices=None),
            ],
        )
        chunk_meta_equal(
            list(vbt.iter_chunk_meta(n_chunks=4, size=4)),
            [
                chunking.ChunkMeta(uuid="", idx=0, start=0, end=1, indices=None),
                chunking.ChunkMeta(uuid="", idx=1, start=1, end=2, indices=None),
                chunking.ChunkMeta(uuid="", idx=2, start=2, end=3, indices=None),
                chunking.ChunkMeta(uuid="", idx=3, start=3, end=4, indices=None),
            ],
        )
        chunk_meta_equal(
            list(vbt.iter_chunk_meta(n_chunks=5, size=4)),
            [
                chunking.ChunkMeta(uuid="", idx=0, start=0, end=1, indices=None),
                chunking.ChunkMeta(uuid="", idx=1, start=1, end=2, indices=None),
                chunking.ChunkMeta(uuid="", idx=2, start=2, end=3, indices=None),
                chunking.ChunkMeta(uuid="", idx=3, start=3, end=4, indices=None),
            ],
        )
        with pytest.raises(Exception):
            list(vbt.iter_chunk_meta(chunk_len=0, size=4))
        chunk_meta_equal(
            list(vbt.iter_chunk_meta(chunk_len=1, size=4)),
            [
                chunking.ChunkMeta(uuid="", idx=0, start=0, end=1, indices=None),
                chunking.ChunkMeta(uuid="", idx=1, start=1, end=2, indices=None),
                chunking.ChunkMeta(uuid="", idx=2, start=2, end=3, indices=None),
                chunking.ChunkMeta(uuid="", idx=3, start=3, end=4, indices=None),
            ],
        )
        chunk_meta_equal(
            list(vbt.iter_chunk_meta(chunk_len=2, size=4)),
            [
                chunking.ChunkMeta(uuid="", idx=0, start=0, end=2, indices=None),
                chunking.ChunkMeta(uuid="", idx=1, start=2, end=4, indices=None),
            ],
        )
        chunk_meta_equal(
            list(vbt.iter_chunk_meta(chunk_len=3, size=4)),
            [
                chunking.ChunkMeta(uuid="", idx=0, start=0, end=3, indices=None),
                chunking.ChunkMeta(uuid="", idx=1, start=3, end=4, indices=None),
            ],
        )
        chunk_meta_equal(
            list(vbt.iter_chunk_meta(chunk_len=4, size=4)),
            [chunking.ChunkMeta(uuid="", idx=0, start=0, end=4, indices=None)],
        )
        chunk_meta_equal(
            list(vbt.iter_chunk_meta(chunk_len=5, size=4)),
            [chunking.ChunkMeta(uuid="", idx=0, start=0, end=4, indices=None)],
        )
        chunk_meta_equal(
            list(vbt.iter_chunk_meta(n_chunks=2, size=2, min_size=2)),
            [
                chunking.ChunkMeta(uuid="", idx=0, start=0, end=1, indices=None),
                chunking.ChunkMeta(uuid="", idx=1, start=1, end=2, indices=None),
            ],
        )
        chunk_meta_equal(
            list(vbt.iter_chunk_meta(n_chunks=2, size=2, min_size=3)),
            [chunking.ChunkMeta(uuid="", idx=0, start=0, end=2, indices=None)],
        )
        with pytest.raises(Exception):
            list(vbt.iter_chunk_meta(n_chunks=2, size=4, chunk_len=2))

    def test_chunk_meta_generators(self):
        def f(a):
            pass

        chunk_meta = [
            chunking.ChunkMeta(uuid="", idx=0, start=0, end=1, indices=None),
            chunking.ChunkMeta(uuid="", idx=1, start=1, end=3, indices=None),
            chunking.ChunkMeta(uuid="", idx=2, start=3, end=6, indices=None),
        ]
        ann_args = parsing.annotate_args(f, (chunk_meta,), {})
        chunk_meta_equal(list(chunking.ArgChunkMeta("a").get_chunk_meta(ann_args)), chunk_meta)

        ann_args = parsing.annotate_args(f, ([1, 2, 3],), {})
        chunk_meta_equal(list(chunking.LenChunkMeta("a").get_chunk_meta(ann_args)), chunk_meta)

    def test_get_chunk_meta_from_args(self):
        def f(a, *args, b=None, **kwargs):
            pass

        chunk_meta = [
            chunking.ChunkMeta(uuid="", idx=0, start=0, end=1, indices=None),
            chunking.ChunkMeta(uuid="", idx=1, start=1, end=2, indices=None),
            chunking.ChunkMeta(uuid="", idx=2, start=2, end=3, indices=None),
        ]

        ann_args = parsing.annotate_args(f, (2, 3, 1), dict(b=[1, 2, 3]))
        chunk_meta_equal(list(vbt.Chunker.get_chunk_meta_from_args(ann_args, size=3, n_chunks=3)), chunk_meta)
        chunk_meta_equal(
            list(
                vbt.Chunker.get_chunk_meta_from_args(
                    ann_args,
                    size=3,
                    n_chunks=lambda ann_args: ann_args["args"]["value"][0],
                )
            ),
            chunk_meta,
        )
        chunk_meta_equal(
            list(vbt.Chunker.get_chunk_meta_from_args(ann_args, size=3, n_chunks=chunking.ArgSizer(arg_query=1))),
            chunk_meta,
        )
        with pytest.raises(Exception):
            list(vbt.Chunker.get_chunk_meta_from_args(ann_args, size=3, n_chunks="a"))

        chunk_meta_equal(list(vbt.Chunker.get_chunk_meta_from_args(ann_args, chunk_len=1, size=3)), chunk_meta)
        chunk_meta_equal(
            list(
                vbt.Chunker.get_chunk_meta_from_args(
                    ann_args,
                    chunk_len=1,
                    size=lambda ann_args: ann_args["args"]["value"][0],
                )
            ),
            chunk_meta,
        )
        chunk_meta_equal(
            list(vbt.Chunker.get_chunk_meta_from_args(ann_args, chunk_len=1, size=chunking.ArgSizer(arg_query=1))),
            chunk_meta,
        )
        with pytest.raises(Exception):
            list(vbt.Chunker.get_chunk_meta_from_args(ann_args, chunk_len=1, size="a"))

        chunk_meta_equal(list(vbt.Chunker.get_chunk_meta_from_args(ann_args, size=3, chunk_len=1)), chunk_meta)
        chunk_meta_equal(
            list(
                vbt.Chunker.get_chunk_meta_from_args(
                    ann_args,
                    size=3,
                    chunk_len=lambda ann_args: ann_args["args"]["value"][1],
                )
            ),
            chunk_meta,
        )
        chunk_meta_equal(
            list(vbt.Chunker.get_chunk_meta_from_args(ann_args, size=3, chunk_len=chunking.ArgSizer(arg_query=2))),
            chunk_meta,
        )
        with pytest.raises(Exception):
            list(vbt.Chunker.get_chunk_meta_from_args(ann_args, size=3, chunk_len="a"))

        chunk_meta_equal(list(vbt.Chunker.get_chunk_meta_from_args(ann_args, chunk_meta=chunk_meta)), chunk_meta)
        chunk_meta_equal(
            list(vbt.Chunker.get_chunk_meta_from_args(ann_args, chunk_meta=chunking.LenChunkMeta("b"))),
            [
                chunking.ChunkMeta(uuid="", idx=0, start=0, end=1, indices=None),
                chunking.ChunkMeta(uuid="", idx=1, start=1, end=3, indices=None),
                chunking.ChunkMeta(uuid="", idx=2, start=3, end=6, indices=None),
            ],
        )
        chunk_meta_equal(
            list(vbt.Chunker.get_chunk_meta_from_args(ann_args, chunk_meta=lambda ann_args: chunk_meta)),
            chunk_meta,
        )

    def test_take_from_args(self):
        def f(a, b, *args, c=None, d=None, **kwargs):
            pass

        lst = [0, 1, 2]

        ann_args = parsing.annotate_args(
            f,
            (lst, lst, lst, (lst, lst)),
            dict(c=lst, d=lst, e=lst, f=dict(g=lst, h=lst)),
        )
        arg_take_spec = dict(
            b=chunking.ChunkSelector(),
            args=chunking.ArgsTaker(None, chunking.SequenceTaker(cont_take_spec=(None, chunking.ChunkSlicer()))),
            d=chunking.ChunkSelector(),
            kwargs=chunking.KwargsTaker(f=chunking.MappingTaker(cont_take_spec=dict(h=chunking.ChunkSlicer()))),
        )
        args, kwargs = vbt.Chunker.take_from_args(
            ann_args,
            arg_take_spec,
            chunking.ChunkMeta(uuid="", idx=0, start=1, end=3, indices=None),
        )
        assert args == (lst, lst[0], lst, (lst, lst[1:3]))
        assert kwargs == dict(c=lst, d=lst[0], e=lst, f=dict(g=lst, h=lst[1:3]))

    def test_chunk_takers(self):
        a = np.arange(6).reshape((2, 3))
        sr = pd.Series(a[:, 0])
        df = pd.DataFrame(a)

        assert chunking.ChunkSelector().apply([1, 2, 3], chunking.ChunkMeta("", 0, 0, 1, None)) == 1
        assert chunking.ChunkSelector(keep_dims=True).apply([1, 2, 3], chunking.ChunkMeta("", 0, 0, 1, None)) == [1]
        assert chunking.ChunkSelector().apply(None, chunking.ChunkMeta("", 0, 0, 1, None)) is None
        with pytest.raises(Exception):
            chunking.ChunkSelector(ignore_none=False).apply(None, chunking.ChunkMeta("", 0, 0, 1, None))
        assert chunking.ChunkSelector(single_type=int).apply(10, chunking.ChunkMeta("", 0, 0, 1, None)) == 10
        with pytest.raises(Exception):
            chunking.ChunkSelector().apply(10, chunking.ChunkMeta("", 0, 0, 1, None))
        assert chunking.ChunkSlicer().apply([1, 2, 3], chunking.ChunkMeta("", 0, 0, 1, None)) == [1]
        np.testing.assert_array_equal(
            chunking.ChunkSlicer().apply(np.array([1, 2, 3]), chunking.ChunkMeta("", 0, None, None, np.array([0, 0]))),
            np.array([1, 1]),
        )
        with pytest.raises(Exception):
            chunking.ChunkSlicer().apply(np.array([1, 2, 3]), chunking.ChunkMeta("", 0, None, None, np.array([3])))

        assert chunking.CountAdapter().apply(10, chunking.ChunkMeta("", 0, 0, 1, None)) == 1
        assert chunking.CountAdapter().apply(10, chunking.ChunkMeta("", 0, 8, 12, None)) == 2
        assert chunking.CountAdapter().apply(10, chunking.ChunkMeta("", 0, 12, 13, None)) == 0

        assert chunking.ShapeSelector(axis=0).apply((1, 2, 3), chunking.ChunkMeta("", 0, 0, 1, None)) == (2, 3)
        assert chunking.ShapeSelector(axis=1).apply((1, 2, 3), chunking.ChunkMeta("", 0, 0, 1, None)) == (1, 3)
        assert chunking.ShapeSelector(axis=2).apply((1, 2, 3), chunking.ChunkMeta("", 0, 0, 1, None)) == (1, 2)
        with pytest.raises(Exception):
            chunking.ShapeSelector(axis=4).apply((1, 2, 3), chunking.ChunkMeta("", 0, 0, 1, None))
        assert chunking.ShapeSelector(axis=0, keep_dims=True).apply(
            (1, 2, 3),
            chunking.ChunkMeta("", 0, 0, 1, None),
        ) == (1, 2, 3)
        with pytest.raises(Exception):
            chunking.ShapeSelector(axis=0).apply((1, 2, 3), chunking.ChunkMeta("", 1, 0, 1, None))
        assert chunking.ShapeSelector(axis=0).apply((1,), chunking.ChunkMeta("", 0, 0, 1, None)) == ()
        assert chunking.ShapeSlicer(axis=0).apply((1, 2, 3), chunking.ChunkMeta("", 0, 0, 1, None)) == (1, 2, 3)
        assert chunking.ShapeSlicer(axis=1).apply((1, 2, 3), chunking.ChunkMeta("", 0, 0, 1, None)) == (1, 1, 3)
        assert chunking.ShapeSlicer(axis=2).apply((1, 2, 3), chunking.ChunkMeta("", 0, 0, 1, None)) == (1, 2, 1)
        with pytest.raises(Exception):
            chunking.ShapeSlicer(axis=4).apply((1, 2, 3), chunking.ChunkMeta("", 0, 0, 1, None))
        assert chunking.ShapeSlicer(axis=0).apply((1, 2, 3), chunking.ChunkMeta("", 0, 0, 2, None)) == (1, 2, 3)
        assert chunking.ShapeSlicer(axis=0).apply((1, 2, 3), chunking.ChunkMeta("", 0, 1, 2, None)) == (2, 3)
        assert chunking.ShapeSlicer(axis=0).apply(
            (1, 2, 3),
            chunking.ChunkMeta("", 0, None, None, np.array([0, 0])),
        ) == (2, 2, 3)
        with pytest.raises(Exception):
            chunking.ShapeSlicer(axis=0).apply((1, 2, 3), chunking.ChunkMeta("", 0, None, None, np.array([1])))

        np.testing.assert_array_equal(
            chunking.ArraySelector(axis=0).apply(a, chunking.ChunkMeta("", 0, 0, 1, None)),
            a[0],
        )
        np.testing.assert_array_equal(
            chunking.ArraySelector(axis=0, keep_dims=True).apply(a, chunking.ChunkMeta("", 0, 0, 1, None)),
            a[[0]],
        )
        np.testing.assert_array_equal(
            chunking.ArraySelector(axis=1).apply(a, chunking.ChunkMeta("", 0, 0, 1, None)),
            a[:, 0],
        )
        with pytest.raises(Exception):
            chunking.ArraySelector(axis=2).apply(a, chunking.ChunkMeta("", 0, 0, 1, None))
        assert chunking.ArraySelector(axis=0).apply(sr, chunking.ChunkMeta("", 0, 0, 1, None)) == sr.iloc[0]
        assert_series_equal(
            chunking.ArraySelector(axis=1).apply(df, chunking.ChunkMeta("", 0, 0, 1, None)),
            df.iloc[:, 0],
        )
        np.testing.assert_array_equal(
            chunking.ArraySlicer(axis=0).apply(a, chunking.ChunkMeta("", 0, 0, 1, None)),
            a[[0]],
        )
        np.testing.assert_array_equal(
            chunking.ArraySlicer(axis=1).apply(a, chunking.ChunkMeta("", 0, 0, 1, None)),
            a[:, [0]],
        )
        np.testing.assert_array_equal(
            chunking.ArraySlicer(axis=0).apply(a, chunking.ChunkMeta("", 0, None, None, np.array([0]))),
            a[[0]],
        )
        with pytest.raises(Exception):
            chunking.ArraySlicer(axis=0).apply(a, chunking.ChunkMeta("", 0, None, None, np.array([2])))
        with pytest.raises(Exception):
            chunking.ArraySlicer(axis=2).apply(a, chunking.ChunkMeta("", 0, 0, 1, None))
        assert_series_equal(
            chunking.ArraySlicer(axis=0).apply(sr, chunking.ChunkMeta("", 0, 0, 1, None)),
            sr.iloc[[0]],
        )
        assert_frame_equal(
            chunking.ArraySlicer(axis=1).apply(df, chunking.ChunkMeta("", 0, 0, 1, None)),
            df.iloc[:, [0]],
        )

    def test_iter_tasks(self):
        def f(a, *args, b=None, **kwargs):
            pass

        ann_args = parsing.annotate_args(f, (2, 3, 1), dict(b=[1, 2, 3]))
        chunk_meta = [
            chunking.ChunkMeta(uuid="", idx=0, start=0, end=1, indices=None),
            chunking.ChunkMeta(uuid="", idx=1, start=1, end=2, indices=None),
            chunking.ChunkMeta(uuid="", idx=2, start=2, end=3, indices=None),
        ]
        arg_take_spec = dict(b=chunking.ChunkSelector())
        result = [vbt.Task(f, 2, 3, 1, b=1), vbt.Task(f, 2, 3, 1, b=2), vbt.Task(f, 2, 3, 1, b=3)]
        assert list(vbt.Chunker.iter_tasks(f, ann_args, chunk_meta, arg_take_spec=arg_take_spec)) == result
        ann_args = parsing.annotate_args(
            f,
            (template.RepEval('ann_args["args"]["value"][1] + 1'), 3, 1),
            dict(b=template.Rep("lst")),
        )
        assert (
            list(
                vbt.Chunker.iter_tasks(
                    f,
                    ann_args,
                    chunk_meta,
                    arg_take_spec=arg_take_spec,
                    template_context={"lst": [1, 2, 3]},
                )
            )
            == result
        )

    def test_chunked(self):
        @chunking.chunked(n_chunks=2, size=vbt.LenSizer(arg_query="a"), arg_take_spec=dict(a=vbt.ChunkSlicer()))
        def f(a):
            return a

        results = f(np.arange(10))
        np.testing.assert_array_equal(results[0], np.arange(5))
        np.testing.assert_array_equal(results[1], np.arange(5, 10))

        f.options.n_chunks = 3
        results = f(np.arange(10))
        np.testing.assert_array_equal(results[0], np.arange(4))
        np.testing.assert_array_equal(results[1], np.arange(4, 7))
        np.testing.assert_array_equal(results[2], np.arange(7, 10))

        results = f(np.arange(10), _n_chunks=4)
        np.testing.assert_array_equal(results[0], np.arange(3))
        np.testing.assert_array_equal(results[1], np.arange(3, 6))
        np.testing.assert_array_equal(results[2], np.arange(6, 8))
        np.testing.assert_array_equal(results[3], np.arange(8, 10))

        results = f(np.arange(10), _n_chunks=1, _skip_single_chunk=False)
        np.testing.assert_array_equal(results[0], np.arange(10))

        results = f(np.arange(10), _n_chunks=1, _skip_single_chunk=True)
        np.testing.assert_array_equal(results, np.arange(10))

        @vbt.chunked(n_chunks=2, size=vbt.LenSizer(arg_query="a"))
        def f2(chunk_meta, a):
            return a[chunk_meta.start : chunk_meta.end]

        results = f2(np.arange(10))
        np.testing.assert_array_equal(results[0], np.arange(5))
        np.testing.assert_array_equal(results[1], np.arange(5, 10))

        @vbt.chunked(n_chunks=2, size=vbt.LenSizer(arg_query="a"), prepend_chunk_meta=False)
        def f3(chunk_meta, a):
            return a[chunk_meta.start : chunk_meta.end]

        results = f3(template.Rep("chunk_meta"), np.arange(10))
        np.testing.assert_array_equal(results[0], np.arange(5))
        np.testing.assert_array_equal(results[1], np.arange(5, 10))

        with pytest.raises(Exception):

            @vbt.chunked(n_chunks=2, size=vbt.LenSizer(arg_query="a"), prepend_chunk_meta=True)
            def f4(chunk_meta, a):
                return a[chunk_meta.start : chunk_meta.end]

            f4(template.Rep("chunk_meta"), np.arange(10))

        @vbt.chunked(
            n_chunks=2,
            size=lambda ann_args, **kwargs: len(ann_args["a"]["value"]),
            arg_take_spec=dict(a=vbt.ChunkSlicer()),
        )
        def f5(a):
            return a

        results = f5(np.arange(10))
        np.testing.assert_array_equal(results[0], np.arange(5))
        np.testing.assert_array_equal(results[1], np.arange(5, 10))

        def arg_take_spec(ann_args, chunk_meta, **kwargs):
            a = ann_args["a"]["value"]
            lens = ann_args["lens"]["value"]
            lens_chunk = lens[chunk_meta.start : chunk_meta.end]
            a_end = np.cumsum(lens)
            a_start = a_end - lens
            a_start = a_start[chunk_meta.start : chunk_meta.end][0]
            a_end = a_end[chunk_meta.start : chunk_meta.end][-1]
            a_chunk = a[a_start:a_end]
            return (a_chunk, lens_chunk), {}

        @vbt.chunked(
            n_chunks=2,
            size=vbt.LenSizer(arg_query="lens"),
            arg_take_spec=arg_take_spec,
            merge_func=lambda results: [list(r) for r in results],
        )
        def f6(a, lens):
            ends = np.cumsum(lens)
            starts = ends - lens
            for i in range(len(lens)):
                yield a[starts[i] : ends[i]]

        results = f6(np.arange(10), [1, 2, 3, 4])
        np.testing.assert_array_equal(results[0][0], np.arange(1))
        np.testing.assert_array_equal(results[0][1], np.arange(1, 3))
        np.testing.assert_array_equal(results[1][0], np.arange(3, 6))
        np.testing.assert_array_equal(results[1][1], np.arange(6, 10))

        def produce_out(*args, **kwargs):
            out = None
            for v in args:
                if out is None:
                    out = v
                else:
                    out += v
            for k, v in kwargs.items():
                if out is None:
                    out = v
                else:
                    out += v
            return out

        @vbt.chunked
        def f7(*args: vbt.ArraySlicer, **kwargs: vbt.ArraySlicer):
            return produce_out(*args, **kwargs)

        results = f7(np.arange(0, 10), np.arange(10, 20), a=np.arange(20, 30), b=np.arange(30, 40), _n_chunks=2)
        np.testing.assert_array_equal(results[0], np.array([60, 64, 68, 72, 76]))
        np.testing.assert_array_equal(results[1], np.array([80, 84, 88, 92, 96]))

        @vbt.chunked
        def f8(
            *args: vbt.ArgsTaker(vbt.ArraySlicer, vbt.ArraySlicer),
            **kwargs: vbt.KwargsTaker(a=vbt.ArraySlicer, b=vbt.ArraySlicer),
        ):
            return produce_out(*args, **kwargs)

        results = f8(np.arange(0, 10), np.arange(10, 20), a=np.arange(20, 30), b=np.arange(30, 40), _n_chunks=2)
        np.testing.assert_array_equal(results[0], np.array([60, 64, 68, 72, 76]))
        np.testing.assert_array_equal(results[1], np.array([80, 84, 88, 92, 96]))

        @vbt.chunked
        def f9(
            *args: vbt.VarArgs(vbt.ArraySlicer, vbt.ArraySlicer),
            **kwargs: vbt.VarKwargs(a=vbt.ArraySlicer, b=vbt.ArraySlicer),
        ):
            return produce_out(*args, **kwargs)

        results = f9(np.arange(0, 10), np.arange(10, 20), a=np.arange(20, 30), b=np.arange(30, 40), _n_chunks=2)
        np.testing.assert_array_equal(results[0], np.array([60, 64, 68, 72, 76]))
        np.testing.assert_array_equal(results[1], np.array([80, 84, 88, 92, 96]))

        @vbt.chunked
        def f10(*args, **kwargs):
            return produce_out(*args, **kwargs)

        results = f10(
            vbt.Chunked(np.arange(0, 10), vbt.ArraySlicer),
            vbt.Chunked(np.arange(10, 20), vbt.ArraySlicer),
            a=vbt.Chunked(np.arange(20, 30), vbt.ArraySlicer),
            b=vbt.Chunked(np.arange(30, 40), vbt.ArraySlicer),
            _n_chunks=2,
        )
        np.testing.assert_array_equal(results[0], np.array([60, 64, 68, 72, 76]))
        np.testing.assert_array_equal(results[1], np.array([80, 84, 88, 92, 96]))

        @vbt.chunked
        def f11(*args: vbt.ChunkedArray, **kwargs: vbt.ChunkedArray):
            return produce_out(*args, **kwargs)

        results = f11(np.arange(0, 10), np.arange(10, 20), a=np.arange(20, 30), b=np.arange(30, 40), _n_chunks=2)
        np.testing.assert_array_equal(results[0], np.array([60, 64, 68, 72, 76]))
        np.testing.assert_array_equal(results[1], np.array([80, 84, 88, 92, 96]))

        @vbt.chunked
        def f12(*args, **kwargs):
            return produce_out(*args, **kwargs)

        results = f12(
            vbt.ChunkedArray(np.arange(0, 10)),
            vbt.ChunkedArray(np.arange(10, 20)),
            a=vbt.ChunkedArray(np.arange(20, 30)),
            b=vbt.ChunkedArray(np.arange(30, 40)),
            _n_chunks=2,
        )
        np.testing.assert_array_equal(results[0], np.array([60, 64, 68, 72, 76]))
        np.testing.assert_array_equal(results[1], np.array([80, 84, 88, 92, 96]))
