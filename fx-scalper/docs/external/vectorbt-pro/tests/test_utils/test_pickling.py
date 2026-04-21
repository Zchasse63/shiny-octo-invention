import os

import pytest

import vectorbtpro as vbt
from tests.utils import *
from vectorbtpro._dtypes import *
from vectorbtpro.utils import pickling

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


class TestPickling:
    @pytest.mark.parametrize("test_file_format", ["ini", "yml", pytest.param("toml", marks=requires_tomlkit)])
    def test_pdict(self, tmp_path, test_file_format):
        index = pd.date_range("2023", periods=5)
        columns = pd.Index(["a", "b", "c"], name="symbol")
        wrapper = vbt.ArrayWrapper(index, columns)
        acc1 = vbt.GenericAccessor(wrapper, wrapper.fill(0).values)
        acc2 = vbt.GenericAccessor(wrapper, wrapper.fill(1).values)
        d1 = dict(acc1=acc1)
        d2 = dict(acc1=acc1, acc2=acc2)
        d3 = d2
        d4 = dict(a=dict(b=dict(d3=d3)))
        pdict = pickling.pdict(hello="world", cls=vbt.ArrayWrapper, d1=d1, d2=d2, d3=d3, d4=d4)
        pdict.save(tmp_path / "pdict")
        assert pickling.pdict.load(tmp_path / "pdict") == pdict
        pdict.save(tmp_path / "pdict", rec_state_only=True)
        assert pickling.pdict.load(tmp_path / "pdict") == pdict
        pdict.save(tmp_path / "pdict", file_format=test_file_format)
        assert pickling.pdict.load(tmp_path / "pdict", file_format=test_file_format) == pdict
        pdict.save(tmp_path / "pdict", file_format=test_file_format, nested=False)
        assert pickling.pdict.load(tmp_path / "pdict", file_format=test_file_format) == pdict
        pdict.save(tmp_path / "pdict", file_format=test_file_format, use_refs=False)
        assert pickling.pdict.load(tmp_path / "pdict", file_format=test_file_format, use_refs=False) == pdict
        pdict.save(tmp_path / "pdict", file_format=test_file_format, use_class_ids=False)
        assert pickling.pdict.load(tmp_path / "pdict", file_format=test_file_format, use_class_ids=False) == pdict

    def test_compression(self, tmp_path):
        vbt.Config(a=0).save(tmp_path)
        with pytest.raises(Exception):
            vbt.Config(a=1).save(tmp_path, compression=True)
        vbt.Config(a=2).save(tmp_path, compression=False)
        vbt.Config(a=3).save(tmp_path, compression="zip")
        vbt.Config(a=4).save(tmp_path, compression="gzip")
        vbt.Config(a=5).save(tmp_path, compression="gz")
        vbt.Config(a=6).save(tmp_path, compression="bz2")

        assert vbt.Config.load(tmp_path)["a"] == 2
        assert vbt.Config.load(tmp_path, compression=False)["a"] == 2
        assert vbt.Config.load(tmp_path, compression="zip")["a"] == 3
        assert vbt.Config.load(tmp_path, compression="gzip")["a"] == 4
        assert vbt.Config.load(tmp_path, compression="gz")["a"] == 5
        assert vbt.Config.load(tmp_path, compression="bz")["a"] == 6
        (tmp_path / "Config.pickle").unlink()
        with pytest.raises(Exception):
            vbt.Config.load(tmp_path)
        (tmp_path / "Config.pickle.bz2").unlink()
        with pytest.raises(Exception):
            vbt.Config.load(tmp_path, compression="bz")

    def test_dumps_loads(self):
        obj = {"a": [1, 2, 3], "b": {"c": "d"}}

        assert pickling.loads(pickling.dumps(obj)) == obj
        assert pickling.loads(pickling.dumps(obj, compression="gzip"), compression="gz") == obj

    def test_save_and_load(self, tmp_path):
        obj = {"a": [1, 2, 3], "b": {"c": "d"}}

        file_path = pickling.save(obj, tmp_path / "obj", append_suffix=True)
        assert file_path == tmp_path / "obj.pickle"
        assert pickling.load(tmp_path / "obj") == obj

        file_path = pickling.save(obj, tmp_path / "obj.pickle.gz")
        assert file_path == tmp_path / "obj.pickle.gz"
        assert pickling.load(tmp_path / "obj", compression=False) == obj
        assert pickling.load(tmp_path / "obj", compression="gz") == obj

        target_dir = tmp_path / "objects"
        target_dir.mkdir()
        file_path = pickling.save(obj, target_dir, compression="gzip", append_suffix=True)
        assert file_path == target_dir / "dict.pickle.gzip"
        assert pickling.load(target_dir / "dict", compression="gz") == obj

        file_path = pickling.save(obj, tmp_path / "payload", compression="gzip")
        assert file_path == tmp_path / "payload"
        assert pickling.load(tmp_path / "payload", compression="gz") == obj

    def test_save_bytes_and_load_bytes(self, tmp_path):
        bytes_ = b"hello world"

        file_path = pickling.save_bytes(bytes_, tmp_path / "data.zip")
        assert file_path == tmp_path / "data.zip"
        assert pickling.load_bytes(tmp_path / "data") == bytes_

        file_path = pickling.save_bytes(bytes_, tmp_path / "plain.pickle.gz")
        assert file_path == tmp_path / "plain.pickle.gz"
        assert pickling.load_bytes(tmp_path / "plain") == bytes_

        plain_bytes = b"plain bytes"
        gz_bytes = b"gz bytes"
        file_path = pickling.save_bytes(plain_bytes, tmp_path / "raw.pickle")
        assert file_path == tmp_path / "raw.pickle"
        file_path = pickling.save_bytes(gz_bytes, tmp_path / "raw.pickle.gz")
        assert file_path == tmp_path / "raw.pickle.gz"
        assert pickling.load_bytes(tmp_path / "raw", compression=False) == plain_bytes
        assert pickling.load_bytes(tmp_path / "raw", compression="gz") == gz_bytes

    def test_file_path_resolution_helpers(self, tmp_path):
        assert pickling.suggest_compression("file.pkl.gz") == "gz"
        assert pickling.suggest_compression("file.pkl") is None
        assert pickling.suggest_compression("file.txt") is None

        assert (
            pickling.suggest_file_path(tmp_path / "base", compression="gz", append_suffix=True)
            == tmp_path / "base.pickle.gz"
        )
        assert (
            pickling.suggest_file_path(tmp_path / "base.pickle", compression="gz", append_suffix=True)
            == tmp_path / "base.pickle.gz"
        )
        assert (
            pickling.suggest_file_path(tmp_path / "base.pickle.gz", compression="gz", append_suffix=True)
            == tmp_path / "base.pickle.gz"
        )
        assert (
            pickling.suggest_file_path(tmp_path / "base.pkl.gzip", compression="gz", append_suffix=True)
            == tmp_path / "base.pkl.gzip"
        )

        (tmp_path / "foo").write_bytes(b"base")
        (tmp_path / "foo.pkl").write_bytes(b"pickle")
        assert pickling.resolve_file_path(tmp_path / "foo") == tmp_path / "foo"

        (tmp_path / "foo2").write_bytes(b"base")
        (tmp_path / "foo2.pkl.gz").write_bytes(b"pickle")
        assert pickling.resolve_file_path(tmp_path / "foo2") == tmp_path / "foo2"
        assert pickling.resolve_file_path(tmp_path / "foo2", compression=False) == tmp_path / "foo2"
        assert pickling.resolve_file_path(tmp_path / "foo2", compression="gzip") == tmp_path / "foo2.pkl.gz"

        (tmp_path / "alias.pkl.gz").write_bytes(b"test")
        assert pickling.resolve_file_path(tmp_path / "alias.pickle.gzip") == tmp_path / "alias.pkl.gz"
        assert pickling.resolve_file_path(tmp_path / "alias.pkl.gzip") == tmp_path / "alias.pkl.gz"
        assert pickling.resolve_file_path(tmp_path / "alias.pickle.gz") == tmp_path / "alias.pkl.gz"

        (tmp_path / "compressed.pickle").write_bytes(b"plain")
        (tmp_path / "compressed.pickle.gz").write_bytes(b"gzip")
        assert pickling.resolve_file_path(tmp_path / "compressed", compression=False) == tmp_path / "compressed.pickle"
        assert (
            pickling.resolve_file_path(tmp_path / "compressed", compression="gz") == tmp_path / "compressed.pickle.gz"
        )
        assert (
            pickling.resolve_file_path(tmp_path / "compressed.pkl", compression="gzip")
            == tmp_path / "compressed.pickle.gz"
        )
        assert (
            pickling.resolve_file_path(tmp_path / "compressed.pickle", compression="gzip")
            == tmp_path / "compressed.pickle.gz"
        )

        (tmp_path / "infer.pickle.gz").write_bytes(b"infer")
        assert pickling.resolve_file_path(tmp_path / "infer.pkl.gz") == tmp_path / "infer.pickle.gz"
        with pytest.raises(FileNotFoundError):
            pickling.resolve_file_path(tmp_path / "infer.pkl.zst")

        (tmp_path / "plain_or_gzip.pickle").write_bytes(b"plain")
        (tmp_path / "plain_or_gzip.pickle.gz").write_bytes(b"gzip")
        assert (
            pickling.resolve_file_path(tmp_path / "plain_or_gzip", compression=False)
            == tmp_path / "plain_or_gzip.pickle"
        )
        assert (
            pickling.resolve_file_path(tmp_path / "plain_or_gzip.pkl", compression=False)
            == tmp_path / "plain_or_gzip.pickle"
        )

        assert (
            pickling.resolve_file_path(tmp_path / "plain_or_gzip", compression="gz")
            == tmp_path / "plain_or_gzip.pickle.gz"
        )
        assert (
            pickling.resolve_file_path(tmp_path / "plain_or_gzip.pkl", compression="gzip")
            == tmp_path / "plain_or_gzip.pickle.gz"
        )

        (tmp_path / "only_compressed.pkl.gz").write_bytes(b"gzip")
        with pytest.raises(FileNotFoundError):
            pickling.resolve_file_path(tmp_path / "only_compressed", compression=False)
        assert pickling.resolve_file_path(tmp_path / "only_compressed", compression=False, raise_error=False) is None
        with pytest.raises(FileNotFoundError):
            pickling.resolve_file_path(tmp_path / "only_compressed.pkl.gz", compression=False)
        assert pickling.resolve_file_path(tmp_path / "only_compressed.pkl.gz", compression=False, raise_error=False) is None

        (tmp_path / "only_plain.pkl").write_bytes(b"plain")
        with pytest.raises(FileNotFoundError):
            pickling.resolve_file_path(tmp_path / "only_plain", compression="gz")
        assert pickling.resolve_file_path(tmp_path / "only_plain", compression="gz", raise_error=False) is None

        (tmp_path / "obj.pickle").write_bytes(b"a")
        (tmp_path / "obj.pkl").write_bytes(b"b")
        with pytest.raises(ValueError):
            pickling.resolve_file_path(tmp_path / "obj")
        assert pickling.resolve_file_path(tmp_path / "obj", raise_error=False) is None

        (tmp_path / "ambig_comp.pkl.gz").write_bytes(b"gz")
        (tmp_path / "ambig_comp.pkl.zip").write_bytes(b"zip")
        with pytest.raises(ValueError):
            pickling.resolve_file_path(tmp_path / "ambig_comp.pkl")
        assert pickling.resolve_file_path(tmp_path / "ambig_comp.pkl", raise_error=False) is None

        (tmp_path / "prefix.pkl").write_bytes(b"a")
        (tmp_path / "prefix_extra.pkl").write_bytes(b"b")
        assert pickling.resolve_file_path(tmp_path / "prefix") == tmp_path / "prefix.pkl"

        (tmp_path / "dircase.pkl").mkdir()
        (tmp_path / "dircase.pickle").write_bytes(b"file")
        assert pickling.resolve_file_path(tmp_path / "dircase") == tmp_path / "dircase.pickle"

        with pytest.raises(FileNotFoundError):
            pickling.resolve_file_path(tmp_path / "missing")
        assert pickling.resolve_file_path(tmp_path / "missing", raise_error=False) is None

        assert pickling.resolve_file_path(tmp_path / "missing" / "child", raise_error=False) is None
        with pytest.raises(FileNotFoundError):
            pickling.resolve_file_path(tmp_path / "does_not_exist" / "file.pkl")
        assert pickling.resolve_file_path(tmp_path / "does_not_exist" / "file.pkl", raise_error=False) is None

    def test_load_bytes_rejects_explicit_compressed_path(self, tmp_path):
        file_path = pickling.save_bytes(b"gzip", tmp_path / "raw_only.pickle.gz")
        assert file_path == tmp_path / "raw_only.pickle.gz"

        with pytest.raises(FileNotFoundError):
            pickling.load_bytes(tmp_path / "raw_only.pickle.gz", compression=False)
