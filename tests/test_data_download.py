"""Network-free contract tests for ``deepSTRF.utils.data_download``.

We don't hit OSF in CI — the OSF endpoint is exercised manually as part of
the NS1 download smoke test. Here we just check the local helpers
(``default_cache_dir``, ``unzip``) and that the URL builder for OSF is
correct.
"""

import io
import os
import zipfile
from pathlib import Path

import pytest


def test_default_cache_dir_respects_env(monkeypatch, tmp_path):
    from deepSTRF.utils.data_download import default_cache_dir

    monkeypatch.setenv("DEEPSTRF_DATA_DIR", str(tmp_path))
    out = default_cache_dir("NS1")
    assert out == tmp_path / "NS1"


def test_default_cache_dir_falls_back_to_platformdirs(monkeypatch):
    from deepSTRF.utils.data_download import default_cache_dir

    monkeypatch.delenv("DEEPSTRF_DATA_DIR", raising=False)
    out = default_cache_dir("NS1")
    assert out.name == "NS1"
    # platformdirs path is OS-specific; we just check the function returned a Path
    assert isinstance(out, Path)


def test_unzip_flat(tmp_path):
    """unzip extracts a flat archive into the destination directory."""
    from deepSTRF.utils.data_download import unzip

    zip_path = tmp_path / "tiny.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("a.txt", "hello")
        zf.writestr("sub/b.txt", "world")

    out_dir = tmp_path / "out"
    unzip(zip_path, out_dir)
    assert (out_dir / "a.txt").read_text() == "hello"
    assert (out_dir / "sub" / "b.txt").read_text() == "world"


def test_unzip_strip_root(tmp_path):
    """unzip(strip_root=True) drops a single common top-level dir."""
    from deepSTRF.utils.data_download import unzip

    zip_path = tmp_path / "with_root.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("project/a.txt", "hello")
        zf.writestr("project/sub/b.txt", "world")

    out_dir = tmp_path / "out"
    unzip(zip_path, out_dir, strip_root=True)
    assert (out_dir / "a.txt").read_text() == "hello"
    assert (out_dir / "sub" / "b.txt").read_text() == "world"
    assert not (out_dir / "project").exists()


def test_osf_download_resolves_url(monkeypatch, tmp_path):
    """osf_download(<guid>, dest) hits the canonical /download/<guid>/ URL."""
    from deepSTRF.utils import data_download as dd

    captured = {}

    def fake_stream(url, dest_path, **kwargs):
        captured["url"] = url
        captured["dest"] = str(dest_path)
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        Path(dest_path).write_bytes(b"")
        return Path(dest_path)

    monkeypatch.setattr(dd, "stream_download", fake_stream)
    out = dd.osf_download("gdwyd", tmp_path / "X.mat")
    assert captured["url"] == "https://osf.io/download/gdwyd/"
    assert out == tmp_path / "X.mat"


def test_stream_download_skips_existing(tmp_path):
    """stream_download is a no-op if the destination already exists."""
    from deepSTRF.utils.data_download import stream_download

    dest = tmp_path / "already-there.bin"
    dest.write_bytes(b"x" * 64)
    # passing a bogus URL: would 404 if it actually got fetched
    out = stream_download("https://nonexistent.invalid/X", dest)
    assert out == dest.resolve()
    assert dest.read_bytes() == b"x" * 64
