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


def test_untar_strip_components(tmp_path):
    """untar(strip_components=N) drops N leading path components per member."""
    import tarfile
    from deepSTRF.utils.data_download import untar

    tar_path = tmp_path / "wrapped.tar.gz"
    src = tmp_path / "tree"
    (src / "crcns" / "aa2" / "all_cells").mkdir(parents=True)
    (src / "crcns" / "aa2" / "all_cells" / "cell.txt").write_text("hi")
    (src / "crcns" / "aa2" / "stim_data.csv").write_text("a,b,c")
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(src / "crcns", arcname="crcns")

    out = tmp_path / "out"
    untar(tar_path, out, strip_components=2)
    assert (out / "all_cells" / "cell.txt").read_text() == "hi"
    assert (out / "stim_data.csv").read_text() == "a,b,c"
    assert not (out / "crcns").exists()
    assert not (out / "aa2").exists()


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


def test_github_raw_download_resolves_url(monkeypatch, tmp_path):
    """github_raw_download builds a raw.githubusercontent.com URL."""
    from deepSTRF.utils import data_download as dd

    captured = {}

    def fake_stream(url, dest_path, **kwargs):
        captured["url"] = url
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        Path(dest_path).write_bytes(b"")
        return Path(dest_path)

    monkeypatch.setattr(dd, "stream_download", fake_stream)
    dd.github_raw_download("monzilur/DNet", "test_data_5ms.mat",
                           tmp_path / "x.mat", ref="master")
    assert captured["url"] == "https://raw.githubusercontent.com/monzilur/DNet/master/test_data_5ms.mat"


class _FakePostResp:
    """Minimal stand-in for ``requests.Response`` (POST + iter_content + ctx mgr)."""

    def __init__(self, body: bytes, headers=None, status_code: int = 200):
        self._body = body
        self.headers = headers or {}
        self.status_code = status_code

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size=1):
        # one big chunk is fine for tests
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]


def test_crcns_download_writes_file(monkeypatch, tmp_path):
    """Happy path: server returns binary, helper writes it to dest."""
    from deepSTRF.utils import data_download as dd

    captured = {}

    def fake_post(url, data, **kwargs):
        captured["url"] = url
        captured["data"] = data
        return _FakePostResp(b"\x00\x01\x02\x03binary-payload",
                             headers={"Content-Length": "20"})

    monkeypatch.setattr(dd.requests, "post", fake_post)

    dest = tmp_path / "out.bin"
    dd.crcns_download("aa-1/foo.bin", dest, username="u", password="p", progress=False)
    assert dest.read_bytes().endswith(b"binary-payload")
    assert captured["url"] == "https://portal.nersc.gov/project/crcns/download/aa-1/foo.bin"
    assert captured["data"]["fn"] == "aa-1/foo.bin"
    assert captured["data"]["username"] == "u"
    assert captured["data"]["password"] == "p"


def test_crcns_download_detects_auth_failure(monkeypatch, tmp_path):
    """If the server returns the login HTML (200 OK), helper must raise."""
    from deepSTRF.utils import data_download as dd

    login_html = (
        b"<html><body><form action=''>"
        b"<input name='username' /><input name='password' type='password' />"
        b"</form></body></html>"
    )

    monkeypatch.setattr(dd.requests, "post",
                        lambda url, data, **kw: _FakePostResp(login_html))

    with pytest.raises(RuntimeError, match="CRCNS auth failed"):
        dd.crcns_download("aa-1/foo.bin", tmp_path / "out.bin",
                          username="bad", password="creds", progress=False)
    # no partial file should be left around
    assert not (tmp_path / "out.bin").exists()


def test_crcns_download_requires_credentials(monkeypatch, tmp_path):
    from deepSTRF.utils import data_download as dd

    monkeypatch.delenv("CRCNS_USERNAME", raising=False)
    monkeypatch.delenv("CRCNS_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="CRCNS credentials missing"):
        dd.crcns_download("aa-1/foo.bin", tmp_path / "out.bin", progress=False)


def test_stream_download_skips_existing(tmp_path):
    """stream_download is a no-op if the destination already exists."""
    from deepSTRF.utils.data_download import stream_download

    dest = tmp_path / "already-there.bin"
    dest.write_bytes(b"x" * 64)
    # passing a bogus URL: would 404 if it actually got fetched
    out = stream_download("https://nonexistent.invalid/X", dest)
    assert out == dest.resolve()
    assert dest.read_bytes() == b"x" * 64
