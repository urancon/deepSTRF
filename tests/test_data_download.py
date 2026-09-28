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


def test_zenodo_download_resolves_url(monkeypatch, tmp_path):
    """zenodo_download(record_id, name, dest) hits the records API URL."""
    from deepSTRF.utils import data_download as dd

    captured = {}

    def fake_stream(url, dest_path, **kwargs):
        captured["url"] = url
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        Path(dest_path).write_bytes(b"")
        return Path(dest_path)

    monkeypatch.setattr(dd, "stream_download", fake_stream)
    dd.zenodo_download(8044773, "A1_NAT4_ozgf.fs100.ch18.tgz", tmp_path / "x.tgz")
    assert captured["url"] == "https://zenodo.org/api/records/8044773/files/A1_NAT4_ozgf.fs100.ch18.tgz/content"


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


class _FakeCRCNSSession:
    """Stand-in for the logged-in ``requests.Session`` used for download.crcns.org.

    ``files`` maps "<dataset>/<path>" -> bytes; ``logged_in`` controls what the
    dataset landing page says after the login POST.
    """

    def __init__(self, files, logged_in=True, html_instead_of=()):
        self.files, self.logged_in, self.html_instead_of = files, logged_in, set(html_instead_of)
        self.headers, self.calls = {}, []

    def post(self, url, data=None, **kw):
        self.calls.append(("POST", url, dict(data or {})))
        return _FakePostResp(b"<html>" + b"x" * 2000 + b"</html>")

    def get(self, url, headers=None, **kw):
        self.calls.append(("GET", url, dict(headers or {})))
        from deepSTRF.utils.data_download import CRCNS_DOWNLOAD_BASE
        path = url[len(CRCNS_DOWNLOAD_BASE) + 1:]
        if "/" not in path:                                   # dataset landing page
            txt = "Logged in as u (User Name). logout" if self.logged_in else "Login Required"
            r = _FakePostResp(txt.encode())
            r.text = txt
            return r
        if path.endswith("/filelist.txt"):
            ds = path.split("/")[0]
            txt = "# mode='default'\n" + "".join(
                f" {k.split('/', 1)[1]} {len(v)} (1 kB)\n" for k, v in self.files.items() if k.startswith(ds + "/"))
            r = _FakePostResp(txt.encode())
            r.text = txt
            return r
        if path in self.html_instead_of:
            return _FakePostResp(b"<html><form>login</form></html>", headers={"Content-Type": "text/html"})
        if path not in self.files:
            return _FakePostResp(b"not found", status_code=404)
        body = self.files[path]
        rng = (headers or {}).get("Range")
        if rng:
            start = int(rng.split("=")[1].rstrip("-"))
            return _FakePostResp(body[start:], headers={"Content-Length": str(len(body) - start)},
                                 status_code=206)
        return _FakePostResp(body, headers={"Content-Length": str(len(body))})


@pytest.fixture
def fake_crcns(monkeypatch):
    from deepSTRF.utils import data_download as dd
    sessions = []

    def install(files, **kw):
        sess = _FakeCRCNSSession(files, **kw)
        sessions.append(sess)
        monkeypatch.setattr(dd.requests, "Session", lambda: sess)
        monkeypatch.setattr(dd, "_crcns_sessions", {})
        return sess
    return install


def test_crcns_download_logs_in_then_fetches_from_download_server(fake_crcns, tmp_path):
    from deepSTRF.utils import data_download as dd
    sess = fake_crcns({"aa-1/foo.bin": b"\x00\x01binary-payload"})
    dest = tmp_path / "out.bin"
    dd.crcns_download("aa-1/foo.bin", dest, username="u", password="p", progress=False)
    assert dest.read_bytes() == b"\x00\x01binary-payload"
    method, url, form = sess.calls[0]
    assert (method, url) == ("POST", dd.CRCNS_LOGIN_URL)
    assert form["__ac_name"] == "u" and form["__ac_password"] == "p"
    assert ("GET", f"{dd.CRCNS_DOWNLOAD_BASE}/aa-1/foo.bin", {}) in sess.calls


def test_crcns_session_is_reused(fake_crcns, tmp_path):
    from deepSTRF.utils import data_download as dd
    sess = fake_crcns({"aa-4/a.tgz": b"a", "aa-4/b.tgz": b"b"})
    dd.crcns_download("aa-4/a.tgz", tmp_path / "a", username="u", password="p", progress=False)
    dd.crcns_download("aa-4/b.tgz", tmp_path / "b", username="u", password="p", progress=False)
    assert sum(c[0] == "POST" for c in sess.calls) == 1        # one login


def test_crcns_download_resumes_partial_file(fake_crcns, tmp_path):
    from deepSTRF.utils import data_download as dd
    sess = fake_crcns({"aa-5/big.tar.gz": b"0123456789"})
    dest = tmp_path / "big.tar.gz"
    (tmp_path / "big.tar.gz.part").write_bytes(b"0123")
    dd.crcns_download("aa-5/big.tar.gz", dest, username="u", password="p", progress=False)
    assert dest.read_bytes() == b"0123456789"
    assert any(c[2].get("Range") == "bytes=4-" for c in sess.calls if c[0] == "GET")


def test_crcns_download_rejected_login(fake_crcns, tmp_path):
    from deepSTRF.utils import data_download as dd
    fake_crcns({"aa-1/foo.bin": b"x"}, logged_in=False)
    with pytest.raises(RuntimeError, match="CRCNS login failed"):
        dd.crcns_download("aa-1/foo.bin", tmp_path / "out.bin", username="bad", password="creds",
                          progress=False)
    assert not (tmp_path / "out.bin").exists()


def test_crcns_download_html_instead_of_file(fake_crcns, tmp_path):
    from deepSTRF.utils import data_download as dd
    fake_crcns({"aa-1/foo.bin": b"x"}, html_instead_of={"aa-1/foo.bin"})
    with pytest.raises(RuntimeError, match="HTML page instead"):
        dd.crcns_download("aa-1/foo.bin", tmp_path / "out.bin", username="u", password="p",
                          progress=False)
    assert not (tmp_path / "out.bin").exists()


def test_crcns_download_missing_file(fake_crcns, tmp_path):
    from deepSTRF.utils import data_download as dd
    fake_crcns({"aa-1/foo.bin": b"x"})
    with pytest.raises(RuntimeError, match="not found on download.crcns.org"):
        dd.crcns_download("aa-1/nope.bin", tmp_path / "out.bin", username="u", password="p",
                          progress=False)


def test_crcns_file_list(fake_crcns):
    from deepSTRF.utils import data_download as dd
    fake_crcns({"aa-5/ZF4F/a.tar.gz": b"abc", "aa-5/docs/x.pdf": b"12345", "aa-4/other": b"z"})
    assert dd.crcns_file_list("aa-5", username="u", password="p") == {"ZF4F/a.tar.gz": 3, "docs/x.pdf": 5}


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
