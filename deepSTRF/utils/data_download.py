"""Auto-download utilities for deepSTRF datasets.

Public surface:
    - ``default_cache_dir(dataset_name) -> Path``       — platformdirs-based default
    - ``stream_download(url, dest_path)``               — resumable streaming download
    - ``unzip(zip_path, dest_dir)``                     — flat unzip with overwrite
    - ``untar(tar_path, dest_dir, strip_components=)``  — tar.gz / .tar / .tar.bz2 unpack
    - ``osf_download(guid, dest)``                      — public OSF storage files
    - ``github_raw_download(repo, path, dest, ref=)``   — public GitHub raw files
    - ``zenodo_download(record_id, filename, dest)``    — public Zenodo records
    - ``figshare_download(article_id, dest_dir, filename=)`` — public figshare articles
    - ``crcns_download(file_path, dest, username=, password=)`` — CRCNS (free account)
    - ``crcns_file_list(dataset)``                       — CRCNS dataset file paths + sizes
"""

from __future__ import annotations

import os
import re
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Optional, Union

# requests + platformdirs are runtime deps (cf. pyproject.toml). Imported here
# so this module can be used standalone outside dataset constructors too.
import requests
from platformdirs import user_cache_dir


def default_cache_dir(dataset_name: str) -> Path:
    """Return ``$DEEPSTRF_DATA_DIR/<dataset>`` if the env var is set,
    otherwise ``platformdirs.user_cache_dir('deepSTRF') / <dataset>``.

    The env-var override is the standard escape hatch for users on shared
    storage / scratch filesystems, and matches the convention in
    torchvision / huggingface_hub.
    """
    base = os.environ.get("DEEPSTRF_DATA_DIR")
    if base:
        return Path(base).expanduser() / dataset_name
    return Path(user_cache_dir("deepSTRF")) / dataset_name


def stream_download(
    url: str,
    dest_path: Union[str, Path],
    *,
    chunk_size: int = 1 << 20,
    progress: bool = True,
) -> Path:
    """Stream-download ``url`` to ``dest_path``. Atomic via a ``.part`` swap.

    Already-existing destination paths are returned unchanged (no-op) — the
    caller is responsible for cache invalidation.

    Parameters
    ----------
    url : str
    dest_path : str | Path
    chunk_size : int, default 1 MiB
    progress : bool, default True
        Show a tqdm progress bar if available; falls back silently otherwise.

    Returns
    -------
    Path
        The destination path (resolved).
    """
    dest = Path(dest_path).expanduser().resolve()
    if dest.exists():
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    # (connect, read): 30s to establish, then 10 min between recvs. The read
    # timeout has to be generous because slow upstream mirrors (CRCNS / NERSC,
    # OSF on a busy day) can pause for tens of seconds between chunks while
    # the file is fetched from cold storage.
    with requests.get(url, stream=True, timeout=(30, 600), allow_redirects=True) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length") or 0)

        bar = None
        if progress:
            try:
                from tqdm.auto import tqdm
                bar = tqdm(total=total or None, unit="B", unit_scale=True,
                           desc=f"download {dest.name}")
            except ImportError:
                bar = None

        try:
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    f.write(chunk)
                    if bar is not None:
                        bar.update(len(chunk))
        finally:
            if bar is not None:
                bar.close()

    tmp.replace(dest)
    return dest


def osf_download(file_guid: str, dest_path: Union[str, Path], **kwargs) -> Path:
    """Download a single file from OSF by its short GUID.

    Resolves to ``https://osf.io/download/<guid>/`` — works for any public
    OSF storage file (the OSF API exposes this URL as the ``download`` link
    in each file's metadata).

    Example
    -------
    >>> osf_download("gdwyd", "MetadataSHEnCneurons.mat")
    """
    return stream_download(f"https://osf.io/download/{file_guid}/", dest_path, **kwargs)


# CRCNS moved its data to AWS in June 2026 (https://crcns.org/news/data-now-
# downloaded-through-aws). Downloads now go through download.crcns.org behind a
# crcns.org login session, as in the official client
# https://github.com/jeffteeters/crcns-downloader (MIT); the old NERSC portal
# (portal.nersc.gov/project/crcns/download/) no longer serves the files.
CRCNS_LOGIN_URL = "https://crcns.org/login_form"
CRCNS_DOWNLOAD_BASE = "https://download.crcns.org"
_CRCNS_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
             "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
_crcns_sessions: dict = {}


def _crcns_credentials(username: Optional[str], password: Optional[str]):
    username = username or os.environ.get("CRCNS_USERNAME")
    password = password or os.environ.get("CRCNS_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "CRCNS credentials missing. Pass username/password explicitly, or set "
            "the CRCNS_USERNAME / CRCNS_PASSWORD env vars. Free account at "
            "https://crcns.org/register."
        )
    return username, password


def _crcns_session(dataset: str, username: Optional[str] = None,
                   password: Optional[str] = None) -> "requests.Session":
    """Logged-in ``requests.Session`` for download.crcns.org (cached per user).

    Logs in once on crcns.org, then checks that the download server
    recognises the session for ``dataset`` (its landing page then reads
    "Logged in as <username> ...").
    """
    username, password = _crcns_credentials(username, password)
    sess = _crcns_sessions.get(username)
    if sess is None:
        sess = requests.Session()
        sess.headers["User-Agent"] = _CRCNS_UA
        resp = sess.post(CRCNS_LOGIN_URL, timeout=(30, 120), data={
            "form.submitted": 1, "js_enabled": 0, "pwd_empty": 0,
            "__ac_name": username, "__ac_password": password, "submit": "Log in",
        })
        resp.raise_for_status()
        page = sess.get(f"{CRCNS_DOWNLOAD_BASE}/{dataset}", timeout=(30, 120))
        page.raise_for_status()
        if "logged in as" not in page.text.lower():
            raise RuntimeError(
                "CRCNS login failed (download.crcns.org does not recognise the session). "
                "Check $CRCNS_USERNAME / $CRCNS_PASSWORD."
            )
        _crcns_sessions[username] = sess
    return sess


def crcns_file_list(dataset: str, *, username: Optional[str] = None,
                    password: Optional[str] = None) -> dict:
    """Return ``{path: size_in_bytes}`` for every file of a CRCNS dataset.

    Reads the dataset's ``filelist.txt`` on download.crcns.org (paths are
    relative to the dataset, e.g. ``"crcns-aa1.zip"``). Needs a CRCNS login.
    """
    sess = _crcns_session(dataset, username, password)
    resp = sess.get(f"{CRCNS_DOWNLOAD_BASE}/{dataset}/filelist.txt", timeout=(30, 120))
    resp.raise_for_status()
    out = {}
    for line in resp.text.splitlines():
        # data lines: " <path> <size> [(<human size>)]"; '#' lines are comments
        m = re.match(r"^[ +]\s*(\S+)\s+(\d+)", line)
        if m:
            out[m.group(1)] = int(m.group(2))
    if not out:
        raise RuntimeError(f"CRCNS: no files listed for dataset {dataset!r}.")
    return out


def crcns_download(
    file_path: str,
    dest_path: Union[str, Path],
    *,
    username: Optional[str] = None,
    password: Optional[str] = None,
    chunk_size: int = 1 << 20,
    progress: bool = True,
) -> Path:
    """Download one file of a CRCNS dataset (free account needed).

    Logs in to crcns.org (once per process and user), then streams
    ``https://download.crcns.org/<file_path>``. Interrupted downloads resume
    from the ``.part`` file via an HTTP ``Range`` request.

    Parameters
    ----------
    file_path : str
        ``"<dataset>/<path in the dataset>"``, e.g. ``"aa-1/crcns-aa1.zip"``
        or ``"aa-4/BlaBro09xxF.tar.gz"``. Use :func:`crcns_file_list` to see
        a dataset's paths.
    dest_path : path-like
    username, password : str, optional
        Default to ``$CRCNS_USERNAME`` / ``$CRCNS_PASSWORD``. Account is
        free at https://crcns.org/register.
    chunk_size, progress
        As ``stream_download``.

    Returns
    -------
    Path
        Resolved destination.

    Raises
    ------
    RuntimeError
        If credentials are missing or rejected, if the server answers with an
        HTML page instead of the file, or if the file does not exist.

    Notes
    -----
    Follows the protocol of the official client
    (https://github.com/jeffteeters/crcns-downloader). CRCNS gives no API
    stability guarantee: if their site changes, this helper breaks.

    Example
    -------
    >>> import os
    >>> os.environ["CRCNS_USERNAME"] = "..."
    >>> os.environ["CRCNS_PASSWORD"] = "..."
    >>> crcns_download("aa-1/crcns-aa1.zip", "/tmp/aa1.zip")
    """
    username, password = _crcns_credentials(username, password)
    dest = Path(dest_path).expanduser().resolve()
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    file_path = file_path.lstrip("/")
    dataset = file_path.split("/", 1)[0]
    sess = _crcns_session(dataset, username, password)

    done = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    # (connect, read): generous read timeout, as the server may pause while
    # fetching a large archive from cold storage.
    with sess.get(f"{CRCNS_DOWNLOAD_BASE}/{file_path}", headers=headers, stream=True,
                  timeout=(30, 600), allow_redirects=True) as resp:
        if resp.status_code == 404:
            raise RuntimeError(
                f"CRCNS: {file_path!r} not found on download.crcns.org. "
                f"See crcns_file_list({dataset!r}) for the dataset's file paths."
            )
        resp.raise_for_status()
        if done and resp.status_code != 206:          # server ignored the Range header
            done = 0
        # A login / error page comes back as 200 + HTML instead of the file.
        chunks = resp.iter_content(chunk_size=chunk_size)     # ONE iterator for the whole body
        first = next(chunks, b"")
        head = first[:4096].lower()
        if "text/html" in resp.headers.get("Content-Type", "") and (b"<html" in head or b"<form" in head):
            raise RuntimeError(
                f"CRCNS returned an HTML page instead of {file_path!r} (login rejected or "
                f"file unavailable). Check $CRCNS_USERNAME / $CRCNS_PASSWORD."
            )

        total = int(resp.headers.get("Content-Length") or 0) + done
        bar = None
        if progress:
            try:
                from tqdm.auto import tqdm
                bar = tqdm(total=total or None, initial=done, unit="B", unit_scale=True,
                           desc=f"download {dest.name}")
            except ImportError:
                bar = None
        try:
            with open(tmp, "ab" if done else "wb") as f:
                if first:
                    f.write(first)
                    if bar is not None:
                        bar.update(len(first))
                for chunk in chunks:
                    if not chunk:
                        continue
                    f.write(chunk)
                    if bar is not None:
                        bar.update(len(chunk))
        finally:
            if bar is not None:
                bar.close()

    tmp.replace(dest)
    return dest


def zenodo_download(
    record_id: Union[int, str],
    filename: str,
    dest_path: Union[str, Path],
    **kwargs,
) -> Path:
    """Download a single file from a public Zenodo record.

    Resolves to ``https://zenodo.org/api/records/<record_id>/files/<filename>/content``
    — the canonical URL for fetching a file from a Zenodo record. Public
    records are accessible without auth.

    Example
    -------
    >>> zenodo_download(8044773, "A1_NAT4_ozgf.fs100.ch18.tgz", "/tmp/X.tgz")
    """
    url = f"https://zenodo.org/api/records/{record_id}/files/{filename}/content"
    return stream_download(url, dest_path, **kwargs)


def figshare_download(
    article_id: Union[int, str],
    dest_dir: Union[str, Path],
    *,
    filename: Optional[str] = None,
    **kwargs,
) -> Path:
    """Download one file from a public figshare article.

    Resolves the article's file list via the public REST API
    (``https://api.figshare.com/v2/articles/<id>``) and streams the matching
    file into ``dest_dir``. With ``filename=None``, the article must contain
    exactly one file; pass an explicit name to disambiguate when there are
    several.

    Parameters
    ----------
    article_id : int | str
        Numeric figshare article id (the trailing component of the DOI
        ``10.6084/m9.figshare.<id>``, e.g. ``29203457``).
    dest_dir : path-like
        Directory the file is downloaded into. Created if missing.
    filename : str, optional
        Name of the file to fetch. Required when the article carries more
        than one file. Matched case-sensitively against the file's ``name``
        field returned by the API.

    Returns
    -------
    Path
        Path to the downloaded file under ``dest_dir``.

    Example
    -------
    >>> figshare_download(29203457, "/tmp/le2025")
    PosixPath('/tmp/le2025/zebf-auditory-restoration-1.zip')
    """
    api_url = f"https://api.figshare.com/v2/articles/{article_id}"
    resp = requests.get(api_url, timeout=(30, 60))
    resp.raise_for_status()
    files = resp.json().get("files", [])
    if not files:
        raise RuntimeError(f"figshare article {article_id} lists no files")

    if filename is None:
        if len(files) > 1:
            names = ", ".join(f["name"] for f in files)
            raise ValueError(
                f"figshare article {article_id} has {len(files)} files; "
                f"pass `filename=` to disambiguate (candidates: {names})"
            )
        chosen = files[0]
    else:
        chosen = next((f for f in files if f.get("name") == filename), None)
        if chosen is None:
            names = ", ".join(f["name"] for f in files)
            raise FileNotFoundError(
                f"file {filename!r} not in figshare article {article_id} (have: {names})"
            )

    dest_dir = Path(dest_dir).expanduser().resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / chosen["name"]
    return stream_download(chosen["download_url"], dest, **kwargs)


def github_raw_download(
    repo: str,
    path_in_repo: str,
    dest_path: Union[str, Path],
    *,
    ref: str = "HEAD",
    **kwargs,
) -> Path:
    """Download a file from a GitHub repo's raw content.

    Useful for paper-companion repos that publish small datasets / model
    artefacts alongside the code (e.g. DNet hosts ``test_data_5ms.mat`` for
    the Rahman et al. 2018 NS1 reanalysis at
    https://github.com/monzilur/DNet).

    Parameters
    ----------
    repo : str
        ``"<owner>/<name>"``, e.g. ``"monzilur/DNet"``.
    path_in_repo : str
        Path of the file within the repo, e.g. ``"test_data_5ms.mat"``.
    dest_path : path-like
    ref : str, default "HEAD"
        Branch / tag / commit. ``"HEAD"`` resolves the default branch.

    Notes
    -----
    Uses the ``raw.githubusercontent.com`` CDN, which has no rate limit for
    anonymous reads (unlike the GitHub REST API).
    """
    url = f"https://raw.githubusercontent.com/{repo}/{ref}/{path_in_repo.lstrip('/')}"
    return stream_download(url, dest_path, **kwargs)


def unzip(zip_path: Union[str, Path], dest_dir: Union[str, Path], *, strip_root: bool = False) -> Path:
    """Unzip ``zip_path`` into ``dest_dir``. Idempotent (overwrites existing files).

    Parameters
    ----------
    zip_path, dest_dir : path-like
    strip_root : bool, default False
        If True and the archive contains a single top-level directory, strip
        it from the extracted layout (so ``foo/a/b`` -> ``a/b``). Mirrors the
        common ``--strip-components=1`` tar idiom.

    Returns
    -------
    Path
        The destination directory.
    """
    zip_path = Path(zip_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        prefix = ""
        if strip_root and members:
            roots = {m.split("/", 1)[0] for m in members if m}
            if len(roots) == 1:
                prefix = next(iter(roots)) + "/"
        for m in members:
            if prefix and not m.startswith(prefix):
                continue
            target = m[len(prefix):] if prefix else m
            if not target:
                continue
            out = dest_dir / target
            if m.endswith("/"):
                out.mkdir(parents=True, exist_ok=True)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
    return dest_dir


def untar(tar_path: Union[str, Path], dest_dir: Union[str, Path],
          *, strip_components: int = 0) -> Path:
    """Extract a tar / tar.gz / tar.bz2 archive into ``dest_dir``.

    Mirrors GNU ``tar --strip-components=N``: drops the first ``N`` path
    components from every member. Useful when an archive wraps everything
    in nested directories that aren't part of the dataset's own layout
    — e.g. CRCNS-AA2 archives all wrap content in ``crcns/aa2/`` (strip 2).

    Parameters
    ----------
    tar_path, dest_dir : path-like
    strip_components : int, default 0
        How many leading path components to drop. Members that have fewer
        components than this are silently skipped.

    Returns
    -------
    Path
        The destination directory.
    """
    tar_path = Path(tar_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "r:*") as tf:
        for member in tf.getmembers():
            parts = member.name.split("/")
            # GNU tar's --strip-components silently skips members with too
            # few components (e.g. the root dir entry itself).
            if strip_components and len(parts) <= strip_components:
                continue
            stripped = "/".join(parts[strip_components:])
            if not stripped:
                continue
            out = dest_dir / stripped
            if member.isdir():
                out.mkdir(parents=True, exist_ok=True)
                continue
            if not (member.isfile() or member.islnk() or member.issym()):
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            with extracted as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
    return dest_dir
