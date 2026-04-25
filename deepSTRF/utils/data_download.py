"""Auto-download utilities for deepSTRF datasets.

Public surface:
    - ``default_cache_dir(dataset_name) -> Path``       — platformdirs-based default
    - ``stream_download(url, dest_path)``               — resumable streaming download
    - ``unzip(zip_path, dest_dir)``                     — flat unzip with overwrite
    - ``osf_download(guid, dest)``                      — public OSF storage files
    - ``github_raw_download(repo, path, dest, ref=)``   — public GitHub raw files
    - ``crcns_download(file_path, dest, username=, password=)`` — CRCNS (free account)
"""

from __future__ import annotations

import os
import shutil
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

    with requests.get(url, stream=True, timeout=60, allow_redirects=True) as resp:
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


def crcns_download(
    file_path: str,
    dest_path: Union[str, Path],
    *,
    username: Optional[str] = None,
    password: Optional[str] = None,
    chunk_size: int = 1 << 20,
    progress: bool = True,
) -> Path:
    """Download a single file from the CRCNS NERSC mirror with form auth.

    The CRCNS download portal at ``https://portal.nersc.gov/project/crcns/
    download/<file_path>`` serves an HTML login form to anonymous GETs. To
    actually fetch the file, the form must be POSTed to the same URL with
    ``username`` / ``password`` / ``fn`` / ``submit`` fields. There is no
    persistent session cookie — auth is per-request, so the same pattern
    works equally well whether you fetch one file or many.

    Parameters
    ----------
    file_path : str
        Path under ``/download/``, e.g. ``"aa-1/crcns-aa1.zip"`` or
        ``"aa-4/BlaBro09xxF.tar.gz"``.
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
        If credentials are missing, or if the response body still looks like
        the login form (auth failed silently — the portal returns 200 OK
        with the login HTML rather than 401 on bad credentials).

    Notes
    -----
    Status: experimental. The auth + URL conventions were reverse-engineered
    from probing the public NERSC mirror; we do not have a contract from
    CRCNS that they'll stay stable. If the portal layout changes, this
    helper will break and is intentionally isolated from the dataset
    constructors so it can be reverted with one commit.
    """
    import os as _os

    username = username or _os.environ.get("CRCNS_USERNAME")
    password = password or _os.environ.get("CRCNS_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "CRCNS credentials missing. Pass username/password explicitly, or set "
            "the CRCNS_USERNAME / CRCNS_PASSWORD env vars. Free account at "
            "https://crcns.org/register."
        )

    dest = Path(dest_path).expanduser().resolve()
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    url = f"https://portal.nersc.gov/project/crcns/download/{file_path.lstrip('/')}"
    form = {
        "fn": file_path.lstrip("/"),
        "username": username,
        "password": password,
        "submit": "Login",
    }

    with requests.post(url, data=form, stream=True, timeout=60, allow_redirects=True) as resp:
        resp.raise_for_status()

        # NERSC returns 200 + the login form HTML on auth failure (no 401).
        # Sniff the first chunk: the real file is binary; the form is small HTML.
        first = next(resp.iter_content(chunk_size=chunk_size), b"")
        if b"<form" in first[:4096] and b"password" in first[:4096]:
            raise RuntimeError(
                f"CRCNS auth failed for {file_path!r} (server returned the login form). "
                f"Check $CRCNS_USERNAME / $CRCNS_PASSWORD."
            )

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
                if first:
                    f.write(first)
                    if bar is not None:
                        bar.update(len(first))
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
