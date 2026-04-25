"""Auto-download utilities for deepSTRF datasets.

Currently supports:
    - OSF (Open Science Framework) — public projects, no account needed.

CRCNS support is intentionally not here yet: the CRCNS download URLs require
a free account (HTTP form login + session cookie), so the pattern is
different. We'll add that under ``crcns_download`` when it lands.

Public surface:
    - ``default_cache_dir(dataset_name) -> Path``  — platformdirs-based default
    - ``stream_download(url, dest_path)``          — resumable streaming download
    - ``osf_download(file_guid, dest_path)``       — convenience for OSF files
    - ``unzip(zip_path, dest_dir)``                — flat unzip with overwrite
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
