"""NAT4 est and val responses must come in the same units (spike counts per bin).

Integration test — needs the unpacked NAT4 release (population recording plus the
per-site files) under the deepSTRF cache (or at $NAT4_DATA) and skips automatically when missing.
"""
from __future__ import annotations

import os

import numpy as np
import pytest


def _root() -> str:
    """$NAT4_DATA if set, else the deepSTRF cache."""
    from deepSTRF.utils.data_download import default_cache_dir
    return os.environ.get("NAT4_DATA") or str(default_cache_dir("NAT4"))


def _has_nat4() -> bool:
    root = _root()
    return (os.path.isdir(os.path.join(root, "A1_NAT4_ozgf.fs100.ch18"))
            and os.path.isdir(os.path.join(root, "A1_single_sites")))


def test_nat4_est_and_val_in_same_units():
    """est (population recording, R=1) and val (per-site files, R=20) responses are both
    spike counts per 10-ms bin: est is integer-valued, and a cell's mean response agrees
    across the two subsets within ~2x (it was ~7x before the est min-max was dropped)."""
    if not _has_nat4():
        pytest.skip("NAT4 cache missing — skip integration test")
    from deepSTRF.datasets.audio.nat4 import NAT4Dataset

    ds = NAT4Dataset(path=_root(), area="A1", subset="all")
    est = [s for s, m in enumerate(ds.stim_meta) if m["subset"] == "est"]
    val = [s for s, m in enumerate(ds.stim_meta) if m["subset"] == "val"]

    def cell_means(stims):
        tot = np.zeros(ds.N_neurons)
        cnt = np.zeros(ds.N_neurons)
        for s in stims:
            for n, r in enumerate(ds.responses[s]):
                r = r.numpy()
                if np.isfinite(r).any():
                    tot[n] += np.nansum(r)
                    cnt[n] += np.isfinite(r).sum()
        return tot / np.where(cnt > 0, cnt, np.nan)

    est_vals = np.concatenate([r.numpy().ravel() for s in est[:20] for r in ds.responses[s]])
    est_vals = est_vals[np.isfinite(est_vals)]
    assert np.allclose(est_vals, np.round(est_vals)), "est responses are not spike counts"

    m_est, m_val = cell_means(est), cell_means(val)
    ok = (m_est > 0) & (m_val > 0)
    ratio = np.median(m_val[ok] / m_est[ok])
    assert 0.5 < ratio < 2.0, f"val/est median per-cell mean ratio = {ratio:.2f}"
