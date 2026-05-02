"""Reproducibility seeding for ``deepSTRF.training``.

See ``docs/_source/md/fitter.md`` §7.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_random_seed(seed: int, *, strict: bool = False) -> None:
    """Seed Python ``random``, NumPy, PyTorch CPU, and CUDA.

    Parameters
    ----------
    seed
        Integer seed value, broadcast to all four RNGs.
    strict
        If True, additionally enables ``torch.use_deterministic_algorithms(True)``
        (with the ``CUBLAS_WORKSPACE_CONFIG`` env var required by CUBLAS) so that
        any non-deterministic CUDA op raises rather than silently breaking
        reproducibility. Trades speed for bit-exactness; mainly useful for
        debugging. Default ``False``.

    Notes
    -----
    Even with ``strict=False``, ``cudnn.deterministic`` is set to True and
    ``cudnn.benchmark`` to False — that is the legacy ``deepSTRF`` behaviour
    and is enough for run-to-run reproducibility on the same hardware so long
    as no opt-in non-deterministic kernel is invoked.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if strict:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
