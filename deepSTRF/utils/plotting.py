"""Plotting helpers shared across the example notebooks.

Every function accepts either ``torch.Tensor`` or ``numpy.ndarray`` and
normalises internally; in line with the rest of the deepSTRF public API.

Functions:

- :func:`plot_stim_with_response` — spectrogram + optional spike raster
  + PSTH (with optional model prediction overlay), shared x-axis.
- :func:`plot_psth_vs_pred` — single-panel target-vs-prediction overlay.

Both return matplotlib objects (``Figure`` and/or ``Axes``); callers
decide whether to ``plt.show()``, save, or compose further. No
``plt.show`` is invoked inside.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


ArrayLike = Union[np.ndarray, "torch.Tensor"]


# Canonical colours / line widths for the PSTH-vs-prediction overlay.
# Matched against fit_ns1_statenet.ipynb and load_pretrained_statenet_ns1.ipynb.
_PSTH_COLOR = "#222"
_PRED_COLOR = "#d62728"
_PSTH_LW = 1.3
_PRED_LW = 1.0


def _to_numpy(x: ArrayLike) -> np.ndarray:
    """Best-effort conversion: torch.Tensor -> numpy, leave numpy alone."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def plot_stim_with_response(
    stim: ArrayLike,
    response: ArrayLike,
    pred: Optional[ArrayLike] = None,
    dt_ms: Optional[float] = None,
    title: Optional[str] = None,
    spec_cmap: str = "magma",
    raster_cmap: str = "Greys",
    axes: Optional[Sequence[Axes]] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[Figure, Sequence[Axes]]:
    """Plot a stimulus spectrogram alongside its recorded (and optionally predicted) response.

    Builds a vertically stacked figure with a shared time axis. The
    middle raster panel is omitted automatically when ``response`` is
    1-D (already a PSTH).

    Parameters
    ----------
    stim : array-like
        Stimulus spectrogram, shape ``(F, T)`` or ``(1, F, T)``.
    response : array-like
        Per-trial responses ``(R, T)`` or pre-averaged PSTH ``(T,)`` /
        ``(1, T)``. ``R > 1`` → spec/raster/PSTH; otherwise spec/PSTH.
    pred : array-like, optional
        Model prediction, shape ``(T,)``. Overlaid on the PSTH panel.
    dt_ms : float, optional
        Bin width in milliseconds. If given, the x-axis is in seconds;
        otherwise it is the bin index.
    title : str, optional
        Suptitle for the whole figure.
    spec_cmap, raster_cmap : str
        Colormaps for the spectrogram and raster panels.
    axes : sequence of matplotlib.axes.Axes, optional
        Pre-made axes to draw into (2 or 3, matching the panel count).
        If None, a new figure is created.
    figsize : (w, h), optional
        Figure size when ``axes`` is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of matplotlib.axes.Axes
        ``[spec_ax, raster_ax, psth_ax]`` or ``[spec_ax, psth_ax]``.
    """
    spec = _to_numpy(stim)
    if spec.ndim == 3:
        if spec.shape[0] != 1:
            raise ValueError(
                f"stim must be (F, T) or (1, F, T); got shape {spec.shape}"
            )
        spec = spec[0]
    if spec.ndim != 2:
        raise ValueError(f"stim must be (F, T) or (1, F, T); got shape {spec.shape}")
    F, T_spec = spec.shape

    resp = _to_numpy(response)
    if resp.ndim == 1:
        psth = resp
        raster = None
    elif resp.ndim == 2:
        R, T_r = resp.shape
        if R == 1:
            psth = resp[0]
            raster = None
        else:
            psth = np.nanmean(resp, axis=0)
            raster = resp
    else:
        raise ValueError(
            f"response must be 1D (T,) or 2D (R, T); got shape {resp.shape}"
        )

    T = psth.shape[-1]
    if T_spec != T:
        raise ValueError(
            f"stim time axis ({T_spec}) and response time axis ({T}) disagree"
        )

    if dt_ms is not None:
        t = np.arange(T) * (dt_ms / 1000.0)
        xlabel = "time (s)"
    else:
        t = np.arange(T)
        xlabel = "time (bin)"

    n_panels = 3 if raster is not None else 2
    if axes is None:
        if figsize is None:
            figsize = (9, 5.5 if n_panels == 3 else 4)
        height_ratios = [2, 2, 1] if n_panels == 3 else [2, 1]
        fig, axes_arr = plt.subplots(
            n_panels, 1, figsize=figsize, sharex=True,
            gridspec_kw={"height_ratios": height_ratios},
        )
        axes = list(np.atleast_1d(axes_arr))
    else:
        axes = list(axes)
        if len(axes) != n_panels:
            raise ValueError(
                f"axes must have length {n_panels} for this response shape; "
                f"got {len(axes)}"
            )
        fig = axes[0].figure

    # --- spec ---
    axes[0].imshow(
        spec, aspect="auto", origin="lower", cmap=spec_cmap,
        extent=[t[0], t[-1], 0, F],
    )
    axes[0].set_ylabel("freq band")

    # --- optional raster ---
    if raster is not None:
        axes[1].imshow(
            raster, aspect="auto", cmap=raster_cmap,
            extent=[t[0], t[-1], 0, raster.shape[0]],
        )
        axes[1].set_ylabel(f"trial (R={raster.shape[0]})")
        psth_ax = axes[2]
    else:
        psth_ax = axes[1]

    # --- PSTH (+ optional prediction) ---
    psth_ax.plot(t, psth, lw=_PSTH_LW, color=_PSTH_COLOR,
                 label="PSTH (target)" if pred is not None else None)
    if pred is not None:
        pred_arr = _to_numpy(pred)
        if pred_arr.ndim != 1 or pred_arr.shape[0] != T:
            raise ValueError(
                f"pred must be 1D with length {T}; got shape {pred_arr.shape}"
            )
        psth_ax.plot(t, pred_arr, lw=_PRED_LW, color=_PRED_COLOR, label="model")
        psth_ax.legend(loc="upper right", fontsize=9)
    psth_ax.set_ylabel("PSTH")
    psth_ax.set_xlabel(xlabel)

    if title is not None:
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.tight_layout()

    return fig, axes


def plot_psth_vs_pred(
    target: ArrayLike,
    pred: ArrayLike,
    dt_ms: Optional[float] = None,
    title: Optional[str] = None,
    target_label: str = "PSTH (target)",
    pred_label: str = "model",
    ax: Optional[Axes] = None,
    legend: bool = True,
) -> Axes:
    """Overlay a target PSTH and a model prediction on a single panel.

    Designed to be called inside a per-cell loop ("best / median / worst")
    that pre-builds a column of axes — this is the canonical
    val/test visualisation in ``fit_ns1_statenet.ipynb`` and
    ``load_pretrained_statenet_ns1.ipynb``.

    Parameters
    ----------
    target : array-like, shape (T,)
        Trial-averaged target PSTH.
    pred : array-like, shape (T,)
        Model prediction.
    dt_ms : float, optional
        Bin width in milliseconds. If given the x-axis is in seconds;
        otherwise it is the bin index.
    title : str, optional
        Axes title.
    target_label, pred_label : str
        Legend labels.
    ax : matplotlib.axes.Axes, optional
        Pre-made axes to draw into. If None, a new figure is created.
    legend : bool, default True
        Whether to render the legend.

    Returns
    -------
    matplotlib.axes.Axes
    """
    y = _to_numpy(target)
    p = _to_numpy(pred)
    if y.ndim != 1:
        raise ValueError(f"target must be 1D (T,); got shape {y.shape}")
    if p.shape != y.shape:
        raise ValueError(
            f"target and pred shapes disagree: {y.shape} vs {p.shape}"
        )
    T = y.shape[0]

    if dt_ms is not None:
        t = np.arange(T) * (dt_ms / 1000.0)
        xlabel = "time (s)"
    else:
        t = np.arange(T)
        xlabel = "time (bin)"

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 2.2))

    ax.plot(t, y, lw=_PSTH_LW, color=_PSTH_COLOR, label=target_label)
    ax.plot(t, p, lw=_PRED_LW, color=_PRED_COLOR, label=pred_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("rate")
    if title is not None:
        ax.set_title(title)
    if legend:
        ax.legend(loc="upper right", fontsize=9)
    return ax
