"""Plotting helpers shared across the example notebooks.

Every function accepts either ``torch.Tensor`` or ``numpy.ndarray`` and
normalises internally; in line with the rest of the deepSTRF public API.

Functions:

- :func:`plot_stim_with_response` — spectrogram + optional spike raster
  + PSTH (with optional model prediction overlay), shared x-axis.
- :func:`plot_psth_vs_pred` — single-panel target-vs-prediction overlay.
- :func:`plot_strf_grid` — grid of STRF / gradmap kernels.

All return matplotlib objects (``Figure`` and/or ``Axes``); callers
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


def plot_strf_grid(
    strfs: Union[ArrayLike, Sequence[ArrayLike]],
    titles: Optional[Sequence[str]] = None,
    dt_ms: Optional[float] = None,
    ncols: int = 4,
    cmap: str = "RdBu_r",
    shared_clim: bool = False,
    suptitle: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[Figure, Sequence[Axes]]:
    """Plot a grid of STRF / gradmap kernels.

    The classical STRF visualisation: each panel is an ``(F, T)`` weight
    map, frequency on the y-axis (low→high), time on the x-axis
    (history; ``[0, T·dt_ms]`` if ``dt_ms`` is given, else bin index).
    Diverging colormap (``RdBu_r`` by default) with per-panel symmetric
    vmax — i.e. each cell gets its own ``|max|`` so the spatial
    structure is comparable across cells of very different gradient
    magnitudes. Pass ``shared_clim=True`` for a global symmetric vmax
    if the kernels are intended to be compared on the same scale (e.g.
    same neuron under different model variants).

    Parameters
    ----------
    strfs : array-like, shape ``(K, F, T)`` or sequence of ``(F, T)``
        Stack of kernels to plot. Numpy ndarrays or torch tensors.
    titles : sequence of str, optional
        Length-``K`` list of per-panel titles. ``None`` → unlabeled.
    dt_ms : float, optional
        Bin width in milliseconds. If given, the x-axis is in ms
        (history extent ``[0, T·dt_ms]``); otherwise it is the bin index.
    ncols : int, default 4
        Number of columns in the grid; rows = ceil(K / ncols).
    cmap : str, default ``"RdBu_r"``
    shared_clim : bool, default False
        If True, use one global symmetric ``vmax = max_k |strf_k|``
        across all panels. Otherwise per-panel.
    suptitle : str, optional
        Figure-level title.
    figsize : (w, h), optional
        Figure size. Default scales with the grid shape.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : flat list of matplotlib.axes.Axes
        Length ``K``; unused grid cells (when ``K < nrows·ncols``) are
        hidden via ``ax.axis('off')`` and not included in the return.
    """
    if hasattr(strfs, "ndim") and getattr(strfs, "ndim", None) == 3:
        # (K, F, T) stack — split into K panels.
        K = strfs.shape[0]
        panels = [_to_numpy(strfs[k]) for k in range(K)]
    else:
        panels = [_to_numpy(s) for s in strfs]
        K = len(panels)
    if K == 0:
        raise ValueError("strfs is empty — nothing to plot.")

    shapes = {p.shape for p in panels}
    if len(shapes) != 1:
        raise ValueError(
            f"all kernels must share the same (F, T) shape; got {shapes}"
        )
    F, T = panels[0].shape

    if titles is not None and len(titles) != K:
        raise ValueError(
            f"titles must have length {K}; got {len(titles)}"
        )

    if dt_ms is not None:
        t_max = T * dt_ms
        extent = [0.0, t_max, 0, F]
        xlabel = "history (ms)"
    else:
        extent = [0, T, 0, F]
        xlabel = "history (bins)"

    if shared_clim:
        global_vmax = max(float(np.abs(p).max()) for p in panels)
        if global_vmax == 0:
            global_vmax = 1e-12

    nrows = (K + ncols - 1) // ncols
    if figsize is None:
        figsize = (ncols * 3.0, nrows * 2.4)

    # constrained_layout: lets matplotlib handle the figure-level colorbar
    # and off-axes spacing without the tight_layout incompatibility warning.
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize,
                            squeeze=False, sharey=True,
                            constrained_layout=True)
    axes_flat = list(axs.flat)

    last_im = None
    for k, panel in enumerate(panels):
        ax = axes_flat[k]
        if shared_clim:
            vmax = global_vmax
        else:
            vmax = float(np.abs(panel).max())
            if vmax == 0:
                vmax = 1e-12
        last_im = ax.imshow(
            panel, aspect="auto", origin="lower",
            cmap=cmap, vmin=-vmax, vmax=vmax, extent=extent,
        )
        if titles is not None:
            ax.set_title(titles[k], fontsize=9)
        if k % ncols == 0:
            ax.set_ylabel("freq band")
        if k // ncols == nrows - 1:
            ax.set_xlabel(xlabel)

    # hide unused grid cells
    for k in range(K, len(axes_flat)):
        axes_flat[k].axis("off")

    if shared_clim and last_im is not None:
        fig.colorbar(last_im, ax=axs, fraction=0.025, pad=0.02,
                     label="STRF weight")

    if suptitle is not None:
        fig.suptitle(suptitle)

    return fig, axes_flat[:K]


def compare_wav2spec_to_groundtruth(
    ds,
    wav2spec,
    stim_idx: int = 0,
    *,
    ground_truth_stims: Optional[Sequence] = None,
    z_score: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    suptitle: Optional[str] = None,
):
    """Side-by-side visual comparison of a learned/hand-built ``wav2spec``
    output against a dataset's precomputed (ground-truth) spectrogram.

    Useful for sanity-checking a new front-end on a dataset that ships both
    raw waveforms and a precomputed spectrogram (e.g. NS1, where the OSF
    release has raw wavs and the DNet companion repo provides the matching
    log-mel ``X_nfht``).

    Parameters
    ----------
    ds : NeuralDataset
        Dataset instance in **waveform mode** (``ds.stims[s].shape ==
        (1, T_audio)``). The dataset's regular spectrogram is treated as
        the ground truth, supplied via ``ground_truth_stims``.
    wav2spec : nn.Module
        Module to apply. Must accept ``(B, 1, T_audio)`` and return
        ``(B, 1, F, T_neural)`` — i.e. the wav2spec slot contract.
    stim_idx : int, default 0
        Which dataset stim to compare. Indexes ``ds.stims``.
    ground_truth_stims : sequence, optional
        Per-stim ground-truth spectrograms (each ``(F, T)`` or ``(1, F, T)``).
        Required: the waveform-mode dataset doesn't carry them itself. For
        NS1 build them by re-instantiating ``NS1Dataset()`` (spec mode) and
        passing its ``stims``.
    z_score : bool, default True
        If True, both spectrograms are independently z-scored (mean 0, std 1)
        before plotting, so a constant offset / global scale mismatch does
        not visually dominate the comparison.

    Returns
    -------
    pred_spec : numpy.ndarray
        The wav2spec output, shape ``(F, T)``.
    truth_spec : numpy.ndarray
        The ground-truth spectrogram, shape ``(F, T)``.
    fig : matplotlib.figure.Figure
        3-panel side-by-side figure (pred | truth | difference).
    """
    import torch

    if ground_truth_stims is None:
        raise ValueError(
            "ground_truth_stims is required — the waveform-mode dataset does "
            "not carry the precomputed spec internally. Pass the spec-mode "
            "dataset's ``stims`` list (e.g. ``NS1Dataset().stims``)."
        )

    wav = ds.stims[stim_idx]
    if wav.dim() == 2:
        wav = wav.unsqueeze(0)  # (1, 1, T_audio)
    wav2spec.eval()
    with torch.no_grad():
        pred = wav2spec(wav)  # (1, 1, F, T)
    pred_np = _to_numpy(pred).squeeze()  # (F, T)

    truth = ground_truth_stims[stim_idx]
    truth_np = _to_numpy(truth).squeeze()  # (F, T)

    # crop to common T
    T_common = min(pred_np.shape[-1], truth_np.shape[-1])
    pred_np = pred_np[..., :T_common]
    truth_np = truth_np[..., :T_common]

    if z_score:
        pred_np = (pred_np - pred_np.mean()) / (pred_np.std() + 1e-12)
        truth_np = (truth_np - truth_np.mean()) / (truth_np.std() + 1e-12)

    fig, axs = plt.subplots(1, 3, figsize=figsize or (10, 3), sharex=True, sharey=True)
    vmax = float(max(np.abs(pred_np).max(), np.abs(truth_np).max()))
    diff = pred_np - truth_np
    vmax_diff = float(np.abs(diff).max() + 1e-12)

    axs[0].imshow(pred_np, aspect="auto", origin="lower", cmap="viridis",
                  vmin=-vmax, vmax=vmax)
    axs[0].set_title("wav2spec output")
    axs[1].imshow(truth_np, aspect="auto", origin="lower", cmap="viridis",
                  vmin=-vmax, vmax=vmax)
    axs[1].set_title("ground truth")
    im = axs[2].imshow(diff, aspect="auto", origin="lower", cmap="RdBu_r",
                       vmin=-vmax_diff, vmax=vmax_diff)
    axs[2].set_title("difference")
    for ax in axs:
        ax.set_xlabel("time bin")
    axs[0].set_ylabel("freq band")
    if suptitle is not None:
        fig.suptitle(suptitle)

    return pred_np, truth_np, fig
