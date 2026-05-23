import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Union, Sequence

from deepSTRF.datasets import NeuralDataset



def hanning_smooth(response: torch.Tensor, window_ms: float, dt_ms: float) -> torch.Tensor:
    """Convolve `response` along its last (time) axis with a Hanning window.

    Parameters
    ----------
    response : torch.Tensor
        Response tensor of any shape; the last axis is assumed to be time.
    window_ms : float
        Full width of the Hanning window in ms. Rounded to the nearest odd
        number of ``dt_ms`` bins (``dt_ms``-floor, then +1 if even).
    dt_ms : float
        Time-bin width of ``response``, in ms.

    Returns
    -------
    torch.Tensor
        Smoothed response, same shape as input.

    Notes
    -----
    Padded with zeros on both sides (``F.pad(..., mode='constant')``), so
    edge bins get attenuated. The kernel is the raw ``np.hanning(K)``, i.e.
    NOT sum-normalized — matches the legacy behaviour used by the Hsu /
    Borst / Theunissen (2004) PSTH smoothing step in the CRCNS-AA datasets.

    NaN-unsafe: NaN values propagate to neighbouring time bins under the
    window. Callers (e.g. ``NeuralDataset.smooth_responses``) must filter
    fully-NaN responses before calling.
    """
    assert window_ms > 0 and dt_ms > 0, "window_ms and dt_ms must be positive"
    K = int(window_ms // dt_ms)
    if K < 1:
        K = 1
    if K % 2 == 0:
        K += 1
    kernel = torch.tensor(np.hanning(K), dtype=response.dtype, device=response.device).view(1, 1, K)
    pad = (K - 1) // 2

    orig_shape = response.shape
    flat = response.reshape(-1, 1, orig_shape[-1])  # (*, 1, T)
    flat = F.pad(flat, (pad, pad), mode='constant', value=0.0)
    smoothed = F.conv1d(flat, kernel)
    return smoothed.view(orig_shape)


class ResponseSmoothingTransform(torch.nn.Module):
    """Temporally smooth responses with a Hanning window (typically ~20-40 ms).

    Parameters
    ----------
    dt_ms : float, default 1
        Time-bin width of the responses, in ms.
    window_size_ms : float, default 21
        Full width of the Hanning window in ms (rounded to an odd number of
        ``dt_ms`` bins).

    References
    ----------
    Hsu, A., Borst, A., & Theunissen, F. E. (2004). Quantifying variability
    in neural responses and its application for the validation of model
    predictions. *Network: Computation in Neural Systems*, 15(2), 91-109.
    https://doi.org/10.1088/0954-898X_15_2_002
    """

    def __init__(self, dt_ms=1, window_size_ms=21, *args, **kwargs):
        super().__init__()
        self.dt_ms = dt_ms
        self.window_size_ms = window_size_ms
        Kt_hanning = (self.window_size_ms // self.dt_ms) if ((self.window_size_ms // self.dt_ms % 2) == 1) else (self.window_size_ms // self.dt_ms) + 1  # odd kernel size
        self.hanning_window = torch.tensor(np.hanning(Kt_hanning)).unsqueeze(0).unsqueeze(0)
        self.padding_size = (Kt_hanning - 1) // 2

    def forward(self, responses, dt=1):
        # responses shape should be (B, N, R, T)  # TODO: add batch size (B)
        N, R, T = responses.shape  # TODO: add batch size (B)
        padded_responses = F.pad(responses, (self.pad_size, self.pad_size), mode='constant')

        # Apply the Hanning window using convolution
        padded_responses = padded_responses.flatten(0, 1).unsqueeze(1)  # (N, R, T) --> (N*R, 1, T)
        smoothed_responses = F.conv1d(padded_responses, self.hanning_window)
        smoothed_responses = smoothed_responses.unflatten(0, (N, R))[:, :, 0, :]  # (N*R, 1, T) --> (N, R, T)

        return smoothed_responses

    def __repr__(self):
        return f"ResponseSmoothingTransform(dt_ms={self.dt_ms}, window_size_ms={self.window_size_ms})"

    def __str__(self):
        return f"ResponseSmoothingTransform(dt_ms={self.dt_ms}, window_size_ms={self.window_size_ms})"


def neural_collate(batch):
    """Collate fn for any :class:`~deepSTRF.datasets.neural_dataset.NeuralDataset`.

    Pads variable-duration stims with zeros along the last (time) axis and
    variable-duration / variable-repeat-count responses with NaN along both
    the repeat and the time axes. Derives a fine-grained ``valid_mask`` from
    the NaN sentinels so downstream loss code can use boolean indexing or
    multiplicative masking without re-scanning.

    Parameters
    ----------
    batch : list of 4-tuples
        Each tuple is ``(stim, per_neuron_responses, per_neuron_mask,
        stim_meta)`` as yielded by ``NeuralDataset.__getitem__`` for a
        single item:

        * ``stim`` — a stim tensor of shape ``(..., T_s)`` (modality-specific
          leading dims, e.g. ``(1, F, T_s)`` for audio).
        * ``per_neuron_responses`` — list of length ``N_selected``; each
          element is a ``(R_{s,n}, T_s)`` spike-count tensor or a
          ``(1, 1)`` NaN sentinel.
        * ``per_neuron_mask`` — ``(N_selected,)`` bool tensor (currently
          ignored; the fine-grained ``valid_mask`` returned by this
          function subsumes it).
        * ``stim_meta`` — per-stim metadata dict.

    Returns
    -------
    stims : torch.Tensor
        ``(B, ..., T_max)`` float tensor, zero-padded along the last axis.
        Contains no NaN.
    responses : torch.Tensor
        ``(B, N_selected, R_max, T_max)`` float tensor. NaN-padded along
        both the repeat (``R``) and time (``T``) axes. Fully-NaN slabs mark
        (stim, neuron) pairs with no recorded data.
    valid_mask : torch.Tensor
        ``(B, N_selected, R_max, T_max)`` bool tensor. ``~responses.isnan()``,
        cached here so downstream loss code does not have to recompute.
    stim_metas : list
        Length-``B`` list of the per-item stim_meta dicts.
    """
    stims_list, resps_list, _masks_list, metas_list = zip(*batch)
    B = len(stims_list)
    N = len(resps_list[0])

    # pad stims along their time axis (last dim) with zeros.
    # fill_missing_data operates over any shape; we ask it to pad the last axis.
    stims = fill_missing_data(stims_list, dims=-1, value=0.0)
    T_max = stims.shape[-1]

    # pad each response to (R_n, T_max) along T first, tracking R_max.
    R_max = 0
    padded_per_item = []
    for b in range(B):
        padded = []
        for n in range(N):
            r = resps_list[b][n]
            pad_t = torch.full((r.shape[0], T_max), float('nan'),
                               dtype=r.dtype, device=r.device)
            pad_t[:, :r.shape[1]] = r
            padded.append(pad_t)
            R_max = max(R_max, pad_t.shape[0])
        padded_per_item.append(padded)

    # pad the repeat axis to R_max — fill the (B, N, R, T) grid.
    responses = torch.full((B, N, R_max, T_max), float('nan'),
                           dtype=stims.dtype, device=stims.device)
    for b in range(B):
        for n in range(N):
            pr = padded_per_item[b][n]
            responses[b, n, :pr.shape[0], :] = pr

    # derive fine-grained mask once per batch — the training loop gets it
    # "for free" and does not need to scan again.
    valid_mask = ~responses.isnan()

    return stims, responses, valid_mask, list(metas_list)


def concat_neural_datasets(datasets: Sequence[NeuralDataset],
                           names: Optional[Sequence[str]] = None,
                           ) -> NeuralDataset:
    """Concatenate neural datasets along BOTH the stim and neuron axes.

    Given ``k`` datasets with ``(S_i, N_i)`` stimuli and neurons each, returns
    a single dataset with ``S = sum(S_i)`` stimuli and ``N = sum(N_i)`` neurons.
    The response grid is block-diagonal: real data where a stimulus belongs
    to a given source dataset *and* the neuron belongs to the same source,
    ``(1, 1)`` NaN sentinels everywhere else. This cross-block missingness is
    paradigm-compliant — ``nrn_masks`` (the derived property) then reflects
    the block-diagonal coverage automatically.

    Primary use case: building "chimeric" datasets that pool recordings
    across species / labs / preparations (e.g. CRCNS AA1 + AA2 + NS1 for
    auditory), so that a single model can be fit to the union.

    Parameters
    ----------
    datasets : sequence of NeuralDataset
        Two or more instances. They must be of compatible types and share
        ``dt`` (bin width) and any modality-specific dimensions (``F`` for
        audio, ``(H, W)`` for video). Compatibility is checked by each
        class's ``_concat_check_compat`` hook; mismatches raise
        ``AssertionError``. Resampling to align ``dt`` or ``F`` is the
        caller's responsibility and must be done before concatenation.
    names : sequence of str, optional
        One label per input dataset, written into ``stim_meta["dataset"]``
        and ``nrn_meta["dataset"]`` on the output as a provenance
        tag. Defaults to ``[type(d).__name__ for d in datasets]`` — i.e.
        the class name (``"CRCNSAA1Dataset"`` etc.). Pass explicit names
        to disambiguate two instances of the same class, or to use a
        shorter human-readable label.

        The tags enable post-hoc selection by source dataset via
        :meth:`NeuralDataset.select_pop_by_nrn_attr` /
        :meth:`select_stims_by_attr` (e.g.
        ``c.select_pop_by_nrn_attr("dataset", "CRCNSAA1Dataset")``).
        Existing ``"dataset"`` entries in the input metadata are
        overwritten — nest-concat callers wanting to preserve inner
        provenance should pass ``names=`` explicitly.

    Returns
    -------
    NeuralDataset
        A fresh instance. Its concrete type is the most-specific class that
        is a superclass of every input (``type(datasets[0])`` when all
        inputs share a type, otherwise walks the MRO). Neuron selection is
        reset (``self.I = []``).

    Notes
    -----
    Concatenation is eager — the output holds its own full ``(S, N)`` grid
    of response references in memory. At deepSTRF scales (S, N in the low
    hundreds) this is negligible; cross-block entries are single-element
    ``(1, 1)`` NaN tensors that cost ~8 bytes each. A lazy wrapper-class
    alternative exists but would complicate ``self.responses[s][n]``
    access for uncertain benefit at this scale.

    Metadata dicts on the output are **shallow copies** of the inputs'
    (the ``"dataset"`` tag is written into the copies, never into the
    sources). Tensors and other shared values inside those dicts are not
    deep-copied — mutate at your own risk.

    Neuron / stim UID uniqueness across inputs is *not* validated — deepSTRF
    trusts the caller to pass mutually exclusive sources, since that is
    the only semantically meaningful case (pooling a dataset's subset with
    its superset is degenerate — use constructor arguments instead).

    The single-dataset case (``len(datasets) == 1``) returns the input
    unchanged, with no ``"dataset"`` tagging applied — provenance only
    becomes meaningful once there is more than one source.
    """
    assert len(datasets) >= 1, "concat_neural_datasets needs at least one dataset"
    if len(datasets) == 1:
        return datasets[0]

    first = datasets[0]
    for other in datasets[1:]:
        assert isinstance(other, NeuralDataset), \
            f"All entries must be NeuralDataset instances (got {type(other).__name__})"
        first._concat_check_compat(other)

    # resolve provenance labels (one per source dataset).
    if names is None:
        names = [type(d).__name__ for d in datasets]
    else:
        names = list(names)
        assert len(names) == len(datasets), (
            f"names must have one entry per dataset "
            f"(got {len(names)} for {len(datasets)} datasets)"
        )

    # determine concrete output type: most-specific common ancestor.
    types = [type(d) for d in datasets]
    if len(set(types)) == 1:
        out_cls = types[0]
    else:
        out_cls = NeuralDataset
        for cls in types[0].__mro__:
            if cls is object:
                break
            if all(isinstance(d, cls) for d in datasets):
                out_cls = cls
                break

    # N cumulative sum, used both for row-offset when laying out responses
    # and for the total N_neurons.
    N_cum = [0]
    for d in datasets:
        N_cum.append(N_cum[-1] + d.N_neurons)
    total_N = N_cum[-1]

    # build the result as a bare instance (skip __init__, which would
    # re-trigger data loading). All required attributes are set below.
    out = out_cls.__new__(out_cls)
    NeuralDataset.__init__(out, path="+".join(d.path for d in datasets), dt_ms=first.dt)
    out._concat_copy_attrs(first)

    # merge the core list-of-X attributes. stim_meta / nrn_meta are
    # shallow-copied per entry so the provenance tag goes onto the output
    # only — input datasets keep their original metadata dicts untouched.
    out.stims = [s for d in datasets for s in d.stims]
    out.stim_meta = [{**m, "dataset": name}
                     for d, name in zip(datasets, names)
                     for m in d.stim_meta]
    out.nrn_meta = [{**m, "dataset": name}
                           for d, name in zip(datasets, names)
                           for m in d.nrn_meta]
    out.N_neurons = total_N

    # build block-diagonal response grid.
    # for each source dataset k and each of its stims, emit a row of length
    # total_N where the k-th block holds real responses and the rest is NaN.
    nan = torch.full((1, 1), float('nan'))
    responses: List[list] = []
    for k, d in enumerate(datasets):
        prefix = [nan] * N_cum[k]
        suffix = [nan] * (total_N - N_cum[k + 1])
        for s_idx in range(len(d.stim_meta)):
            responses.append(prefix + list(d.responses[s_idx]) + suffix)
    out.responses = responses

    out.validate()
    return out


def concatenate_datasets(ds1: NeuralDataset, ds2: NeuralDataset) -> NeuralDataset:
    """Deprecated — use ``concat_neural_datasets([ds1, ds2])`` instead."""
    return concat_neural_datasets([ds1, ds2])


def fill_missing_data(stims: Sequence[torch.Tensor],
                      dims: Union[int, Sequence[int]],
                      value: float = 0.0
                      ) -> torch.Tensor:
    """Pad a list of tensors along one or more dimensions to match their maxima.

    Parameters
    ----------
    stims : sequence of torch.Tensor
        ``S`` tensors, each of shape ``(D0, D1, ..., Dk-1)``. Shapes may
        differ at the dimensions in ``dims`` but must agree on all others.
    dims : int or sequence of int
        Dimension index or indices (negatives allowed) along which to pad.
        Refer to the tensors' 0-based axes.
    value : float, default 0.0
        Fill value for padding.

    Returns
    -------
    torch.Tensor
        A tensor of shape ``(S, D0', D1', ..., Dk-1')`` where, for each ``d``
        in ``dims``, ``Dd' = max_i stims[i].shape[d]`` and, for other axes,
        ``Dd' = stims[0].shape[d]``.
    """
    if not stims:
        raise ValueError("`stims` must be a non-empty sequence of tensors")

    # Normalize dims to a sorted list of positive indices
    if isinstance(dims, int):
        dims = [dims]
    dims = sorted({d if d >= 0 else d + stims[0].ndim for d in dims})

    # Validate tensor shapes, dtype, device
    first = stims[0]
    if not isinstance(first, torch.Tensor):
        raise TypeError("All elements of `stims` must be torch.Tensor")

    k = first.ndim
    dtype, device = first.dtype, first.device

    # Check dims are in range
    for d in dims:
        if not (0 <= d < k):
            raise IndexError(f"Dimension {d} is out of bounds for tensors of ndim={k}")

    # Compute max sizes for each dim in dims
    max_sizes = {d: 0 for d in dims}
    for t in stims:
        if not isinstance(t, torch.Tensor):
            raise TypeError("All elements of `stims` must be torch.Tensor")
        if t.ndim != k:
            raise ValueError(f"All tensors must have the same number of dims; got {t.ndim} vs {k}")
        if t.dtype != dtype or t.device != device:
            raise ValueError("All tensors must share the same dtype and device")
        for d in dims:
            max_sizes[d] = max(max_sizes[d], t.shape[d])

    # Determine output shape
    out_shape = []
    for axis in range(k):
        if axis in max_sizes:
            out_shape.append(max_sizes[axis])
        else:
            # ensure consistent base shape on non-padded dims
            base = first.shape[axis]
            for t in stims:
                if t.shape[axis] != base:
                    raise ValueError(
                        f"Dimension {axis} mismatch: got {t.shape[axis]} vs {base}"
                    )
            out_shape.append(base)

    S = len(stims)
    # Preallocate output tensor: (S, *out_shape)
    out = torch.full((S, *out_shape), fill_value=value, dtype=dtype, device=device)

    # Copy each tensor into the padded output
    for i, t in enumerate(stims):
        # Build slice objects for each axis
        # out has dims (S, D0', D1', ..., Dk-1')
        # so original axis j maps to out axis j+1
        slices = [i]
        for axis in range(k):
            if axis in dims:
                slices.append(slice(0, t.shape[axis]))
            else:
                slices.append(slice(None))
        out[tuple(slices)] = t

    return out



if __name__ == "__main__":
    from deepSTRF.datasets import CRCNSAA1Dataset
    from deepSTRF.datasets import CRCNSAA2Dataset
    aa1 = CRCNSAA1Dataset('../deepSTRF/datasets/audio/CRCNS_AA1/data/', areas=('MLd', 'Field_L'), stimuli=('conspecific, flatrip'))
    aa2 = CRCNSAA2Dataset('../deepSTRF/datasets/audio/CRCNS_AA2/data/', areas=('MLd', 'OV', 'CM'), stimuli=('conspecific', 'songrip'))
    aa12 = concatenate_datasets(aa1, aa2)
