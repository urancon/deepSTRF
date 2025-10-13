import torch
from typing import List, Union, Sequence

from deepSTRF.datasets import NeuralDataset
from deepSTRF.datasets.audio import CRCNS_AA4_Dataset


def aa4_collate(batch):
    # TODO:
    #  1) make more general, to all AudioNeuralDatasets, not just AA4
    #  2) make more general, to all NeuralDatasets, not just Audio ones
    """Collate function for CRCNS_AA4_Dataset DataLoader."""
    specs_list, resps_list, masks_list, metas_list = zip(*batch)
    # specs_list: list length B of (1,F,T_s)
    # resps_list: list length B of list length N of (R_n,T_s)
    # masks_list: list length B of (N,)
    B = len(specs_list)
    N = masks_list[0].shape[0]
    # pad specs along time dim (dim=2)
    specs = fill_missing_data(specs_list, dims=2, value=0.0)
    # pad responses along trial dim (dim=1) and time dim (dim=2)
    # first flatten responses to list per batch and neuron
    # we want shape (B, N, R_max, T_max)
    # prepare per-stim per-neuron zipping
    Rmax = 0
    Tmax = specs.shape[-1]
    # collect all response tensors, pad time to Tmax
    padded_resps = []
    for b in range(B):
        per_stim = resps_list[b]
        # pad each neuron's resp to time
        padded = []
        for n in range(N):
            r = per_stim[n]
            # pad time dim to Tmax
            pad_t = torch.full((r.shape[0], Tmax), float('nan'), dtype=r.dtype, device=r.device)
            pad_t[:, :r.shape[1]] = r
            padded.append(pad_t)
            Rmax = max(Rmax, pad_t.shape[0])
        padded_resps.append(padded)
    # now pad trial dim to Rmax
    resps = torch.full((B, N, Rmax, Tmax), float('nan'), dtype=specs.dtype, device=specs.device)
    for b in range(B):
        for n in range(N):
            pr = padded_resps[b][n]
            resps[b, n, :pr.shape[0], :] = pr
    # masks: stack
    masks = torch.stack(masks_list, dim=0)
    return specs, resps, masks, list(metas_list)


def concatenate_datasets(ds1: CRCNS_AA4_Dataset, ds2: CRCNS_AA4_Dataset) -> CRCNS_AA4_Dataset:
    """
    TODO: make ds1 and ds2 AudioNeuralDatasets or even NeuralDataset --> move what makes CRCNS_AA4_Dataset so special
     (i.e., its structure and attributes, but which ones ?) up a level.

    Concatenate two CRCNS_AA4_Dataset instances with disjoint neurons and stimuli.
    Returns a new dataset with combined stimuli and neurons, padding missing responses.
    Select all neurons of both datasets by default.
    """
    # create new instance without calling __init__
    new_ds = object.__new__(CRCNS_AA4_Dataset)  # TODO: rather instanciate an AudioNeuralDataset while calling its constructor
    # copy configuration
    new_ds.dt = ds1.dt
    new_ds.F = ds1.F
    new_ds.smooth = ds1.smooth
    new_ds.stim_types = ds1.stim_types.union(ds2.stim_types)
    new_ds.animals = list(ds1.animals) + [a for a in ds2.animals if a not in ds1.animals]

    # combine stimuli
    new_ds.stims = ds1.stims + ds2.stims
    new_ds.stim_meta = ds1.stim_meta + ds2.stim_meta
    S1, S2 = len(ds1.stims), len(ds2.stims)
    # combine neuron metadata
    new_ds.nrn_meta = ds1.nrn_meta + ds2.nrn_meta
    N1, N2 = len(ds1.nrn_meta), len(ds2.nrn_meta)
    new_ds.N_neurons = N1 + N2

    # build new responses and masks
    new_responses = []
    new_masks = []
    nan_tensor = torch.full((1,1), float('nan'))
    # ds1 stimuli: pad ds1 responses with ds2 neurons missing
    for i in range(S1):
        resp1 = ds1.responses[i]
        mask1 = ds1.nrn_masks[i]
        # pad responses
        combined = []
        for r in resp1:
            combined.append(r)
        for _ in range(N2):
            combined.append(nan_tensor)
        # pad mask
        new_mask = torch.cat([mask1, torch.zeros(N2, dtype=torch.bool)], dim=0)
        new_responses.append(combined)
        new_masks.append(new_mask)
    # ds2 stimuli: pad ds2 responses with ds1 neurons missing
    for j in range(S2):
        resp2 = ds2.responses[j]
        mask2 = ds2.nrn_masks[j]
        combined = []
        for _ in range(N1):
            combined.append(nan_tensor)
        for r in resp2:
            combined.append(r)
        new_mask = torch.cat([torch.zeros(N1, dtype=torch.bool), mask2], dim=0)
        new_responses.append(combined)
        new_masks.append(new_mask)

    new_ds.responses = new_responses
    new_ds.nrn_masks = new_masks
    # default SELECT list
    new_ds.I = list(range(N1+N2))
    return new_ds


def fill_missing_data(stims: Sequence[torch.Tensor],
                      dims: Union[int, Sequence[int]],
                      value: float = 0.0
                      ) -> torch.Tensor:
    """Pad a list of tensors along one or more specified dimensions to match their maxima.

    Args:
        stims (Sequence[torch.Tensor]): List of S tensors, each of shape
            (D0, D1, ..., Dk-1), where shapes may differ at the dimensions in `dims`
            but must agree on all other dimensions.
        dims (int or Sequence[int]): Dimension index or indices (can be negative)
            along which to pad. These refer to the tensor’s axes (0-based).
        value (float, optional): Fill value for padding. Defaults to 0.0.

    Returns:
        torch.Tensor: A tensor of shape (S, D0', D1', ..., Dk-1'),
            where for each d in `dims`, Dd' = max_i stims[i].shape[d],
            and for other axes Dd' = stims[0].shape[d].
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
    from deepSTRF.datasets import CRCNS_AA1_Dataset
    from deepSTRF.datasets import CRCNS_AA2_Dataset
    aa1 = CRCNS_AA1_Dataset('../deepSTRF/datasets/audio/CRCNS_AA1/data/', areas=('MLd', 'Field_L'), stimuli=('conspecific, flatrip'))
    aa2 = CRCNS_AA2_Dataset('../deepSTRF/datasets/audio/CRCNS_AA2/data/', areas=('MLd', 'OV', 'CM'), stimuli=('conspecific', 'songrip'))
    aa12 = concatenate_datasets(aa1, aa2)
