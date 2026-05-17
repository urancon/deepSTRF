import os
from typing import List, Tuple, Optional, Sequence, Union

import numpy as np
import torch
import pandas as pd

try:
    import xarray as xr
    _XARRAY_AVAILABLE = True
except ImportError:
    _XARRAY_AVAILABLE = False


_XARRAY_INSTALL_HINT = (
    "UWChallenge_Dataset reads netCDF files via xarray. Install it with:\n"
    "    pip install 'deepSTRF[allen]'\n"
    "(the [allen] extra installs both xarray and allensdk)."
)

# re-use your fill_missing_data helper (expects same dtype/device across tensors)
def fill_missing_data(stims: Sequence[torch.Tensor],
                      dims: Union[int, Sequence[int]],
                      value: float = 0.0) -> torch.Tensor:
    if not stims:
        raise ValueError("`stims` must be a non-empty sequence of tensors")
    if isinstance(dims, int):
        dims = [dims]
    dims = sorted({d if d >= 0 else d + stims[0].ndim for d in dims})
    first = stims[0]
    if not isinstance(first, torch.Tensor):
        raise TypeError("All elements of `stims` must be torch.Tensor")
    k = first.ndim
    dtype, device = first.dtype, first.device
    for d in dims:
        if not (0 <= d < k):
            raise IndexError(f"Dimension {d} is out of bounds for tensors of ndim={k}")
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
    out_shape = []
    for axis in range(k):
        if axis in max_sizes:
            out_shape.append(max_sizes[axis])
        else:
            base = first.shape[axis]
            for t in stims:
                if t.shape[axis] != base:
                    raise ValueError(f"Dimension {axis} mismatch: got {t.shape[axis]} vs {base}")
            out_shape.append(base)
    S = len(stims)
    out = torch.full((S, *out_shape), fill_value=value, dtype=dtype, device=device)
    for i, t in enumerate(stims):
        slices = [i]
        for axis in range(k):
            if axis in dims:
                slices.append(slice(0, t.shape[axis]))
            else:
                slices.append(slice(None))
        out[tuple(slices)] = t
    return out


class UWChallenge_Dataset:
    """
    Loader for the UW Neural Data Challenge dataset (Kaggle).
    Expected folder contents (path):
      - spike_data.nc
      - stim.npy
      - stim_wpos_dva.csv  (optional)
      - sub.csv             (optional, per-unit subject info)
      - train.csv           (optional metadata)
    """

    def __init__(self, path: str, device: Optional[torch.device] = None):
        if not _XARRAY_AVAILABLE:
            raise ImportError(_XARRAY_INSTALL_HINT)

        self.path = path
        self.device = device or torch.device('cpu')

        # final attributes
        self.stims: List[torch.Tensor] = []        # [S * (3, H, W, 1)]
        self.stim_meta: List[Tuple] = []          # [S * (idx, ...)]
        self.resps: List[List[torch.Tensor]] = [] # [S * [N * (R, T)]]
        self.nrn_meta: List[Tuple] = []           # [N * ...]
        self.nrn_masks: List[torch.Tensor] = []   # [S * (N,)]

        # load stimuli
        stim_path = os.path.join(path, 'stim.npy')
        if not os.path.isfile(stim_path):
            raise FileNotFoundError(f"stim.npy not found in {path}")
        stim_arr = np.load(stim_path)  # expect (S, H, W, C) or (S, H, W) etc.
        # standardize to (S, C, H, W)
        if stim_arr.ndim == 3:  # (S, H, W)
            stim_arr = stim_arr[..., np.newaxis]  # (S, H, W, 1)
        if stim_arr.shape[-1] == 3:
            # (S, H, W, 3)
            stim_arr = stim_arr.transpose(0, 3, 1, 2)  # (S, C, H, W)
        else:
            # move channel to dim 1, pad/truncate to 3 channels if needed
            # broadcast single-channel to 3 identical channels
            if stim_arr.shape[-1] == 1:
                stim_arr = np.repeat(stim_arr, 3, axis=-1)
                stim_arr = stim_arr.transpose(0, 3, 1, 2)
            else:
                # unexpected channel number; try to put channels first as-is
                stim_arr = stim_arr.transpose(0, 3, 1, 2)

        S = stim_arr.shape[0]

        # convert to torch tensors, add time dim = 1
        self.stims = [torch.from_numpy(stim_arr[i].astype(np.float32)).to(self.device).unsqueeze(-1)
                      for i in range(S)]
        # read optional stim_wpos_dva.csv for meta (if present)
        stim_wpos_path = os.path.join(path, 'stim_wpos_dva.csv')
        stim_wpos_df = None
        if os.path.isfile(stim_wpos_path):
            stim_wpos_df = pd.read_csv(stim_wpos_path, index_col=0)
        # create basic stim_meta: (stim_index, maybe position row)
        for i in range(S):
            meta = (i,)
            if stim_wpos_df is not None and str(i) in stim_wpos_df.index:
                meta = (i, stim_wpos_df.loc[str(i)].to_dict())
            self.stim_meta.append(meta)

        # load spike_data.nc (netCDF) via xarray
        nc_path = os.path.join(path, 'spike_data.nc')
        if not os.path.isfile(nc_path):
            raise FileNotFoundError(f"spike_data.nc not found in {path}")

        ds = xr.open_dataset(nc_path)
        # heuristics to pick the variable likely containing responses
        candidate_vars = [v for v in ds.data_vars.keys()
                          if any(k in v.lower() for k in ('spike', 'resp', 'response', 'count', 'rates'))]
        if not candidate_vars:
            # fallback: pick first numeric variable
            candidate_vars = list(ds.data_vars.keys())

        chosen_var = candidate_vars[0]
        arr = ds[chosen_var].values  # numpy array
        # diagnostic info (useful to copy/paste if shape mapping is wrong)
        print(f"[UW loader] chosen variable in spike_data.nc: '{chosen_var}', shape={arr.shape}, dtype={arr.dtype}")

        # arr can have many possible orderings. We'll try to find the stimulus axis by matching S.
        S_in_arr_axes = [i for i, sz in enumerate(arr.shape) if sz == S]
        if not S_in_arr_axes:
            # If no axis exactly matches S, but S is small, try a nearest candidate
            sorted_axes = sorted(range(arr.ndim), key=lambda i: abs(arr.shape[i] - S))
            stim_axis = sorted_axes[0]
            print(f"[UW loader] warning: no exact axis matching S={S}; choosing axis {stim_axis} (size {arr.shape[stim_axis]}) as stimulus axis.")
        else:
            stim_axis = S_in_arr_axes[0]

        # Now we will permute arr so that stimulus axis is first: (S, ...)
        if stim_axis != 0:
            perm = [stim_axis] + [i for i in range(arr.ndim) if i != stim_axis]
            arr = np.transpose(arr, perm)

        # After permutation arr.shape = (S, ...). We need to map the remaining axes to neuron, repeat, time.
        # Heuristics:
        # - If arr.ndim == 4, assume (S, N, R, T)
        # - If arr.ndim == 3, assume (S, N, T) or (S, N, R) but we treat the last as time if typical length small
        # - If arr.ndim == 2, assume (S, N) -> treat as (S, N, R=1, T=1)
        shape_after = arr.shape
        print(f"[UW loader] permuted response array shape (S,...): {shape_after}")

        # Map dims:
        if arr.ndim == 4:
            # straightforward: (S, N, R, T)
            _, N, R_max, T_max = arr.shape
            # convert directly
            responses_raw = arr  # shape (S, N, R, T)
        elif arr.ndim == 3:
            # try to guess if last axis is time or repeats:
            # if last axis length <= 10: likely time; else treat as repeats with T=1
            last_len = arr.shape[2]
            if last_len <= 50:
                # assume (S, N, T) with R=1
                S_, N, T = arr.shape
                responses_raw = arr.reshape(S_, N, 1, T)
            else:
                # assume (S, N, R) with T=1
                S_, N, R = arr.shape
                responses_raw = arr.reshape(S_, N, R, 1)
        elif arr.ndim == 2:
            # (S, N) -> (S, N, R=1, T=1)
            S_, N = arr.shape
            responses_raw = arr.reshape(S_, N, 1, 1)
        else:
            raise ValueError(f"Unexpected array ndim={arr.ndim} in spike_data.nc variable '{chosen_var}'")

        # After computing responses_raw and S2, N, R_max, T_max
        S2, N, R_max, T_max = responses_raw.shape

        # Attempt to map response rows (0..S2-1) to global stim.npy indices (0..S-1).
        da = ds[chosen_var]  # xarray DataArray we used
        # Identify the name of the stimulus axis corresponding to the "first" axis after permutation
        # We permuted such that stimulus axis is first; for the DataArray the original dim name might be anywhere,
        # but da.dims[0] corresponds to the first axis of the DataArray (which we used as stimulus axis)
        stim_dim_name = da.dims[0] if hasattr(da, 'dims') and len(da.dims) > 0 else None
        mapping = None

        if stim_dim_name is not None and stim_dim_name in da.coords:
            coord_vals = da.coords[stim_dim_name].values
            # If coordinates are integers that index into stim.npy, use them
            if np.issubdtype(coord_vals.dtype, np.integer):
                # ensure values are in the valid range of stim.npy
                if coord_vals.min() >= 0 and coord_vals.max() < S:
                    mapping = coord_vals.astype(int)
                    # mapping length should be S2
                    if mapping.shape[0] != S2:
                        mapping = None

        # if mapping still None, try any coordinate of da that might contain global stim ids
        if mapping is None:
            for cname, cval in da.coords.items():
                cvals = cval.values
                if cvals.shape[0] == S2 and np.issubdtype(cvals.dtype, np.integer):
                    if cvals.min() >= 0 and cvals.max() < S:
                        mapping = cvals.astype(int)
                        print(f"[UW loader] using coordinate '{cname}' to map responses -> stim.npy indices.")
                        break

        if mapping is None:
            # No coordinate mapping found: fallback option
            print(f"[UW loader] WARNING: spike_data has {S2} rows while stim.npy has {S} images.")
            print(
                "[UW loader] No coordinate mapping found in NetCDF file. Using first S2 images from stim.npy as fallback.")
            mapping = np.arange(S2, dtype=int)

        # Build self.stims and self.stim_meta using the mapping we discovered.
        # mapping[s] = index in stim.npy that corresponds to responses_raw[s]
        self.stims = []
        self.stim_meta = []
        for s in range(S2):
            global_idx = int(mapping[s])
            # get image from stim_arr (which is shape (S, C, H, W) or similar after our preproc)
            img = torch.from_numpy(stim_arr[global_idx].astype(np.float32)).to(self.device).unsqueeze(-1)
            self.stims.append(img)
            # Use the same stim_meta convention as before: (global_idx, maybe position)
            meta = (global_idx,)
            if stim_wpos_df is not None and str(global_idx) in stim_wpos_df.index:
                meta = (global_idx, stim_wpos_df.loc[str(global_idx)].to_dict())
            self.stim_meta.append(meta)

        # Now build nrn_meta (as before)
        sub_path = os.path.join(path, 'sub.csv')
        subs = None
        if os.path.isfile(sub_path):
            subs = pd.read_csv(sub_path, index_col=0)
        for n in range(N):
            meta = (n,)
            if subs is not None and str(n) in subs.index:
                meta = (n, subs.loc[str(n)].to_dict())
            self.nrn_meta.append(meta)

        # Finally fill self.resps and self.nrn_masks using responses_raw (S2 matches len(self.stims) now)
        for s in range(S2):
            resp_list = []
            mask = []
            for n in range(N):
                arr_nt = responses_raw[s, n]  # shape (R, T)
                if np.isnan(arr_nt).all() or arr_nt.size == 0:
                    resp_list.append(torch.full((1, 1), float('nan'), dtype=torch.float32, device=self.device))
                    mask.append(False)
                else:
                    t = torch.from_numpy(arr_nt.astype(np.float32)).to(self.device)
                    resp_list.append(t)
                    mask.append(True)
            self.resps.append(resp_list)
            self.nrn_masks.append(torch.tensor(mask, dtype=torch.bool, device=self.device))

        print(f"[UW loader] finished building dataset: S={len(self.stims)}, N={len(self.nrn_meta)}")

    def __len__(self):
        return len(self.stims)

    def __getitem__(self, idx):
        # follow same semantics as CRCNS_AA4_Dataset: return item(s) for selected neurons
        # here we have no selection mechanism by default; we just return everything
        if isinstance(idx, int):
            single = True
            idxs = [idx]
        elif isinstance(idx, slice):
            idxs = list(range(*idx.indices(len(self.stims))))
            single = False
        elif isinstance(idx, (list, tuple)):
            idxs = list(idx)
            single = False
        else:
            raise TypeError("Invalid index type")
        specs = [self.stims[i] for i in idxs]
        resps = [self.resps[i] for i in idxs]
        masks = [self.nrn_masks[i] for i in idxs]
        metas = [self.stim_meta[i] for i in idxs]
        if single:
            return specs[0], resps[0], masks[0], metas[0]
        return specs, resps, masks, metas


def uw_collate(batch):
    """
    Collate for the UWChallenge dataset.
    batch: list of tuples (spec (3,H,W,1), resps [N*(R,T)], mask (N,), meta)
    Returns:
      specs: (B, 3, H, W, T_max)
      resps: (B, N, R_max, T_max)
      masks: (B, N)
      metas: list length B
    Behavior:
      - static image is duplicated along time to match the stimulus' T (max across neurons for that stimulus)
      - then images/time and responses are padded across batch to T_max and R_max
      - image padding uses 0.0, responses fill missing with NaN
    """
    specs_list, resps_list, masks_list, metas_list = zip(*batch)
    B = len(specs_list)
    N = masks_list[0].shape[0]

    # For each sample in batch, determine per-sample T_b (max time across neurons for that stim)
    per_sample_T = []
    for b in range(B):
        # each resps_list[b] is list of length N of tensors (R, T)
        T_b = 0
        for n in range(N):
            r = resps_list[b][n]
            T_b = max(T_b, r.shape[1])  # r shape could be (1,1) for missing
        per_sample_T.append(T_b if T_b > 0 else 1)

    Tmax = max(per_sample_T)

    # duplicate each static image along time to its T_b
    dup_specs = []
    for b in range(B):
        spec = specs_list[b]         # (3,H,W,1)
        T_b = per_sample_T[b]
        # repeat on last axis
        spec_dup = spec.repeat(1, 1, 1, T_b)  # (3,H,W,T_b)
        dup_specs.append(spec_dup)

    # pad duplicated specs along time to Tmax (dims index = 3)
    specs = fill_missing_data(dup_specs, dims=3, value=0.0)  # shape (B,3,H,W,Tmax)

    # build resps tensor with padding (B, N, Rmax, Tmax) filling with NaN
    Rmax = 0
    for b in range(B):
        for n in range(N):
            Rmax = max(Rmax, resps_list[b][n].shape[0])
    # use float dtype
    dtype = specs.dtype
    device = specs.device
    resps = torch.full((B, N, Rmax, Tmax), float('nan'), dtype=dtype, device=device)
    for b in range(B):
        for n in range(N):
            r = resps_list[b][n].to(device=device, dtype=dtype)  # (R, T)
            R, T = r.shape
            resps[b, n, :R, :T] = r

    masks = torch.stack(list(masks_list), dim=0)  # (B, N)
    return specs, resps, masks, list(metas_list)
