from abc import ABC
from typing import Optional, Sequence

import torch
from torch.utils.data.dataset import Dataset


# TODO:
#  - __getitem()__ --> do not return nrn_mask ?
#  - select stims by dimension (eg, R > 5, T > 1000 ms, etc.)


class NeuralDataset(Dataset, ABC):
    """General base class for datasets of sensory neural responses.

    deepSTRF datasets are **triply ragged** (variable stim duration, variable
    repeat count per (stim, neuron), sparse stim/neuron coverage). They are
    stored as Python lists of tensors, with NaN used as the single channel
    for encoding missingness. See ``docs/_source/md/data_paradigm.md`` for
    the full rationale, collate behaviour, and recommended loss pattern.

    Subclass contract
    -----------------
    A concrete subclass must populate the following attributes in its
    ``__init__`` and then call ``self.validate()`` as its last line:

    - ``self.stims``           — list of length ``S``, each element a stimulus
                                  tensor of modality-specific shape
                                  (audio: ``(1, F, T_s)``, video:
                                  ``(1, H, W, T_s)``). ``T_s`` may vary.
    - ``self.responses``       — list of length ``S``, each element itself a
                                  list of length ``N``. ``responses[s][n]`` is
                                  a ``(R_{s,n}, T_s)`` float tensor of spike
                                  counts per repeat × time bin, or a
                                  ``(1, 1)`` NaN tensor if neuron ``n`` did
                                  not hear stim ``s``.
    - ``self.stim_meta``       — list of length ``S``, per-stim metadata dicts.
    - ``self.nrn_meta`` — list of length ``N``, per-neuron metadata dicts.
    - ``self.N_neurons``       — int, must equal ``len(self.nrn_meta)``.

    Derived attributes (no explicit population needed):

    - ``self.nrn_masks`` — ``(S, N)`` bool tensor, derived on the fly from
      the NaN sentinels in ``self.responses``. ``nrn_masks[s, n]`` is
      ``True`` iff neuron ``n`` has real data for stim ``s``. Implemented
      as a ``@property`` so it is always consistent with the current
      ``self.responses`` — no risk of the mask going out of sync.

    Key invariants
    --------------
    - **Stim tensors never contain NaN.** Batch-level collate zero-pads them
      on the right along ``T``.
    - **Response tensors may contain NaN.** Use ``self.nrn_masks`` (dataset
      level) or the derived ``valid_mask`` from collate (batch level).
    - Response-side preprocessing (``smooth_responses``, ``normalize_responses``,
      any user-written transform) must be NaN-aware — either use
      ``nanmean`` / ``nanstd`` / etc., or apply the mask before reducing.
    """

    def __init__(self, path: str, dt_ms: float):
        super().__init__()
        self.path = path
        self.dt = dt_ms

        # core deepSTRF dataset attributes (subclass populates)
        self.responses = []
        self.stims = []
        self.stim_meta = []
        self.nrn_meta = []
        self.N_neurons = 0

        # Lazy cache for ``nrn_masks`` — see the @property. Computed on
        # first access and reused thereafter. Subclasses (or callers
        # that structurally mutate ``self.responses``) should call
        # ``self._invalidate_nrn_masks()`` to force recomputation.
        self._nrn_masks_cache = None

        # selected-neuron indices (defaults to empty; filled by select_* or on first __getitem__)
        self.I = []
        # selected-stim indices: None == no restriction, [] == explicit empty.
        # The asymmetry with self.I is intentional: explicit zero-stim selection
        # (e.g. ``select_stims_by_attr`` on an attribute no stim has) must
        # yield zero items rather than silently disabling the filter.
        self.S_sel = None

    def get_N(self):
        """Return the total number of selectable neurons."""
        return self.N_neurons

    def get_S(self):
        """Return the total number of stimuli presented to the whole neural population."""
        return len(self.stim_meta)

    def get_nrn_meta(self):
        """Retrieve metadata for each currently selected neuron."""
        return [self.nrn_meta[i] for i in self.I]

    @property
    def nrn_masks(self) -> torch.Tensor:
        """Derived ``(S, N)`` bool tensor: True iff neuron n has real data for stim s.

        Derived from the NaN sentinels in ``self.responses`` — single
        source of truth, cannot go out of sync. Lazy-cached on first
        access: subsequent reads are O(1). Callers that *structurally*
        mutate the response list (replace a real tensor with a
        ``(1, 1)`` NaN sentinel, or vice versa) should call
        ``self._invalidate_nrn_masks()`` afterwards. Shape-preserving
        mutations (``smooth_responses``, normalization, etc.) leave
        the mask unchanged and do not require invalidation.

        For a bare dataset (no populated responses), returns an empty
        ``(0, N_neurons)`` tensor.
        """
        if self._nrn_masks_cache is None:
            self._nrn_masks_cache = self._compute_nrn_masks()
        return self._nrn_masks_cache

    def _compute_nrn_masks(self) -> torch.Tensor:
        """Build the ``(S, N)`` mask from ``self.responses``.

        Fast path: the deepSTRF data paradigm guarantees the ``(1, 1)``
        shape *is* the canonical missing-sentinel marker, and that
        non-missing responses do not contain NaN. So the per-cell check
        collapses to a Python shape comparison — no tensor ops, no
        ``.item()`` syncs.

        Defensive fallback: if a non-``(1, 1)`` tensor does contain NaN
        (legacy data, user-injected NaN), we still treat that cell as
        missing for that stim, matching the original semantics.
        """
        S = len(self.responses)
        if S == 0:
            return torch.zeros((0, self.N_neurons), dtype=torch.bool)
        mask = torch.zeros((S, self.N_neurons), dtype=torch.bool)
        for s in range(S):
            for n in range(self.N_neurons):
                r = self.responses[s][n]
                if tuple(r.shape) == (1, 1):
                    mask[s, n] = False
                elif r.isnan().any():
                    mask[s, n] = False
                else:
                    mask[s, n] = True
        return mask

    def _invalidate_nrn_masks(self) -> None:
        """Drop the cached ``nrn_masks``; next access recomputes.

        Call after structurally mutating ``self.responses`` (e.g.
        swapping a real tensor for a ``(1, 1)`` NaN sentinel, or vice
        versa). The shape-preserving response transforms shipped with
        the library (``smooth_responses``, ``standardize_stims``)
        cannot change ``nrn_masks`` and do not invalidate the cache.
        """
        self._nrn_masks_cache = None

    def _selected_stims(self) -> list:
        """Effective stim selection. Falls back to all stims when ``self.S_sel`` is None."""
        if self.S_sel is None:
            return list(range(len(self.stim_meta)))
        return list(self.S_sel)

    def _selected(self) -> list:
        """Effective neuron selection.

        Bidirectional rule: when ``self.S_sel`` is set, neurons with no valid
        response across *any* of the currently-selected stimuli are also
        hidden. So a user who selects ``subset='val'`` on NAT4 will not see
        cells that lack val data at all — those cells' only data lies
        outside the selected stim subset and would yield NaN-only batches.

        Falls back to all neurons when ``self.I`` is empty (modulo the stim
        cross-filter above).
        """
        base = list(self.I) if self.I else list(range(self.N_neurons))
        if self.S_sel is None:
            return base
        s_idxs = self._selected_stims()
        if not s_idxs:
            # explicit empty stim selection -> empty neuron selection too
            return []
        masks = self.nrn_masks
        if masks.shape[0] == 0:
            return []
        # neuron is kept iff it has >=1 valid response within the selected stims
        return [n for n in base if masks[s_idxs, n].any().item()]

    @property
    def _iter_idx(self) -> list:
        """Stim indices visible to iteration: those with >=1 valid response among selected neurons,
        intersected with ``self.S_sel`` if that's set.

        This is the canonical filter that ``__len__`` and ``__getitem__`` agree on.
        For a concatenated dataset, selecting only one source's neurons makes
        only that source's stims iterable — cross-block stims (full-NaN for
        the selected neurons) are filtered out automatically.
        """
        masks = self.nrn_masks  # (S, N) bool
        if masks.shape[0] == 0:
            return []
        sel = self._selected()
        if not sel:
            return []
        sel_mask = masks[:, sel].any(dim=1)  # (S,) bool
        candidate = sel_mask.nonzero(as_tuple=True)[0].tolist()
        if self.S_sel is None:
            return candidate
        allowed = set(self.S_sel)
        return [s for s in candidate if s in allowed]

    def __len__(self):
        """Number of stimuli with at least one valid response among the currently selected neurons."""
        return len(self._iter_idx)

    def __getitem__(self, idx):
        """Retrieve stimulus-response pairs for the iteration index ``idx`` (or indices).

        ``idx`` indexes into the *iterable* stim space — the subset of stimuli
        for which at least one currently-selected neuron has valid response
        data. This makes ``__getitem__`` consistent with ``__len__`` and with
        PyTorch ``DataLoader`` semantics (which iterates ``range(len(ds))``).

        Returns
        -------
        stim, responses, mask, stim_meta
            Tuple for a scalar index, or 4 lists for slice / list indexing.
            ``responses`` and ``mask`` are restricted to the selected neurons.
        """
        iter_idx = self._iter_idx  # snapshot once; O(S * |I|) per call

        if isinstance(idx, int):
            n_iter = len(iter_idx)
            i = idx + n_iter if idx < 0 else idx
            if not (0 <= i < n_iter):
                raise IndexError(
                    f"Dataset index {idx} out of range (len={n_iter} under current selection)"
                )
            indices = [iter_idx[i]]
            single = True
        elif isinstance(idx, slice):
            indices = [iter_idx[i] for i in range(*idx.indices(len(iter_idx)))]
            single = False
        elif isinstance(idx, (list, tuple)):
            indices = [iter_idx[i] for i in idx]
            single = False
        else:
            raise TypeError(f"Invalid index type {type(idx)}")

        selected = self._selected()
        all_masks = self.nrn_masks   # snapshot property once
        stims = [self.stims[i] for i in indices]
        metas = [self.stim_meta[i] for i in indices]
        resps = [[self.responses[i][n] for n in selected] for i in indices]
        masks = [all_masks[i][selected] for i in indices]

        if single:
            return stims[0], resps[0], masks[0], metas[0]
        return stims, resps, masks, metas

    def __repr__(self):
        return (f"{self.__class__.__name__}(N_neurons={self.get_N()}, "
                f"selected={len(self.I)}, N_stims={len(self.stim_meta)}, dt_ms={self.dt})")

    def __str__(self):
        return self.__repr__()

    # neural population selection API (manual)
    def select_neuron(self, neuron_index: int):
        assert isinstance(neuron_index, int) and 0 <= neuron_index < self.N_neurons, \
            f"neuron_index must be in [0, {self.N_neurons})"
        self.I = [neuron_index]

    def select_population(self, neuron_indices):
        for neuron_index in neuron_indices:
            assert isinstance(neuron_index, int) and 0 <= neuron_index < self.N_neurons, \
                f"neuron_index must be in [0, {self.N_neurons})"
        self.I = list(neuron_indices)

    # neural population selection API (advanced)
    def select_pop_by_nrn_attr(self, attribute_name: str, value):
        """Select neurons whose ``nrn_meta[attribute_name] == value``.

        Neurons whose metadata dict does not contain ``attribute_name`` are
        silently skipped — this lets a single filter call work against a
        concatenated dataset that pools sources with different metadata
        schemas (e.g. AA1's ``area`` is not present on AA4 neurons; calling
        ``select_pop_by_nrn_attr("area", "Field_L")`` on the concatenation
        keeps only AA1 neurons in Field L, with no ``KeyError``).

        TODO:
         - allow multiple conditions (AND / OR), eg with attribute_name and value as lists
            ==> additional argument? "and", "or"
        """
        _MISSING = object()
        selected_nrn_indices = []
        for n, nrn_metadata in enumerate(self.nrn_meta):
            if nrn_metadata.get(attribute_name, _MISSING) == value:
                selected_nrn_indices.append(n)
        self.I = selected_nrn_indices
        return selected_nrn_indices

    def select_pop_by_stim_attr(self, attribute_name: str, value):
        """Select neurons with >=1 non-null response to stimuli matching a given attribute.

        Looks up stimuli whose ``stim_meta[attribute_name] == value`` and
        keeps only neurons whose ``nrn_masks`` is True for at least one of
        them. Stims missing the key are silently skipped (same convention
        as :meth:`select_pop_by_nrn_attr`).
        """
        _MISSING = object()
        s_idxs = [s for s, sm in enumerate(self.stim_meta)
                  if sm.get(attribute_name, _MISSING) == value]
        if not s_idxs:
            self.I = []
            return []
        masks = self.nrn_masks
        selected = [n for n in range(self.N_neurons) if masks[s_idxs, n].any().item()]
        self.I = selected
        return selected

    # stim selection API
    def select_stim(self, stim_index: int):
        """Restrict iteration to a single stimulus index.

        Pairs with the bidirectional rule in :meth:`_selected`: cells whose
        only valid responses lie outside the selected stim are auto-hidden
        from ``__getitem__``.
        """
        assert isinstance(stim_index, int) and 0 <= stim_index < len(self.stim_meta), \
            f"stim_index must be in [0, {len(self.stim_meta)})"
        self.S_sel = [stim_index]

    def select_stims(self, stim_indices):
        """Restrict iteration to the listed stimulus indices."""
        for s in stim_indices:
            assert isinstance(s, int) and 0 <= s < len(self.stim_meta), \
                f"stim_index must be in [0, {len(self.stim_meta)})"
        self.S_sel = list(stim_indices)

    def select_stims_by_attr(self, attribute_name: str, value):
        """Restrict iteration to stimuli matching ``stim_meta[attr] == value``.

        Stims whose metadata dict does not contain ``attribute_name`` are
        silently skipped — same convention as :meth:`select_pop_by_nrn_attr`,
        so a single call works on a concatenated dataset whose sources have
        heterogeneous stim metadata schemas.

        Returns the selected stim indices.
        """
        _MISSING = object()
        selected = [s for s, sm in enumerate(self.stim_meta)
                    if sm.get(attribute_name, _MISSING) == value]
        self.S_sel = selected
        return selected

    def reset_stim_selection(self):
        """Clear ``self.S_sel`` so all stimuli are eligible again."""
        self.S_sel = None

    def reset_pop_selection(self):
        """Clear the population selection so all neurons are eligible again.

        Mirror of ``reset_stim_selection``. Restores ``self.I`` to its
        empty default (interpreted as "no neuron-side restriction" by
        ``_selected()``).
        """
        self.I = []

    def standardize_stims(self, stim_indices: Optional[Sequence[int]] = None,
                          per_band: bool = True, eps: float = 1e-8) -> dict:
        """Standardize ``self.stims`` in place: ``(x − mean) / std``.

        Statistics are computed over the stims selected by
        ``stim_indices`` (typically train + validation indices) and
        applied to **all** stims in the dataset — so the held-out test
        stims are automatically transformed with the same train+val
        statistics, preventing leakage of test-set first-order moments
        into the standardisation while still ensuring train / val / test
        all live in the same standardised space.

        Parameters
        ----------
        stim_indices : sequence of int, optional
            Indices of stims to compute statistics from. If None
            (default), statistics are computed over **all** stims —
            equivalent to "no held-out test set"; useful for
            single-split exploratory analysis but introduces a tiny
            (first-order) leakage if a test set is held out downstream.
        per_band : bool, default True
            If True, statistics are per-frequency-band (axis ``-2``):
            mean / std are tensors of shape broadcastable to
            ``(C, F, 1)``. If False, a single scalar mean and std are
            computed over the whole concatenated stim tensor.
        eps : float, default 1e-8
            Floor on ``std`` to avoid division by zero on constant bands.

        Returns
        -------
        dict
            ``{'mean': Tensor, 'std': Tensor, 'per_band': bool,
            'stim_indices': list | None}`` — also stored on
            ``self.stim_normalization`` for inspection (e.g. to fold
            into a model kernel for STRF visualisation).

        Notes
        -----
        Not idempotent: calling twice double-standardizes. To re-do with
        different statistics, rebuild the dataset.
        """
        sub = (self.stims if stim_indices is None
               else [self.stims[i] for i in stim_indices])
        if not sub:
            raise ValueError("standardize_stims: no stims to compute statistics from")
        cat = torch.cat(sub, dim=-1)
        if per_band:
            # cat shape: (C, F, T_total) for audio. Reduce over (T_total,)
            # and the leading (C,) channel dim, keep F.
            non_F_dims = tuple(d for d in range(cat.dim()) if d != cat.dim() - 2)
            mean = cat.mean(dim=non_F_dims, keepdim=True)
            std = cat.std(dim=non_F_dims, keepdim=True).clamp(min=eps)
        else:
            mean = cat.mean()
            std = cat.std().clamp(min=eps)
        self.stims = [(s - mean) / std for s in self.stims]
        self.stim_normalization = {
            'mean': mean.detach().clone(),
            'std': std.detach().clone(),
            'per_band': per_band,
            'stim_indices': (list(stim_indices) if stim_indices is not None else None),
        }
        return self.stim_normalization

    def smooth_responses(self, window_ms: float = 21.0) -> None:
        """Temporally smooth each non-NaN response in place with a Hanning window.

        Parameters
        ----------
        window_ms : float, default 21.0
            Full width of the Hanning window in ms. Rounded to the nearest odd
            number of ``self.dt`` bins.

        Notes
        -----
        Follows Hsu, Borst & Theunissen (2004) for reducing PSTH estimator
        variance — a common preprocessing step across spike-count datasets.
        ``(1, 1)`` NaN-sentinel responses (neurons that did not hear a given
        stim) are preserved unchanged.
        """
        # lazy import to avoid a circular dep (utils.data imports from datasets)
        from deepSTRF.utils.data import hanning_smooth

        for s in range(len(self.responses)):
            for n in range(self.N_neurons):
                r = self.responses[s][n]
                if r.isnan().any():
                    continue
                self.responses[s][n] = hanning_smooth(r, window_ms=window_ms, dt_ms=self.dt)

    def normalize_responses(self, method: str = "max",
                            stim_indices: Optional[Sequence[int]] = None,
                            eps: float = 1e-8) -> dict:
        """Normalize ``self.responses`` in place, per neuron.

        Statistics are computed on a chosen stim subset (typically train
        +val) and applied to **all** stims, mirroring ``standardize_stims``.
        ``(1, 1)`` NaN sentinels for structurally missing
        ``(stim, neuron)`` pairs are preserved unchanged.

        Parameters
        ----------
        method : {'max', 'zscore'}, default 'max'
            'max' — divide each neuron's responses by their max across all
            ``(s, r, t)`` in ``stim_indices``. Preserves non-negativity;
            range becomes ``[0, max] -> [0, 1]``. Natural for rate-coded /
            spike-count targets where 0 is meaningful.

            'zscore' — subtract per-neuron mean, divide by per-neuron std,
            both computed NaN-aware over the same flat ``(s, r, t)``
            samples. Maps signed continuous targets (EEG, LFP) to ``N(0, 1)``.
        stim_indices : sequence of int, optional
            Stims used to compute statistics. If None, all stims are used.
        eps : float, default 1e-8
            Floor on the divisor.

        Returns
        -------
        dict
            ``{'method': str, 'scale': Tensor[N], 'offset': Tensor[N],
            'stim_indices': list | None}`` — also stored on
            ``self.response_normalization``. ``scale`` is the divisor
            (max or std); ``offset`` is the subtracted location (0 for
            'max', mean for 'zscore').

        Notes
        -----
        Not idempotent — calling twice double-normalizes.
        """
        if method not in ("max", "zscore"):
            raise ValueError(f"method must be 'max' or 'zscore', got {method!r}")
        sub_idx = (list(range(len(self.responses)))
                   if stim_indices is None else list(stim_indices))

        offset = torch.zeros(self.N_neurons)
        scale = torch.ones(self.N_neurons)
        for n in range(self.N_neurons):
            chunks = []
            for s in sub_idx:
                r = self.responses[s][n]
                if r.shape == (1, 1) and torch.isnan(r).all():
                    continue
                chunks.append(r.flatten())
            if not chunks:
                continue
            cat = torch.cat(chunks)
            valid = cat[~torch.isnan(cat)]
            if valid.numel() < 2:
                continue
            if method == "max":
                scale[n] = float(valid.abs().max().clamp(min=eps))
            else:  # zscore
                offset[n] = float(valid.mean())
                scale[n] = float(valid.std().clamp(min=eps))

        for s in range(len(self.responses)):
            for n in range(self.N_neurons):
                r = self.responses[s][n]
                if r.shape == (1, 1) and torch.isnan(r).all():
                    continue
                self.responses[s][n] = (r - offset[n]) / scale[n]

        self.response_normalization = {
            "method": method,
            "scale": scale.detach().clone(),
            "offset": offset.detach().clone(),
            "stim_indices": (list(stim_indices)
                             if stim_indices is not None else None),
        }
        return self.response_normalization

    def __add__(self, other):
        """Concatenate two datasets on BOTH the stim and neuron axes (sugar for :func:`deepSTRF.utils.data.concat_neural_datasets`).

        Returns a new dataset with ``S_a + S_b`` stimuli and ``N_a + N_b``
        neurons. Cross-block ``(stim_a, neuron_b)`` and ``(stim_b, neuron_a)``
        responses are filled with the canonical ``(1, 1)`` NaN sentinel —
        ``nrn_masks`` (derived property) then reflects the block-diagonal
        coverage automatically. See ``concat_neural_datasets`` for the full
        semantics and ``_concat_check_compat`` for per-modality compatibility
        requirements.
        """
        if not isinstance(other, NeuralDataset):
            return NotImplemented
        # lazy import to avoid circular dep with utils.data
        from deepSTRF.utils.data import concat_neural_datasets
        return concat_neural_datasets([self, other])

    def _concat_check_compat(self, other: "NeuralDataset") -> None:
        """Assert that ``other`` is compatible for concatenation with ``self``.

        Subclasses should call ``super()._concat_check_compat(other)`` and
        then add their own checks (e.g. ``AudioNeuralDataset`` checks that
        ``self.F == other.F``).
        """
        assert self.dt == other.dt, \
            f"dt mismatch: {self.dt} vs {other.dt}. Resample responses to a common bin width before concatenating."

    def _concat_copy_attrs(self, source: "NeuralDataset") -> None:
        """Copy modality-specific attributes from ``source`` onto ``self``.

        Called by ``concat_neural_datasets`` on the bare result instance after
        the merged core attributes (``stims``, ``responses``, ``stim_meta``,
        ``nrn_meta``, ``N_neurons``, ``dt``, ``I``, ``path``) are set.
        Subclasses override to propagate things like ``self.F`` (audio) or
        ``self.H, self.W`` (video). Base implementation is a no-op.
        """
        # base has no modality-specific attributes to copy
        pass

    def validate(self):
        """Check that the instance is deepSTRF-compatible.

        Subclasses should call ``super().validate()`` and then add their own checks
        (e.g. ``AudioNeuralDataset`` checks ``self.F > 0``).
        """
        assert isinstance(self.path, str), "self.path must be a str"
        assert isinstance(self.dt, (int, float)) and self.dt > 0, \
            f"self.dt must be a positive number (got {self.dt!r})"
        assert isinstance(self.N_neurons, int) and self.N_neurons > 0, \
            f"self.N_neurons must be a positive int (got {self.N_neurons!r})"

        S = len(self.stim_meta)
        assert S > 0, "self.stim_meta must be non-empty"
        assert len(self.stims) == S, \
            f"len(self.stims) ({len(self.stims)}) must equal len(self.stim_meta) ({S})"
        assert len(self.responses) == S, \
            f"len(self.responses) ({len(self.responses)}) must equal len(self.stim_meta) ({S})"
        assert len(self.nrn_meta) == self.N_neurons, (
            f"len(self.nrn_meta) ({len(self.nrn_meta)}) "
            f"must equal self.N_neurons ({self.N_neurons})"
        )

        # self.nrn_masks is a derived @property; its shape and dtype are
        # guaranteed by construction, so no separate check is needed.
