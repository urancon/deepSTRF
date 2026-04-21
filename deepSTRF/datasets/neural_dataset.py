from abc import ABC

import torch
from torch.utils.data.dataset import Dataset


# TODO:
#  - __getitem()__ --> do not return nrn_mask ?
#  - method to select neural population by presented stims (either stim indices or stim metadata)
#  - select stims by dimension (eg, R > 5, T > 1000 ms, etc.)


class NeuralDataset(Dataset, ABC):
    """General mother class for handling datasets of sensory neural responses.

    Subclasses must populate the following attributes in their ``__init__``,
    then call ``self.validate()`` as the last line:
        - ``self.stims``          — list or tensor of stimuli (modality-specific shape)
        - ``self.responses``      — list of lists of response tensors, indexed [s][n]
        - ``self.stim_meta``      — list of per-stimulus metadata (tuple or dict)
        - ``self.neuron_metadata`` — list of per-neuron metadata (dict or tuple)
        - ``self.N_neurons``      — total number of neurons

    After populating these, the subclass should call ``self.compute_nrn_masks()``
    (which fills ``self.nrn_masks``) before iteration.
    """

    def __init__(self, path: str, dt_ms: float):
        super().__init__()
        self.path = path
        self.dt = dt_ms

        # core deepSTRF dataset attributes (subclass populates)
        self.responses = []
        self.stims = []
        self.stim_meta = []
        self.neuron_metadata = []
        self.N_neurons = 0

        # selected-neuron indices (defaults to empty; filled by select_* or on first __getitem__)
        self.I = []

        # boolean mask (S, N) of which neuron saw which stimulus; populated by compute_nrn_masks()
        self.nrn_masks = None

    def get_N(self):
        """Return the total number of selectable neurons."""
        return self.N_neurons

    def get_S(self):
        """Return the total number of stimuli presented to the whole neural population."""
        return len(self.stim_meta)

    def get_neuron_metadata(self):
        """Retrieve metadata for each currently selected neuron."""
        return [self.neuron_metadata[i] for i in self.I]

    def __len__(self):
        """Return number of stimuli for which at least one SELECTED neuron has a valid response."""
        if self.nrn_masks is None:
            raise RuntimeError(
                "nrn_masks is not populated. Call self.compute_nrn_masks() at the end of __init__."
            )
        if not self.I:
            return 0
        count = 0
        for mask in self.nrn_masks:
            if any(mask[i].item() for i in self.I):
                count += 1
        return count

    def __getitem__(self, idx):
        """Retrieve stimulus-response pairs for given stimulus index or indices,
        only for selected neurons (self.I).

        Args:
            idx (int, slice, list of int): stimulus index or indices.

        Returns:
            stim: Tensor or list of Tensors
            responses: list or list of lists of Tensors [(R, T)], only for neurons in self.I
            nrn_masks: Tensor or list of Tensors [(len(self.I),)]
            stim_meta: metadata or list of metadata
        """
        if self.nrn_masks is None:
            raise RuntimeError(
                "nrn_masks is not populated. Call self.compute_nrn_masks() at the end of __init__."
            )

        if isinstance(idx, int):
            indices = [idx]
            single = True
        elif isinstance(idx, slice):
            indices = list(range(*idx.indices(len(self.stims))))
            single = False
        elif isinstance(idx, (list, tuple)):
            indices = list(idx)
            single = False
        else:
            raise TypeError(f"Invalid index type {type(idx)}")

        # default: select entire population if no explicit selection
        if not self.I:
            self.I = list(range(self.N_neurons))

        stims = [self.stims[i] for i in indices]
        metas = [self.stim_meta[i] for i in indices]
        resps = []
        masks = []
        for i in indices:
            all_resps = self.responses[i]
            all_mask = self.nrn_masks[i]
            resps.append([all_resps[n] for n in self.I])
            masks.append(all_mask[self.I])

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
        """Select neurons whose ``neuron_metadata[attribute_name] == value``.

        TODO:
         - allow multiple conditions (AND / OR), eg with attribute_name and value as lists
            ==> additional argument? "and", "or"
        """
        selected_nrn_indices = []
        for n, nrn_metadata in enumerate(self.neuron_metadata):
            if nrn_metadata[attribute_name] == value:
                selected_nrn_indices.append(n)
        self.I = selected_nrn_indices
        return selected_nrn_indices

    def select_pop_by_stim_attr(self, attribute_name: str, value):
        """Select neurons with ≥1 non-null response to stimuli matching a given attribute.

        TODO:
         - implement first version
         - allow multiple conditions (AND / OR), eg with attribute_name and value as lists
        """
        raise NotImplementedError

    # TODO: method to select stims --> getitem() will only give out these stims

    def compute_nrn_masks(self):
        """Compute the ``self.nrn_masks`` attribute:
        a ``(S, N)`` boolean tensor with False when stimulus s was not presented to neuron n.
        """
        nrn_masks = []
        for s in range(len(self.stim_meta)):
            temp_mask = []
            for n in range(len(self.neuron_metadata)):
                if not self.responses[s][n].isnan().any():
                    temp_mask.append(torch.ones(1).bool())
                else:
                    temp_mask.append(torch.zeros(1).bool())
            nrn_masks.append(torch.cat(temp_mask))   # (N,)
        self.nrn_masks = torch.stack(nrn_masks)      # (S, N)

    def smooth_responses(self, window_length: int):
        """Temporally smooth the neural responses with a Hanning window."""
        # TODO: implement (window_length in ms, we already have self.dt)
        raise NotImplementedError

    def normalize_responses(self):
        """Normalize each neuron's activity so that its maximum PSTH across stimuli is 1."""
        # TODO
        raise NotImplementedError

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
        assert len(self.neuron_metadata) == self.N_neurons, (
            f"len(self.neuron_metadata) ({len(self.neuron_metadata)}) "
            f"must equal self.N_neurons ({self.N_neurons})"
        )

        if self.nrn_masks is not None:
            assert self.nrn_masks.dtype == torch.bool, \
                f"self.nrn_masks must be a bool tensor (got dtype {self.nrn_masks.dtype})"
            assert tuple(self.nrn_masks.shape) == (S, self.N_neurons), (
                f"self.nrn_masks shape {tuple(self.nrn_masks.shape)} "
                f"must be (S={S}, N={self.N_neurons})"
            )
