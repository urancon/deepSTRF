import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from deepSTRF.models.neural_model import NeuralModel


class AudioEncodingModel(NeuralModel):
    """
    Base class for encoding models of audio neural responses.

    Forward signature: input ``(B, 1, F, T)`` spectrogram → output
    ``(B, N, R=1, T)`` neural activity. Concrete subclasses populate
    the four canonical slots ``wav2spec`` / ``prefiltering`` / ``core``
    / ``readout`` (see :class:`NeuralModel`).

    Parameters
    ----------
    n_frequency_bands : int
        Number of input spectrogram frequency bands ``F``.
    temporal_window_size : int
        STRF temporal extent ``T`` in frames. Used by ``STRF_gradmap``
        to size the null stimulus.
    out_neurons : int, default 1
        Number of output neurons ``N``.
    prefiltering : nn.Module, optional
        Optional spectrogram prefilter (e.g. ``AdapTrans``,
        ``ICAdaptation``). Must expose an ``out_channels`` integer
        attribute so the model can size ``C_in`` automatically. ``None``
        (default) gives ``nn.Identity()`` and ``C_in = 1``.
    wav2spec : nn.Module, optional
        Optional raw-waveform front-end. Maps a mono waveform
        ``(B, 1, T_audio)`` to a spectrogram ``(B, 1, F, T_neural)`` and
        slots in at the top of the canonical forward pipeline (see
        :class:`~deepSTRF.models.neural_model.NeuralModel`). Must expose an
        ``out_channels: int`` attribute equal to ``n_frequency_bands``.
        Pair with a dataset in waveform mode (e.g.
        ``NS1Dataset(return_waveform=True)``). ``None`` (default) keeps the
        slot as ``nn.Identity()`` — the model then expects spectrogram
        input ``(B, 1, F, T)`` as before.
    """

    def __init__(self, n_frequency_bands: int, temporal_window_size: int,
                 out_neurons: int = 1, prefiltering: nn.Module = None,
                 wav2spec: nn.Module = None,
                 *args, **kwargs):
        super().__init__(out_neurons=out_neurons, *args, **kwargs)

        # general attributes for AUDIO response models
        self.F = n_frequency_bands
        self.T = temporal_window_size

        # wav2spec: optional raw-waveform front-end (defaults to Identity).
        if wav2spec is not None:
            assert isinstance(wav2spec, nn.Module), \
                f"wav2spec must be an nn.Module instance, got {type(wav2spec).__name__}"
            assert getattr(wav2spec, 'out_channels', None) == n_frequency_bands, (
                f"wav2spec.out_channels ({getattr(wav2spec, 'out_channels', None)}) "
                f"must equal n_frequency_bands ({n_frequency_bands})"
            )
            self.wav2spec = wav2spec

        # prefiltering: an nn.Module exposing out_channels (int)
        if prefiltering is None:
            self.prefiltering = nn.Identity()
            self.C_in = 1
        else:
            assert isinstance(prefiltering, nn.Module), \
                f"prefiltering must be an nn.Module instance, got {type(prefiltering).__name__}"
            self.prefiltering = prefiltering
            self.C_in = getattr(prefiltering, 'out_channels', 1)

    def validate(self):
        super().validate()
        assert isinstance(self.F, int) and self.F > 0, \
            f"self.F must be a positive int (got {self.F!r})"
        assert isinstance(self.T, int) and self.T > 0, \
            f"self.T must be a positive int (got {self.T!r})"
        assert isinstance(self.C_in, int) and self.C_in >= 1, \
            f"self.C_in must be a positive int (got {self.C_in!r})"

    def STRF_gradmap(self, T: int = None):
        """
        Compute one STRF gradient map per output neuron in parallel.

        For each of the ``N`` output neurons, finds the changes in a
        null spectrogram that elicit an increase in that neuron's
        activity at the last timestep (a Spike-Triggered-Average-like
        readout, computed by autodiff). The batch dimension is used to
        parallelize across neurons in a single forward / backward pass.

        Parameters
        ----------
        T : int, optional
            Time-axis length of the null stimulus. Defaults to
            ``self.T``.

        Returns
        -------
        Tensor of shape ``(N, 1, F, T)``
            Per-neuron gradient map.

        References
        ----------
        Rançon et al. (2025), "Temporal recurrence as a general
        mechanism to explain neural responses in the auditory system."

        Notes
        -----
        Future work:

        - Handle multi-channel inputs (the gradient is currently shaped
          ``(N, 1, F, T)`` regardless of ``C_in``; an AdapTrans-prefiltered
          model has ``C_in == 2`` and the per-channel gradients differ).
        - Allow custom losses (e.g. sustained activity rather than
          last-timestep-only).
        """
        B = self.O      # use the batch dimension to parallelize across neurons

        # initial stim = null stimulus = absence of bias / no information.
        # Place it on whatever device the model lives on so this works
        # regardless of whether the user moved the model to CUDA.
        device = next(self.parameters()).device
        T_eff = T if T is not None else self.T
        stim_opt = Parameter(torch.zeros(B, 1, self.F, T_eff, device=device),
                             requires_grad=True)

        # forward pass — output is (B=N, N, R=1, T_eff). We temporarily mask
        # any ``wav2spec`` slot to Identity because the null stimulus IS
        # already at the spectrogram-output rank; what we want is the model's
        # STRF in the space of *spectrogram inputs*, regardless of whether
        # the production model takes waveform or spectrogram input. Swap (not
        # bypass) so that subclasses with a custom ``forward`` (Transformer,
        # StateNet, etc.) still go through their own pipeline.
        saved_wav2spec = self.wav2spec
        self.wav2spec = nn.Identity()
        try:
            response = self.forward(stim_opt)
        finally:
            self.wav2spec = saved_wav2spec

        # Spike-Triggered-Average loss = sum of diagonal activations at last
        # timestep. response[:, :, 0, -1] is (N, N): row b is the prediction
        # for all N output neurons given the b-th batched null stim; the
        # diagonal picks each row's matching neuron.
        loss = - torch.trace(response[:, :, 0, -1])

        # backward pass populates stim_opt.grad
        loss.backward()

        return stim_opt.grad
