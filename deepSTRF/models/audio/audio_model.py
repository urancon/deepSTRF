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

    def waveform_gradmap(self, stimulus, neuron=None, reduce='last'):
        """Gradient of a neuron's response w.r.t. the input **waveform**.

        The waveform-domain analogue of :meth:`STRF_gradmap`: instead of
        probing the *spectrogram* input, backprop a neuron's response all the
        way through the (learnable) ``wav2spec`` front-end to the raw audio
        samples. The returned gradient is itself a waveform — a listenable,
        time-domain receptive field — only defined for wav-native models (a
        non-``Identity`` ``wav2spec``).

        Parameters
        ----------
        stimulus : array-like, shape (T_audio,), (1, T_audio) or (1, 1, T_audio)
            Audio to compute the gradmap around (e.g. a real stimulus). A real
            stimulus is recommended over silence — adaptive front-ends (PCEN)
            are ill-conditioned at zero energy.
        neuron : int, optional
            Which output neuron. ``None`` (default) sums over all neurons
            (a population gradmap).
        reduce : {'last', 'peak', 'sum'}, default 'last'
            How to reduce the neuron's response over time before backprop.
            ``'last'`` (default, matching :meth:`STRF_gradmap`) maximizes the
            activation at the **last timestep**, so the gradient is supported
            only within the receptive field before it — it reveals the RF and
            decays to ~zero further into the past. ``'peak'`` does the same at
            the peak-response timestep. ``'sum'`` time-integrates over *all*
            output timesteps, which makes the gradient non-zero almost
            everywhere by construction (a whole-stimulus saliency map, **not**
            a receptive field).

        Returns
        -------
        torch.Tensor, shape ``(T_audio,)``
            Per-audio-sample gradient — the waveform-domain receptive field.
            Computed in ``eval`` mode (the strictly-causal inference regime).
            With ``reduce='last'`` the support is the RF length (e.g. ~45 ms
            for a T=9 STRF on a mel front-end; longer for adaptive front-ends
            like LEAF, whose PCEN smoother adds a decaying temporal memory).
        """
        if isinstance(self.wav2spec, nn.Identity):
            raise RuntimeError(
                "waveform_gradmap requires a waveform front-end (a non-Identity "
                "wav2spec); for spectrogram-input models use STRF_gradmap.")

        device = next(self.parameters()).device
        x = torch.as_tensor(stimulus, dtype=torch.float32, device=device)
        while x.dim() < 3:
            x = x.unsqueeze(0)                       # -> (1, 1, T_audio)
        if x.dim() != 3 or tuple(x.shape[:2]) != (1, 1):
            raise ValueError(
                "stimulus must be (T_audio,), (1, T_audio) or (1, 1, T_audio); "
                f"got {tuple(x.shape)}")
        x = x.detach().clone().requires_grad_(True)

        was_training = self.training
        self.eval()
        try:
            y = self.forward(x)                      # (1, N, 1, T)
            resp = y[0, :, 0, :] if neuron is None else y[0, neuron:neuron + 1, 0, :]
            if reduce == 'sum':
                obj = resp.sum()
            elif reduce == 'last':
                obj = resp[..., -1].sum()
            elif reduce == 'peak':
                obj = resp.max(dim=-1).values.sum()
            else:
                raise ValueError(f"reduce must be 'sum', 'last' or 'peak' (got {reduce!r})")
            self.zero_grad(set_to_none=True)
            obj.backward()
        finally:
            if was_training:
                self.train()

        return x.grad.detach().reshape(-1)           # (T_audio,)
