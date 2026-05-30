from typing import Optional

import torch

from deepSTRF.datasets.neural_dataset import NeuralDataset


class AudioNeuralDataset(NeuralDataset):
    """Neural dataset class for auditory stimuli.

    Stim shape is polymorphic depending on the loading mode:

    - **Spectrogram mode** (default): ``self.stims[s]`` is a ``(1, F, T)``
      tensor where ``F = self.F`` is the frequency-band count and ``T`` is the
      neural time-bin count.
    - **Waveform mode** (opt-in, subclass-specific): ``self.stims[s]`` is a
      ``(1, T_audio)`` mono float32 tensor at sample rate ``self.audio_fs``.
      Subclasses that support this mode expose a ``return_waveform=True``
      constructor flag and set ``self.audio_fs`` to a positive int. The
      ``(1, ...)`` leading dim is the mono-channel axis, kept for collate
      compatibility (``neural_collate`` zero-pads the last axis only).

    Subclasses must additionally set ``self.F`` (number of frequency bins in
    the spectrogram — kept positive even in waveform mode so downstream models
    know the *target* spectrogram width a ``wav2spec`` module should produce)
    in their ``__init__``, before calling ``self.validate()``.

    Attributes
    ----------
    F : int
        Frequency-band count of the target spectrogram. Set by the subclass.
    audio_fs : int or None
        Sample rate of the raw waveform when in waveform mode; ``None``
        otherwise. Subclasses without a waveform branch leave this ``None``.
    hearing_range_hz : tuple of float or None
        Optional ``(low, high)`` informational bound on the species' canonical
        hearing range in Hz (e.g. ``(200.0, 40000.0)`` for ferret). Purely
        advisory — nothing is enforced against it; it exists so notebooks /
        tooling can display the range and users can choose to clamp a
        ``wav2spec``'s frequency limits. ``None`` when unknown.
    """

    def __init__(self, path: str, dt_ms: float):
        super().__init__(path, dt_ms)
        self.F = -1
        self.audio_fs: Optional[int] = None
        self.hearing_range_hz: Optional[tuple] = None

    def get_F(self):
        """Return the number of frequency bins in the spectrograms."""
        return self.F

    @property
    def hop(self) -> Optional[int]:
        """Audio samples per neural bin in waveform mode (``None`` in spec mode).

        ``hop = round(audio_fs * dt_ms / 1000)``. The grid-lock contract (see
        :meth:`validate`) requires this to be an exact integer, so a
        ``wav2spec`` front-end's own ``hop`` must equal this value for the
        audio→neural resampling to stay aligned with the response bins.
        """
        if self.audio_fs is None:
            return None
        return int(round(self.audio_fs * self.dt / 1000.0))

    def _concat_check_compat(self, other):
        super()._concat_check_compat(other)
        assert isinstance(other, AudioNeuralDataset), \
            f"Cannot concatenate AudioNeuralDataset with {type(other).__name__}"
        assert self.F == other.F, \
            f"F mismatch: {self.F} vs {other.F}. Re-instantiate with matching n_mels."
        assert self.audio_fs == other.audio_fs, \
            f"audio_fs mismatch: {self.audio_fs} vs {other.audio_fs}."

    def _concat_copy_attrs(self, source):
        super()._concat_copy_attrs(source)
        self.F = source.F
        self.audio_fs = source.audio_fs
        self.hearing_range_hz = source.hearing_range_hz

    def validate(self):
        super().validate()
        assert isinstance(self.F, int) and self.F > 0, \
            f"self.F must be a positive int (got {self.F!r})"
        if self.hearing_range_hz is not None:
            hr = self.hearing_range_hz
            assert isinstance(hr, (tuple, list)) and len(hr) == 2 \
                and 0 < hr[0] < hr[1], (
                f"self.hearing_range_hz must be a (low, high) pair of positive "
                f"increasing numbers or None (got {hr!r})"
            )
        if self.audio_fs is not None:
            assert isinstance(self.audio_fs, int) and self.audio_fs > 0, \
                f"self.audio_fs must be a positive int or None (got {self.audio_fs!r})"
            self._validate_waveform_grid()

    def _validate_waveform_grid(self):
        """Enforce the waveform grid-lock contract (convention C1).

        In waveform mode every stim must be a ``(1, T_audio)`` tensor whose
        length is an exact multiple of ``hop = audio_fs * dt_ms / 1000``, and
        ``T_audio // hop`` must equal that stim's neural response length. This
        pins audio sample ``j`` to response bin ``j // hop``, which is what lets
        a strictly-causal ``wav2spec`` stay causal w.r.t. the responses. See the
        "Waveform conventions" section of ``docs/_source/md/data_paradigm.md``.
        """
        ratio = self.audio_fs * self.dt / 1000.0
        hop = int(round(ratio))
        assert hop >= 1 and abs(ratio - hop) < 1e-6, (
            f"waveform grid lock: audio_fs * dt_ms / 1000 = {ratio} is not an "
            f"integer number of samples per bin. Choose an audio_fs such that "
            f"audio_fs * {self.dt} / 1000 is an integer (e.g. 48000 at dt=5 ms)."
        )
        for s, stim in enumerate(self.stims):
            assert stim.dim() == 2 and stim.shape[0] == 1, (
                f"waveform stim {s} must be (1, T_audio); got {tuple(stim.shape)}"
            )
            T_audio = stim.shape[1]
            assert T_audio % hop == 0, (
                f"waveform stim {s}: T_audio={T_audio} is not a multiple of "
                f"hop={hop}. Each waveform must be right-padded / cropped to an "
                f"exact multiple of hop so it aligns with the response bins."
            )
            T_resp = self._stim_response_length(s)
            if T_resp is None:
                continue  # no neuron heard this stim — nothing to align against
            assert T_audio // hop == T_resp, (
                f"waveform stim {s}: T_audio // hop = {T_audio // hop} neural "
                f"frames but the response has {T_resp} bins. The waveform length "
                f"must equal T_resp * hop = {T_resp * hop} samples."
            )

    def _stim_response_length(self, s: int) -> Optional[int]:
        """Neural response length (in bins) for stim ``s``, or ``None`` if no
        neuron has real (non-sentinel) data for it.

        ``(1, 1)`` all-NaN tensors are the missing-data sentinels of the
        deepSTRF response paradigm and are skipped.
        """
        for r in self.responses[s]:
            if tuple(r.shape) == (1, 1) and bool(torch.isnan(r).all()):
                continue
            return int(r.shape[1])
        return None
