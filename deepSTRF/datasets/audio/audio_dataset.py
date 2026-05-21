from typing import Optional

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
    """

    def __init__(self, path: str, dt_ms: float):
        super().__init__(path, dt_ms)
        self.F = -1
        self.audio_fs: Optional[int] = None

    def get_F(self):
        """Return the number of frequency bins in the spectrograms."""
        return self.F

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

    def validate(self):
        super().validate()
        assert isinstance(self.F, int) and self.F > 0, \
            f"self.F must be a positive int (got {self.F!r})"
        if self.audio_fs is not None:
            assert isinstance(self.audio_fs, int) and self.audio_fs > 0, \
                f"self.audio_fs must be a positive int or None (got {self.audio_fs!r})"
