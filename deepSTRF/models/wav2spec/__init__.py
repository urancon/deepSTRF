"""Waveform-to-spectrogram front-ends for audio encoding models.

A ``wav2spec`` module is an ``nn.Module`` that maps a mono audio waveform
``(B, 1, T_audio)`` to a (batched) spectrogram ``(B, 1, F, T_neural)`` for
downstream consumption by an ``AudioEncodingModel``. The leading ``1`` on
the output is the ``C_in`` channel axis carried by the rest of the pipeline
(``prefiltering → core → readout``); it is a singleton for most front-ends
and can be enlarged by a downstream prefilter such as ``AdapTrans``.

Every wav2spec module satisfies the deepSTRF strict-causality contract:
output frame ``t`` depends only on input audio samples
``[0, (t+1) * hop)`` — i.e. it cannot leak any future audio across the
neural bin boundary. The contract is enforced by a parametrised Jacobian
test in ``tests/test_wav2spec.py``.

Modules expose:

- ``out_channels: int``   — the spec-channel count produced by the module
                            (``F`` in the rest of the pipeline).
- ``hop: int``            — audio-samples-per-neural-bin (used by causality
                            tests and downstream collate / batching logic).
- ``audio_fs: int``       — sample rate the module expects on its input.
"""

from .causal_mel import CausalMelSpectrogram
from .gammatone import CausalGammatone
from .gammatonegram import Gammatonegram
from .leaf import CausalLEAF
from .sincnet import SincNet


__all__ = ["CausalMelSpectrogram", "CausalGammatone", "Gammatonegram",
           "CausalLEAF", "SincNet", "make_wav2spec"]


def make_wav2spec(kind: str, audio_fs: int, dt_ms: float, **kwargs):
    """Factory for constructing a ``wav2spec`` module from compact arguments.

    Parameters
    ----------
    kind : {'mel', 'gammatone', 'gammatonegram', 'sincnet', 'leaf'}
        Which front-end to build. ``'mel'`` and ``'gammatone'`` are
        non-learnable cochleagrams; ``'gammatonegram'`` is the faithful causal
        reproduction of the canonical Slaney/Heeris gammatone-gram (the
        ``gammatone`` package / NEMS / Meliza native transform); ``'sincnet'``
        has learnable filter cutoffs; ``'leaf'`` is the fully-learnable LEAF
        frontend (Gabor + pooling + sPCEN).
    audio_fs : int
        Audio sample rate (Hz). Must match the dataset's ``audio_fs``.
    dt_ms : float
        Neural time-bin width in milliseconds (matches ``dataset.dt_ms``).
        Determines the STFT hop: ``hop = audio_fs * dt_ms / 1000`` samples
        per neural bin.
    **kwargs
        Forwarded to the underlying module's ``__init__``. See each class's
        docstring for the available knobs.

    Returns
    -------
    nn.Module
        Configured wav2spec instance with ``out_channels`` / ``hop`` /
        ``audio_fs`` attributes.
    """
    kind = kind.lower()
    if kind == "mel":
        return CausalMelSpectrogram(audio_fs=audio_fs, hop_ms=dt_ms, **kwargs)
    if kind == "gammatone":
        return CausalGammatone(audio_fs=audio_fs, hop_ms=dt_ms, **kwargs)
    if kind == "gammatonegram":
        return Gammatonegram(audio_fs=audio_fs, hop_ms=dt_ms, **kwargs)
    if kind == "sincnet":
        return SincNet(audio_fs=audio_fs, hop_ms=dt_ms, **kwargs)
    if kind == "leaf":
        return CausalLEAF(audio_fs=audio_fs, hop_ms=dt_ms, **kwargs)
    raise ValueError(
        f"Unknown wav2spec kind {kind!r}. Currently supported: "
        f"'mel', 'gammatone', 'gammatonegram', 'sincnet', 'leaf'."
    )
