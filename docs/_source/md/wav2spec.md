# The `wav2spec` slot

`wav2spec` is the first slot of the canonical four-slot audio model
pipeline (`wav2spec → prefiltering → core → readout` — see
[`model_paradigm.md`](model_paradigm.md)). It is a `nn.Module` that maps
a raw mono audio waveform `(B, 1, T_audio)` to a spectrogram
`(B, 1, F, T_neural)` ready for consumption by the rest of the model.

The slot defaults to `nn.Identity()`, in which case the model expects a
precomputed spectrogram as input — the canonical deepSTRF setup since
v0. Setting `wav2spec=<a module>` flips the model to consume raw
waveforms; pair it with a dataset that exposes a waveform branch (e.g.
`NS1Dataset(return_waveform=True, audio_fs=16000)`).

## 1. The slot contract

A `wav2spec` module must expose three attributes:

| Attribute      | Type    | Meaning                                                  |
|----------------|---------|----------------------------------------------------------|
| `out_channels` | `int`   | Spectrogram band count `F` produced by the module        |
| `hop`          | `int`   | Audio samples per output frame (= `audio_fs · dt_ms / 1000`) |
| `audio_fs`     | `int`   | Sample rate the module expects on its input              |

and one forward shape contract:

```python
y = wav2spec(x)
# x.shape = (B, 1, T_audio)
# y.shape = (B, 1, F, T_neural)         where T_neural = T_audio // hop
```

The leading `1` on the output is the `C_in` channel axis that the rest
of the pipeline carries (prefiltering may turn it into 2 if you pair
with `AdapTrans`, for example). The `out_channels = F` constraint is
enforced at model-construction time by `AudioEncodingModel.__init__`,
which raises if `wav2spec.out_channels != n_frequency_bands`.

## 2. Strict causality

**Every `wav2spec` module shipped in deepSTRF satisfies strict causality:**
output frame `t` depends only on audio samples `[0, (t+1) · hop)` —
no leakage from neural bin `t+1` or later. The contract is enforced by
a parametrised Jacobian-probe test in `tests/test_wav2spec.py` that
every registered module must pass.

The full audio-model causality contract (waveform OR spectrogram input
→ output) is enforced by `tests/test_audio_models.py`. If you write
your own `wav2spec`, the easiest way to add it to the test bank is to
append a `(label, ctor)` tuple to `WAV2SPEC_CASES`.

## 3. Factory API

```python
from deepSTRF.models.wav2spec import make_wav2spec

mel = make_wav2spec("mel", audio_fs=16000, dt_ms=5.0)
sn  = make_wav2spec("sincnet", audio_fs=16000, dt_ms=5.0,
                     n_filters=34, kernel_size=251, envelope=True)
```

The factory mirrors `make_prefiltering` — it dispatches a string `kind`
against the shipped registry and forwards remaining kwargs to the
underlying class constructor. The shipped kinds:

| `kind`      | Class                  | Learnable? |
|-------------|------------------------|------------|
| `'mel'`     | `CausalMelSpectrogram` | no         |
| `'sincnet'` | `SincNet`              | yes (filter cutoffs) |

Both classes are also directly importable from
`deepSTRF.models.wav2spec` if you prefer to instantiate by hand
(e.g. to pass non-default `f_min` / `f_max`).

## 4. Shipped front-ends

### 4.1 `CausalMelSpectrogram` — non-learnable mel baseline

Strictly-causal log-mel spectrogram with defaults that reproduce the
[Rahman et al. 2019](https://doi.org/10.1371/journal.pcbi.1006618)
cochleagram used by the NS1 dataset: 10 ms Hanning window, 5 ms hop,
34 log-spaced channels 500–22 627 Hz, amplitude (not power) spectrogram,
threshold-clipped log. Causality: left-padded STFT (`win - hop` zeros)
+ `n_fft = win` + `center=False` on `torch.stft`. Acceptance: on NS1
``Linear(wav2spec=CausalMelSpectrogram(audio_fs=48000))`` reaches
test ``cc_norm`` 0.573 vs the precomputed-spec baseline at 0.548.

```python
from deepSTRF.models.wav2spec import CausalMelSpectrogram
# Defaults reproduce Rahman et al. 2019: 10 ms Hanning win, 500–22 627 Hz,
# amplitude, threshold-clipped log. Pair with audio_fs >= 45 kHz.
m = CausalMelSpectrogram(audio_fs=48000, n_mels=34)
```

### 4.2 `SincNet` — parametric bandpass (Ravanelli & Bengio 2018)

Each of the `n_filters` channels is a bandpass `Conv1d` filter with two
learnable parameters (low cutoff `f1`, high cutoff `f2`). The
time-domain impulse response is built on the fly each forward from the
analytic sinc-difference formula multiplied by a Hamming window. Two
output modes:

- `envelope=False` (default) — strided conv, signed bandpass-filtered
  audio sample per frame. Use when SincNet is followed by additional
  conv layers that can extract envelopes themselves (this is the ICNet
  regime).
- `envelope=True` — stride-1 conv → `abs()` → `avg_pool(hop)`. Produces
  a proper power-envelope spectrogram, comparable to mel. Use when
  SincNet is the *only* front-end before a thin readout.

Two activations carried over from the literature:

- `'symlog'` = `sgn(x) · log(|x|+1)` — ICNet's choice (sign-preserving
  log-compression).
- `'logabs'` = `log(|x|+1)` — standard SincNet (half-wave rectified).
- `'none'` — identity.

```python
from deepSTRF.models.wav2spec import SincNet
m = SincNet(audio_fs=48000, n_filters=34, kernel_size=753,
             hop_ms=5.0, init="mel", activation="logabs", envelope=True,
             env_window_ms=10.0)
```

### 4.3 ICNet's encoder — internal to `deepSTRF.models.audio.ICNet`

[ICNet](https://doi.org/10.1038/s42256-025-01104-9) (Drakopoulos et al.
Nat. Mach. Intell. 2025) has its own SincNet-and-conv-stack front-end:
`SincNet(48 filters, K=64, stride 1, symlog)` → 5× causal `Conv1d(128
ch, K=64, PReLU)` → bottleneck `Conv1d(64 ch, K=64, stride 1, PReLU)`,
producing a 64-channel bottleneck latent at the neural rate. It slots
into ICNet's own `wav2spec` attribute but is not exposed in the public
`deepSTRF.models.wav2spec` namespace — it's tightly coupled to the
ICNet model and not particularly useful as a generic front-end. Build
the full ICNet model instead:

```python
from deepSTRF.models.audio import ICNet
m = ICNet(audio_fs=48000, out_neurons=119, dt_ms=5.0)   # NS1 config
# m.wav2spec is the encoder; treat it as opaque.
```

## 5. Writing your own

A new `wav2spec` module needs three things: the contract attributes
(`out_channels`, `hop`, `audio_fs`), strict causality (use left-only
padding before any `Conv1d` / `stft` you call), and the
`(B, 1, T_audio) → (B, 1, F, T_neural)` shape.

To register it with the parametrised test bank, add a
`(label, lambda: YourModule(...))` entry to `WAV2SPEC_CASES` in
`tests/test_wav2spec.py`; the bank then exercises shape,
eval-determinism, Jacobian causality, and input-rank validation. If
you want a factory entry, register a new `kind` in
`make_wav2spec(...)` in `deepSTRF/models/wav2spec/__init__.py`.
