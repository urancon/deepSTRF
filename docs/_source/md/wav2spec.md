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

Strictly-causal log-mel spectrogram. Left-padded STFT (`win - hop` zeros)
+ `n_fft = win` (so the STFT frame stride matches the windowed region
exactly) + `center=False` on `torch.stft` + mel filterbank + `log(mel +
log_offset)`. The pipeline-validation phase 2 acceptance test trains
`Linear(wav2spec=CausalMelSpectrogram(...))` on NS1 and confirms test
`cc_norm` matches the precomputed-spec baseline.

```python
from deepSTRF.models.wav2spec import CausalMelSpectrogram
m = CausalMelSpectrogram(audio_fs=16000, n_mels=34, hop_ms=5.0,
                          win_ms=25.0, f_min=300.0, f_max=8000.0)
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
m = SincNet(audio_fs=16000, n_filters=34, kernel_size=251,
             hop_ms=5.0, init="mel", activation="logabs", envelope=True)
```

### 4.3 `ICNetFrontend` — the deep frontend of Drakopoulos et al. 2025

The convolutional encoder of [ICNet](https://doi.org/10.1038/s42256-025-01104-9):
`SincNet(48 filters, K=64, stride 1, symlog)` → 5× causal `Conv1d(128
ch, K=64, PReLU)` → bottleneck `Conv1d(64 ch, K=64, stride 1, PReLU)`.
Output `out_channels = 64` (the bottleneck latent — not a "spectrogram"
in the conventional sense, but it slots into the contract).

The 5 encoder strides multiply to `audio_fs · dt_ms / 1000`. Paper
defaults (24414 Hz / 1.31 ms) give `[2,2,2,2,2]` (total ÷32); NS1
(16 kHz / 5 ms) gives `[2,2,2,2,5]` (total ÷80). The full ICNet model
(encoder + Poisson decoder) is `deepSTRF.models.audio.ICNet`.

```python
from deepSTRF.models.wav2spec import ICNetFrontend
fe = ICNetFrontend(audio_fs=16000, dt_ms=5.0)   # ~5.0M params on NS1
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
