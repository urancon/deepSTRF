# The deepSTRF model paradigm

This note documents how deepSTRF structures encoding models, what the
forward contract is, and what invariants any model author must respect. It
is the single reference for the "model contract" of the library and should
stay in sync with the `NeuralModel` / `AudioEncodingModel` /
`VideoEncodingModel` base classes.

It is the encoding-model counterpart to `data_paradigm.md`. The two
documents together describe how data and models meet at the loss.

## 1. Scope: encoding models

All current models in `deepSTRF.models.audio` and `deepSTRF.models.video`
are **encoding** models: they take a sensory stimulus and predict the
neural response (PSTH or per-trial activity).

A future extension will introduce **decoding** models (response → stim).
The dataloader contract does not change — only the model class hierarchy.
Encoding models live under `AudioEncodingModel` / `VideoEncodingModel`;
decoding models will get parallel `*DecodingModel` base classes when a
concrete use case drives their design.

## 2. The forward template

Every encoding model in deepSTRF is a four-stage pipeline:

```
       stimulus
          │
          ▼
   ┌─────────────┐
   │  wav2spec   │   future slot — Identity by default. Learnable
   │             │   audio front-end (e.g. LEAF) when raw waveforms
   │             │   become first-class inputs.
   └─────────────┘
          │
          ▼
   ┌─────────────┐
   │ prefiltering│   AdapTrans / ICAdaptation / Identity. Stim-side
   │             │   non-trainable or lightly-trainable preprocessing.
   │             │   Declares `out_channels: int` so the downstream
   │             │   core can size its inputs.
   └─────────────┘
          │
          ▼
   ┌─────────────┐
   │    core     │   Stim-shared feature backbone. ConvNet / RNN / SSM
   │             │   / Transformer / etc. For LN models the core is
   │             │   `nn.Identity()` — no shared features, the readout
   │             │   does all the work.
   └─────────────┘
          │
          ▼
   ┌─────────────┐
   │   readout   │   Per-neuron projection. `out_channels = N`. Owns
   │             │   the output activation. STRFReadout for L/LN/NRF/
   │             │   DNet; LinearReadout for ConvNet / RNN / Transformer
   │             │   models.
   └─────────────┘
          │
          ▼
       response
```

The base class implements `forward` once:

```python
def forward(self, x):
    x = self.wav2spec(x)
    x = self.prefiltering(x)
    f = self.core(x)
    return self.readout(f)
```

Concrete models populate the slots in their `__init__`. The base
defines reasonable defaults (`nn.Identity()` for `wav2spec`,
`prefiltering`, and `core`); the `readout` slot has no default and must
be set by every concrete model — `validate()` enforces this.

Here is the actual implementation of `Linear` in `audio_zoo.py`, the
minimal reference example:

```python
class Linear(AudioEncodingModel):
    def __init__(self, n_frequency_bands=34, temporal_window_size=9, out_neurons=1,
                 output_activation=None, prefiltering=None, kernel=None):
        super().__init__(
            n_frequency_bands=n_frequency_bands,
            temporal_window_size=temporal_window_size,
            out_neurons=out_neurons,
            prefiltering=prefiltering,
        )
        # core: causal per-timestep frequency normalization
        self.core = CausalLayerNorm(self.F, dim=-2)
        # readout: pluggable STRF kernel + output activation
        self.readout = STRFReadout(
            F=self.F, T=self.T, C_in=self.C_in, out_neurons=self.O,
            kernel=kernel,
            activation=output_activation or nn.Identity(),
        )
        # forward inherited from NeuralModel
```

Subclasses *may* override `forward` when the architecture genuinely cannot
fit the template — StateNet's RNN reshape and Transformer's per-forward
attention-mask construction are current examples. These overrides are
exceptions and should be called out in the model docstring.

## 3. The forward signature

### Input shape (per modality)

| Modality | Stimulus shape           | Notes                                                      |
|---       |---                       |---                                                         |
| audio    | `(B, 1, F, T_max)`       | float32, zero-padded on the right along T. **Never NaN.**  |
| video    | `(B, 1, H, W, T_max)`    | float32, zero-padded on the right along T. **Never NaN.**  |

These match the output of `neural_collate` exactly. Models should never
need to reshape the input; if they do, the reshape is internal to one of
the four slots (typically the core).

### Output shape

**All models output `(B, N, R=1, T_max)`.**

The R-axis is degenerate today (always 1) but kept for forward-
compatibility with probabilistic models that emit one or more samples
(or distribution parameters) per time bin. Single-prediction models
unsqueeze; probabilistic models populate the axis.

### Where the R=1 axis comes from

| Slot         | Output rank                                                    |
|---           |---                                                             |
| `wav2spec`   | `(B, 1, F, T)` — same as input                                 |
| `prefiltering` | `(B, C_in, F, T)` where `C_in ∈ {1, 2}`                      |
| `core`       | model-specific — `(B, C_feat, F, T)` for conv backbones, `(B, T, H)` for RNN/Transformer |
| `readout`    | `(B, N, 1, T)`                                                 |

The readout owns the responsibility of projecting to N output channels and
unsqueezing the R axis. This is by design: it keeps the rank-normalization
in one place, and the `STRF_weight` / `STRF_gradmap` introspection helpers
have a single object to query.

## 4. Causality

deepSTRF encoding models are **strictly causal** by convention: the
prediction at time `t` depends only on stim `[..., t]`, never on
`[t+1, ...]`.

This is a hard contract, not an aspiration. It matters for:

- **Real-time inference** — closed-loop neuroscience experiments.
- **Generalization across context lengths** — a non-causal model trained
  on `T=100` clips may behave very differently on `T=1000` because the
  effective receptive field shifted.
- **Comparability with biological neurons** — they're causal too.

### Causal building blocks

| Operation                              | Causal? | Notes                                       |
|---                                     |---      |---                                          |
| `nn.Conv{1,2,3}d` with `padding=0`     | ✓       | left-pad explicitly with `nn.ZeroPad`       |
| `F.pad(..., (K-1, 0))`                 | ✓       | left-only temporal padding                  |
| `F.pad(..., (K-1, 0), mode='replicate')` | ✓     | causal extrapolation                        |
| `CausalLayerNorm(C, dim=1)`            | ✓       | LayerNorm over the channel axis at every (B, F, T) position |
| `CausalLayerNorm(F, dim=-2)`           | ✓       | LayerNorm over the frequency axis at every (B, C, T) position |
| `nn.GRU` / `nn.LSTM` / `nn.RNN`        | ✓       | inherently causal                           |
| RNN-style SSMs (S4, Mamba)             | ✓       | inherently causal                           |
| `nn.TransformerEncoder` + causal mask  | ✓       | mask required — without it, attention is bidirectional |

### Non-causal traps to avoid

| Operation                              | Why it breaks causality                     |
|---                                     |---                                          |
| `nn.BatchNorm{1,2,3}d`                 | Pools statistics across batch and time      |
| `padding=(0, (K-1)//2)` (symmetric)    | Includes future timesteps                   |
| `nn.AdaptiveAvgPool` along T           | Pools over the full clip                    |
| `nn.GroupNorm(G, C)` over `(F, T)`     | Pools statistics over time within each sample |
| `nn.TransformerEncoder` without mask   | Attention sees all positions                |

**Use `CausalLayerNorm` instead of `BatchNorm`** — it normalizes across
a single axis (channel or frequency, by ``dim`` argument) at each
position of every other axis, never pooling across time. Defined in
`deepSTRF.models.layers`; thin wrapper around `nn.LayerNorm` with a
`movedim` to reach a non-trailing target axis.

## 5. Where the weights live

Heuristic for placing learnable parameters:

| Parameter shape / role                          | Slot          |
|---                                              |---            |
| Per-neuron (`out_channels = N`, fan-out from a shared feature space) | readout       |
| Stim-shared (`out_channels` independent of N)   | core          |
| Stim-side, often biology-derived                | prefiltering  |
| Audio-frontend, raw-waveform-to-spectrogram     | wav2spec (future) |

### Worked example: LN model

For a Linear-Nonlinear model the entire trainable apparatus that
distinguishes one neuron from another is the STRF kernel
`(N, C_in, F, T)` — N independent filters, one per neuron — held by the
readout. The core does only one thing: per-timestep frequency
normalization.

```python
self.wav2spec     = nn.Identity()
self.prefiltering = AdapTrans(...) or nn.Identity()
self.core         = CausalLayerNorm(F, dim=-2)    # input freq norm
self.readout      = STRFReadout(F, T, C_in, N,
                                kernel=...,
                                activation=nn.Sigmoid())
```

The core's LayerNorm has `2*F` parameters (γ, β per frequency band) but
is stim-shared across all N neurons. The per-neuron weights all live
in the readout. The template handles deeper models the same way — they
just populate `core` with more layers.

### Worked example: ConvNet2D

```python
self.prefiltering = ...
self.core         = nn.Sequential(
                       CausalLayerNorm(F, dim=-2),                     # input freq norm
                       nn.ZeroPad2d((3*(K[1]-1), 0, 0, 0)),             # explicit causal left-pad
                       Conv2d(C_in, C, K), CausalLayerNorm(C, dim=1), LeakyReLU(),
                       Conv2d(C, C, K),    CausalLayerNorm(C, dim=1), LeakyReLU(),
                       Conv2d(C, C, K),    CausalLayerNorm(C, dim=1), LeakyReLU(),
                       nn.Flatten(start_dim=1, end_dim=2),              # (B, C, F_down, T) → (B, C*F_down, T)
                    )
self.readout      = LinearReadout(C * F_down, N,
                                  hidden=H,
                                  activation=nn.Sigmoid())
```

The convolutional features are stim-shared (every neuron sees the same
backbone output); the per-neuron projection happens in the readout's
final `nn.Linear(H, N)`.

## 6. Prefiltering

Prefiltering is a stim-side preprocessing slot, often biologically
motivated (cochlear adaptation, retinal lateral inhibition).

### Contract

A prefilter is **any `nn.Module`** that:

1. Takes a stimulus tensor of shape `(B, 1, F, T)` (audio) or
   `(B, 1, H, W, T)` (video).
2. Returns a tensor of shape `(B, C_out, F, T)` or `(B, C_out, H, W, T)`
   with the same spatial / spectral dimensions and an arbitrary number
   of channels.
3. Exposes an integer attribute `out_channels`.

The contract is duck-typed (no formal ABC) — the base model reads
`prefiltering.out_channels` to size the downstream core.

### Currently shipped

| Class           | Module                          | `out_channels` | Learnable? | Reference                  |
|---              |---                              |---             |---         |---                          |
| `Identity`      | `nn.Identity` (out_channels=1)  | 1              | —          | default no-op               |
| `ICAdaptation`  | `deepSTRF.models.prefiltering`  | 1              | No         | Willmore et al. 2016, J. Neurosci. |
| `AdapTrans`     | `deepSTRF.models.prefiltering`  | 2              | Yes (default; `learnable=False` available) | Rançon et al. 2024, PLOS CB |

**`ICAdaptation` is intentionally frozen** — paper-faithful to Willmore
et al. who derive the time constants analytically from the cochlear
frequency map (`freq_to_tau` formula).

### Ergonomics: `make_prefiltering`

For users who don't want to construct a prefilter by hand:

```python
from deepSTRF.models.prefiltering import make_prefiltering

prefilt = make_prefiltering("adaptrans", n_frequency_bands=34, dt=5.0,
                            min_freq=500, max_freq=20000, scale="mel")
```

Equivalent to the hand-construction. The factory is convenience, not
contract — passing a `nn.Module` instance directly is also fine.

## 7. Readouts

A readout takes the core's output and emits `(B, N, 1, T)`. Two are
shipped; more can be added per modality or per parameterization scheme.

### `STRFReadout`

A learnable `(C_in, N, F, T)` STRF kernel applied via causal Conv2d. The
"linear" half of an LN model.

```python
STRFReadout(F, T, C_in, N,
            kernel: nn.Module = None,    # default: vanilla full kernel
            activation: nn.Module = nn.Identity(),
            bias: bool = True)
```

The `kernel` slot is itself pluggable, which is how parameterization
schemes ride into the model:

| Kernel class            | What it parameterizes                                                          |
|---                       |---                                                                              |
| `None` (default)        | Vanilla full `(N, C_in, F, T)` kernel — `nn.Conv2d`                            |
| `ParametricSTRF`        | Sum of K Gaussians with learnable positions/sigmas (DCLS, Khalfaoui-Hassani 2023, ICLR) |
| `SeparableSTRF`         | Frequency × time outer product (rank-1 by construction)                        |
| Future: `RankRSTRF`     | Rank-r factorization                                                            |

Internally `STRFReadout` left-pads its input by `T-1` zeros on the time
axis (causality) and applies `Conv2d` with the kernel. Output is
`(B, N, 1, T)` — the singleton frequency axis after `(F → 1)` reduction
serves naturally as the R axis after a transpose.

`STRFReadout.STRF_weight(polarity='ON' | 'OFF')` returns the underlying
kernel for visualization, regardless of parameterization. Models with an
`STRFReadout` proxy this method: `model.STRF_weight()` → readout's.

### `LinearReadout`

A per-neuron projection from a flat feature vector — the "fc" at the end
of ConvNet / RNN / Transformer cores.

```python
LinearReadout(in_features, out_neurons,
              hidden: int = None,        # optional 1-hidden-layer MLP
              activation: nn.Module = nn.Identity(),
              bias: bool = True)
```

Accepts `(B, in_features, T)` (channel-as-dim-1 convention, matching
`Conv2d` outputs) or `(B, in_features, 1, T)` (the singleton-spatial-axis
shape produced by an STRF-style conv that collapsed `F → 1`); both shapes
route through the same projection. Always emits `(B, N, 1, T)`.

## 8. Output activations

Activations live **inside the readout**, not on the base class. Each
readout takes an `activation: nn.Module` kwarg.

### Shipped

| Class                          | Use                                                                | Reference                              |
|---                             |---                                                                  |---                                     |
| `nn.Identity`                  | Centered targets (z-scored PSTH)                                    | default                                |
| `nn.Sigmoid`                   | Bounded `[0, 1]` targets                                            | —                                      |
| `nn.Softplus`                  | Non-negative spike-rate targets, smooth                             | —                                      |
| `ParametricSigmoid`            | 4-param `b/(1+exp(-(x-c)/d)) + a`, per-neuron                      | Willmore et al. 2016, J. Neurosci.     |
| `ParametricDoubleExponential`  | 4-param `a · exp(-exp(k·x - s)) + b`, per-neuron                   | Thorson et al. 2015, PLOS CB           |

The parametric activations have one set of learnable parameters per
output neuron (`N` instances each).

### Caveat

In past internal experiments, parametric activations did not consistently
improve correlations vs `nn.Sigmoid`. Re-implementation against the
original papers is part of the modernization branch — bug or genuine
finding is open. See `TODO.md`.

## 9. STRF introspection

Two complementary helpers:

| Method            | What it returns                                              | Where                  | Generality                                     |
|---                |---                                                            |---                     |---                                             |
| `STRF_weight`     | The underlying kernel `(N, C_in, F, T)` from learned weights | `STRFReadout`           | L / LN / NRF / DNet (closed-form readout)      |
| `STRF_gradmap`    | Gradient of output activity w.r.t. a null input              | `NeuralModel` (base)   | Any model (architecture-agnostic)              |

`STRF_gradmap` parallelizes one gradmap per output neuron in a single
forward/backward pass, using the batch dimension (`B = self.O`). This is
worth preserving because it scales linearly with neurons-on-GPU and
avoids per-neuron re-instantiation.

A dedicated `gradmap.md` and tutorial notebook are planned (see
`TODO.md`).

## 10. The `validate()` contract

Every concrete model must satisfy:

1. **Slots populated.** `wav2spec`, `prefiltering`, `core`, `readout`
   are all `nn.Module` instances. The first three default to
   `nn.Identity` if the subclass doesn't set them; `readout` has no
   default and must be assigned before `validate()` is called.
2. **`out_neurons > 0`** (`self.O > 0`).
3. **Output rank.** `forward(x)` emits `(B, N, R=1, T)`. The contract
   tests in `tests/test_audio_models.py` enforce this for every
   concrete audio model.
4. **Causality.** `forward(x)[..., :t]` is independent of `x[..., t:]`.
   Enforced by the bit-causality test in
   `tests/test_audio_models.py::test_bitwise_causality_in_eval_mode`.

Subclasses extend `validate()` with modality-specific invariants —
`AudioEncodingModel.validate()` checks `F, T > 0` and `C_in >= 1`;
`VideoEncodingModel.validate()` checks `H, W, T > 0`.

## 11. Invariants for model authors

1. **Forward emits `(B, N, R=1, T)`.** Always. No exceptions for "I only
   have one neuron" or "I'm a deterministic model". The R axis is
   structural.
2. **Causality is non-negotiable.** No `BatchNorm`, no symmetric padding,
   no global temporal pooling.
3. **Per-neuron weights live in the readout.** If your model has
   `out_channels = N` somewhere outside the readout, that's a smell.
4. **Stim-side input is never NaN.** Don't write defensive `nan_to_num`
   in the model; the dataset guarantees clean stims.
5. **Predictions are emitted at every position** — including padded
   regions and uncorded neurons. The loss handles masking, not the
   model. (Same rule as `data_paradigm.md` §7.5.)
6. **Subclasses call `self.validate()` as the last line of `__init__`.**

## 12. Future extensions noted in TODO.md

- **Decoding models.** Parallel `*DecodingModel` hierarchy when the first
  concrete decoder lands.
- **Raw-waveform front-end.** `wav2spec` slot becomes an active
  position; LEAF (Zeghidour et al. 2021) is the first candidate.
- **Transformer RoPE option.** Sinusoidal positional encoding ships
  today; RoPE (Su et al. 2021) is a planned alternative once it has
  empirical support.
- **`gradmap.md`** and `examples/strf_gradmap.ipynb`.
- **Output activations recheck.** Verify `ParametricSigmoid` and
  `ParametricDoubleExponential` against their original papers; add
  `Softplus` for non-negative-rate targets.
- **SSM dependency cleanup.** Replace vendored `s4.py` / `mamba.py` /
  `lmu.py` with upstream packages where they've stabilized.
