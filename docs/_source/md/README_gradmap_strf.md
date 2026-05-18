# Retrieving the linear STRF from gradient maps

A trained nonlinear encoding model (ConvNet, RNN, Transformer, …) does
not expose its spectro-temporal receptive field as an explicit weight
tensor — the way a Linear / LN model does. The **gradient map**
(*GradMap*) is the standard workaround: a single backward pass through
the trained model, starting from a null stimulus, produces the
"effective linear STRF" of any model, irrespective of its internal
architecture.

This page documents the math, the deepSTRF API, the sign convention,
and the limitations. The companion demo is
[`examples/strf_gradmap_aa2.ipynb`](../ipynb/strf_gradmap_aa2.ipynb)
(ConvNet2D fit on CRCNS AA2, gradmaps for the top-12 best-predicted
cells).

## 1. From linear to nonlinear models

For a linear model the response to a stimulus $\mathbf{x} \in
\mathbb{R}^{F \times T}$ is

$$
\hat r_n[T] \;=\; \sum_{f,\,\tau} W_n[f, \tau]\, \mathbf{x}[f, T-\tau],
$$

so the **STRF** $W_n \in \mathbb{R}^{F \times T}$ is literally a
learned parameter of the model and can be visualised directly. For a
nonlinear model, no such weight tensor exists; what we have instead is
the response function $\hat r_n = f_\theta(\mathbf{x})$, with $\theta$
the trained parameters of the model.

The gradient map is the natural generalisation:

$$
\boxed{\;\;
\mathbf{g}_n
\;=\; \left.\frac{\partial \mathcal{L}_n}{\partial \mathbf{x}}\right|_{\mathbf{x}=\mathbf{x}_0}
\quad\text{with}\quad
\mathcal{L}_n(\hat r) = -\,\hat r_n[T],
\quad
\mathbf{x}_0 = \mathbf{0} \in \mathbb{R}^{F \times T}.
\;\;}
$$

In words: at the *null stimulus* $\mathbf{x}_0$ (a constant-zero
spectrogram), we ask for the gradient of the **negative** last-timestep
activity of neuron $n$ with respect to the input. The result
$\mathbf{g}_n \in \mathbb{R}^{F \times T}$ has the same shape as a
classical STRF and can be plotted the same way.

For a population $\mathcal{N}$ of neurons one substitutes
$\mathcal{L}_\mathcal{N}(\hat r) = -\tfrac{1}{|\mathcal{N}|}
\sum_{n \in \mathcal{N}} \hat r_n[T]$ and obtains the population
gradmap.

## 2. Why this generalises the LN STRF

The choice of formula is not arbitrary. Consider a Linear–Nonlinear
(LN) model

$$
\hat r_n[T] \;=\; \sigma\!\bigl(W_n \cdot \mathbf{x} + b_n\bigr),
$$

with $\sigma$ a pointwise monotone nonlinearity. Differentiating at
$\mathbf{x} = \mathbf{x}_0 = \mathbf{0}$ gives

$$
\left.\frac{\partial \hat r_n[T]}{\partial \mathbf{x}}\right|_{\mathbf{x}_0}
= \sigma'(b_n)\, W_n,
\qquad
\mathbf{g}_n \;=\; -\,\sigma'(b_n)\, W_n.
$$

So for an LN model the gradient map *is* the STRF, up to a positive
scalar $\sigma'(b_n)$ and an overall sign flip. For a nonlinear model
trained on the same data, $\mathbf{g}_n$ recovers an analogous
"effective" STRF — the first-order linearisation of $f_\theta$ around
silence. This is exactly the picture that `STRF_gradmap` returns.

## 3. The deepSTRF API

Every `AudioEncodingModel` exposes the method directly on the model
instance:

```python
model = ConvNet2D(...)
# ... train it ...

model.eval()
gradmaps = model.STRF_gradmap()       # (N, 1, F, T)
gradmaps = model.STRF_gradmap(T=24)   # override the temporal extent
```

A single forward + backward pass populates one gradmap per output
neuron. The batch dimension is reused as the neuron dimension, so all
$N$ gradmaps are computed in parallel.

The returned tensor has shape `(N, 1, F, T)`. `N` is the number of
output neurons of the model, `F` is `model.F` (the number of input
frequency bands), and `T` defaults to `model.T`
(`temporal_window_size`) — the STRF extent set at construction time —
but can be overridden per call.

## 4. Sign convention when plotting

The returned tensor follows the paper's convention:
$\mathbf{g}_n = \partial \mathcal{L}_n / \partial \mathbf{x}$ with
$\mathcal{L}_n = -\hat r_n[T]$. Under this convention,

$$
\mathbf{g}_n[f, \tau] > 0
\;\;\Longleftrightarrow\;\;
\text{adding stimulus energy at }(f, T-\tau)
\text{ would }\textbf{decrease}\text{ the neuron's response.}
$$

If you want the more intuitive *excitatory-as-positive* visualisation —
red regions = features the cell prefers — plot $-\mathbf{g}_n$ instead
of $\mathbf{g}_n$ (or flip the colormap, e.g. `RdBu` instead of
`RdBu_r`). With the chosen sign convention, $-\mathbf{g}_n$ is also
the gradient-ascent direction on $\hat r_n[T]$.

## 5. Caveats and limitations

- **Single-channel gradient.** For prefiltered models with
  $C_\text{in} > 1$ (e.g. `AdapTrans` exposes two channels — a fast and
  a slow adaptation channel — to the downstream core), the gradmap is
  currently returned at the raw spectrogram level only, shape
  `(N, 1, F, T)`. The per-channel decomposition that would let you
  see the fast-vs-slow contribution separately is not exposed yet.
- **Last-timestep readout only.** The loss is hardcoded to
  $\mathcal{L}_n = -\hat r_n[T]$. A complementary diagnostic is the
  *time-averaged* readout
  $-\tfrac{1}{T} \sum_t \hat r_n[t]$, which highlights features that
  drive *sustained* responses rather than transient ones. The
  `STRF_gradmap` method does not currently accept a custom loss.
- **First-order linearisation only.** GradMap captures the local
  behaviour of the model at $\mathbf{x}_0$. The full procedure
  proposed in Rançon et al. (2025) — "Dreams" — iterates the gradient
  step

  $$
  \mathbf{x}_{t+1} \;=\; \mathbf{x}_t \;-\; \alpha\, \nabla_{\mathbf{x}_t} \mathcal{L}(\hat r),
  $$

  for ${\sim}1500$ steps with the Adam optimiser to synthesise
  spectrograms that maximally drive a neuron (or population) — a
  nonlinear generalisation of the spike-triggered average. The
  iterative version is not in deepSTRF today; the single-step
  `STRF_gradmap` is the linearised special case ($t = 0$).

## 6. Demo notebook

[`examples/strf_gradmap_aa2.ipynb`](../ipynb/strf_gradmap_aa2.ipynb)
walks through a complete workflow: train a `ConvNet2D` on a subset of
CRCNS AA2 (zebra finch ovoidalis, conspecific stimuli only), then
extract and plot the gradmaps of the 12 best-predicted cells in a
single backward pass.

## 7. Citation

If you use gradmaps in published work, please cite the original paper:

> Rançon U., Masquelier T., Cottereau B. R. *Temporal recurrence as a
> general mechanism to explain neural responses in the auditory
> system*. **Communications Biology** 8:1456 (2025).
> [doi:10.1038/s42003-025-08858-3](https://doi.org/10.1038/s42003-025-08858-3)
