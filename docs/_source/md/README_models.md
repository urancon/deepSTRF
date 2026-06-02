# Computational neural response models

We provide a handful of computational models models that can be trained to fit biological auditory neural responses from 
stimuli in the form of a spectrogram, or "cochleagram".

---

## Conventions

All models are available inside `deepSTRF/models/models.py`, and are Pytorch classes inheriting from a mother class 
named `AudioResponseModel`, itself a subclass of `torch.nn.Module`.

As a result, they all implement inference inside `forward()` method, in pure PyTorch convention.
This method expects a `(B, 1, F, T)` input spectrogram tensor and outputs a  `(B, N, T)` neuronal response tensor.

Models share basic common attributes, such as:
* `self.T` --> the past *temporal context* to predict each timestep / convolution kernel size along time dimension
* `self.F` --> the number of frequency bins expected for the input spectrograms
* `self.O` --> number of output neurons whose activity to simultaneously predict
* `self.H` --> generally the number of hidden units
* ...

Models can be defined with or without the [AdapTrans model](https://www.biorxiv.org/content/10.1101/2024.01.17.576002v1), a prefiltering of the spectrogram that accounts for 
ON/OFF responses and is beneficial to fitting performances. In case of usage, it computes a bipolar  ON/OFF spectrogram 
prior to the backbone. Hyperparameters for AdapTrans can be given trhough the `prefiltering` argument of the constructor
as a dictionary. For example, `{'prefiltering': 'AdapTrans', 'dt': 1.0', 'min_freq': 500, 'max_freq': 20000, 'scale': 'mel'}`.

Models also generally implement a `STRFs(...)` method which returns the explicit spectro-temporal weighting of the cochleagram.

---


## Model Zoo

### Linear (L)

*Arguably the most naive model of the bunch, also known as **Spectro-Temporal Receptive Field (STRF)** model. The
predicted activity of this model at each timestep is simply a linear combination of past spectro-temporal time bins of 
input spectrogram, plus a bias representing baseline activity.
It has long been a popular model because of its simplicity, but often fails at properly rendering the activity of real 
neurons, which are highly nonlinear most of the time.*

*To mitigate overfitting due to too high numbers of learnable parameters, several techniques exist. For now, this repo 
only proposes a in-house parameterization called DCLS, but we aim to implement more common ones in the future, such as 
separable kernels. Similar to AdapTrans, prefiltering, hyperparameters for parameterization can be passed as an argument 
to the class constructor.*


Torch class: [`Linear`](https://deepstrf.readthedocs.io/en/latest/_source/deepSTRF.models.audio.html#deepSTRF.models.audio.audio_zoo.Linear); Parameterization available


### Linear-Nonlinear (LN)

*Consists of a Linear model, with an added output activation which makes it nonlinear. The latter often takes the form of 
a sigmoid or parameterized function (see e.g. Rahman et al. or Willmore et al.).*

Torch class: [`LinearNonlinear`](https://deepstrf.readthedocs.io/en/latest/_source/deepSTRF.models.audio.html#deepSTRF.models.audio.audio_zoo.LinearNonlinear); Parameterization available


### Network Receptive Field (NRF)

*In a nutshell, a LN model with several hidden units.*

Torch class: [`NetworkReceptiveField`](https://deepstrf.readthedocs.io/en/latest/_source/deepSTRF.models.audio.html#deepSTRF.models.audio.audio_zoo.NetworkReceptiveField); Parameterization available; Original paper: [Harper et al. (2016)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005113)





### Dynamic Network (DNet)

*In a nutshell, a NRF model in which hidden and output units follow leaky dynamics (as in LIF spiking neurons, but 
without spikes), with learnable time constants.*

Torch class: [`DNet`](https://deepstrf.readthedocs.io/en/latest/_source/deepSTRF.models.audio.html#deepSTRF.models.audio.audio_zoo.DNet); Parameterization available; Original paper: [Rahman et al. (2019)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006618)




### 2D-CNN

*Contrarily to other models, this one is not based on STRF model, as it is composed of successive convolutional steps whose
kernels do not entirely span all frequencies of the input spectrogram. Fully connected prediction head after a convlutional 
extraction stage.

Torch class: [`ConvNet2D`](https://deepstrf.readthedocs.io/en/latest/_source/deepSTRF.models.audio.html#deepSTRF.models.audio.audio_zoo.ConvNet2D);  Original paper: [Pennington & David (2023)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011110)


### Recurrent / state-space network (StateNet)

*A per-timestep spectral encoder feeding a recurrent backbone (GRU / LSTM / Mamba
/ S4 / LMU). It captures long-range temporal dependencies through the recurrent
state and is the strongest model in the zoo on NS1.*

Torch class: [`StateNet`](https://deepstrf.readthedocs.io/en/latest/_source/deepSTRF.models.audio.html#deepSTRF.models.audio.audio_zoo.StateNet); Original paper: [Rançon et al. (2025)](https://doi.org/10.1038/s42003-025-08858-3)


### Transformer

*Patch-embedding + sinusoidal positional encoding + a per-forward causal
self-attention mask, so it generalizes to any sequence length. An optional
finite `context_window` makes attention band-causal.*

Torch class: [`Transformer`](https://deepstrf.readthedocs.io/en/latest/_source/deepSTRF.models.audio.html#deepSTRF.models.audio.audio_zoo.Transformer); Architecture: [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762); designed as the attention baseline in [Rançon et al. (2025)](https://doi.org/10.1038/s42003-025-08858-3)


### ICNet

*A deep, waveform-native model: a SincNet + strided-conv encoder + bottleneck
feeding a per-neuron readout with a Poisson (softplus) head. It consumes raw
audio directly (no precomputed spectrogram). Designed for midbrain (IC)
recordings; it ports cleanly into deepSTRF but is oversized for small cortical
datasets like NS1.*

Torch class: [`ICNet`](https://deepstrf.readthedocs.io/en/latest/_source/deepSTRF.models.audio.html#deepSTRF.models.audio.ICNet); Original paper: [Drakopoulos et al. (2025)](https://doi.org/10.1038/s42256-025-01104-9)

---

## Waveform front-ends (`wav2spec`)

Every audio model defaults to consuming a precomputed spectrogram, but can
instead take a **raw waveform** through its `wav2spec` slot — a front-end that
maps audio to a neural-rate spectrogram and that can itself be **learned**
end-to-end with the model. Shipped front-ends:

| Front-end | Learnable | Notes |
|---|---|---|
| `CausalMelSpectrogram` | no | strictly-causal log-mel (Rahman 2019 defaults) |
| `CausalGammatone` | no | ERB gammatone filterbank; optional causal PCEN |
| `SincNet` | filter cutoffs | parametric bandpass (Ravanelli & Bengio 2018) |
| `CausalLEAF` | full | learnable Gabor + Gaussian pooling + sPCEN (Zeghidour 2021) |

All are strictly causal. See the dedicated [`wav2spec`](wav2spec.md) page for the
slot contract, the `make_wav2spec` factory, and the NS1 benchmarks. Waveform-native
models additionally expose `waveform_gradmap(...)`, a listenable **time-domain**
receptive field (the waveform analogue of `STRF_gradmap`).

