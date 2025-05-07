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


Torch class: `Linear(...)`; Parameterization available


### Linear-Nonlinear (LN)

*Consists of a Linear model, with an added output activation which makes it nonlinear. The latter often takes the form of 
a sigmoid or parameterized function (see e.g. [Rahman et al.]() or [Willmore et al.]()).*

Torch class: `LinearNonlinear(...)`; Parameterization available


### Network Receptive Field (NRF)

*In a nutshell, a LN model with several hidden units.*

Torch class: `NetworkReceptiveField(...)`; Parameterization available; Original paper: [Harper et al. (2016)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005113)





### Dynamic Network (DNet)

*In a nutshell, a NRF model in which hidden and output units follow leaky dynamics (as in LIF spiking neurons, but 
without spikes), with learnable time constants.*

Torch class: `DNet(...)`; Parameterization available; Original paper: [Rahman et al. (2016)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006618)




### 2D-CNN

*Contrarily to other models, this one is not based on STRF model, as it is composed of successive convolutional steps whose
kernels do not entirely span all frequencies of the input spectrogram. Fully connected prediction head after a convlutional 
extraction stage.

Torch class: `ConvNet2D(...)`;  Original paper: [Pennington et al.](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011110)

