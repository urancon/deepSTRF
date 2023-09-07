# Tensor formats

In this short note, we indicate the expected format (dimensions) of tensors in our torch-based framework. Most of the 
code is compatible with numpy.


### Dimension Abbreviations
For cleaner documentation, we denote each dimension by a specific letter. Here is the conversion table:
* `S` --> *Stimulus-Response pairs*
* `B` --> *Batches*
* `R` --> *Repeats*
* `T` --> *Time-bins*
* `F` --> *Frequency-bins*
* `C` --> *Channels*
* `N` --> *Neurons*

Finally, `*` means any number of dimensions.
You can find these abbreviations all over the code contained within this repository.


### Tensors
Our framework expects a specific dimensionality for each nature of tensor. To keep the high generality, please respect 
these conventions when building on the project.
* *stimuli (spectrogram form)*: `(B, C=1, F, T)`
* *predicted responses*: `(B, R=1, T, N)`
* *observed responses (groundtruth)*: `(B, R, T, N)`

Dataset classes and pytorch dataloaders are supposed to serve the data in the above formats; similarly, tensor reshapes
should be made within the `forward()` method of model classes.
This allows minimal tensor manipulation in fitting and evaluation loops, making the framework general to most datasets, 
models and performance metrics. It also contributes to a  cleaner and much more modular code, in which you only need to 
change datasets and models construction.


### Example
Here is an illustration of a proper usage of the conventions set by our framework.

```python
from models import MyModelClass
model = MyModelClass(args)

B, R, T, F, C, N = 4, 10, 999, 64, 1, 1
spectrogram = torch.rand(B, C, F, T)    # a batch of 4, 1-channel spectrograms of 49 frequency bands and 999 time-bins
responses = torch.rand(B, R, T, N)      # a batch of 4 responses of 999 time-bins for 1 neuron, each with 4 repeats

prediction = model(responses)           # shape: (B, 1, T, 1)

from metrics import correlation_coefficient
cc = correlation_coefficient(prediction, responses)  # (B, ) or (0,)
```
