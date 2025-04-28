# How to use the repository

This file illustrates how to use the library, in particular: datasets, models and performance metrics.


## Datasets

Our library provide preprocessed and ready to use electrophysiology datasets ! How they function is described in the 
[note on datasets](README_datasets.md). Here is a small usage example to prepare data for a specific neuron:

```python
import deepSTRF
from deepSTRF.datasets.Wehr_Dataset import WehrDataset, WEHR_NEURONS_SPLIT_NATURAL, WEHR_VALID_NEURONS

# instanciate dataset with all neurons, or all valid ones...
timestep = 5  # ms
path = 'deepSTRF/datasets/CRCNS_AC1_Wehr/data/'
dataset = WehrDataset(path, WEHR_VALID_NEURONS, dt=timestep)

# ...then choose the specific neuron you want
target_nrn_idx = 15
dataset.select(target_nrn_idx)
```
or alternatively:
```python
# instanciate dataset with the target neuron directly
timestep = 5  # ms
target_nrn_idx = 15
path = 'deepSTRF/datasets/CRCNS_AC1_Wehr/data/'
dataset = WehrDataset(path, (target_nrn_idx,), dt=timestep)
dataset.select(0)  # only one neuron is loaded in the dataset class, so it has index #0
```

From there, you can split the dataset according to your needs (methods might slightly change across datasets):
```python
import torch
from torch.utils.data import DataLoader

# splitting
train_set, valid_set, test_set = torch.utils.data.random_split(dataset, WEHR_NEURONS_SPLIT_NATURAL[target_nrn_idx])
batch_size = 1
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_set, batch_size=1, shuffle=False)
test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False)
```
and then train:
```python
for spectrogram, responses, ccmax, ttrc in train_dataloader:
    ...  # do your things
```


## Models

Our library also includes a zoo of PyTorch models ! You can start playing with them before developing your own. 
Guidelines on how to build a compatible model are available in the [note on models](README_models.md).
Important models include:
- the classical, stateless, **Linear (L) STRF** model
- the classical, stateless, **Linear-Nonlinear (LN)** model
- the **Network Receptive Field (NRF)** model
- the **Dynamic Network (DNet)** model
- the **Recurrent-STRF** model
- **Convolutional Neural-Network (CNN)** models

Here is a minimal example of how these classes can be imported and used:
```python
import deepSTRF
model = deepSTRF.model_zoo.Linear(T=20, F=64)

# let's assume some dummy responses of an hypothetical neuron, and the corresponding model predictions
B, C, F, T = 4, 1, 18, 999
spectrogram = torch.tensor(B, C, F, T)
predicted_response = model(spectrogram)     # output shape: (N, 1, T, N)
```



## Performance Metrics

Given a prediction tensor of shape `(B, 1, T, N)`and actual responses of shape `(B, R, T, N)` 
(see the [note on tensors](README_formats.md)), a wide range of commonly used performance metrics are available:
- **raw Pearsons Correlation Coefficient**
- **normalized CCs with different kind of normalizations**
- **explained / explainable power**
- ...

Access to these functions is obtained the following way:
```python
# let's assume some dummy responses of an hypothetical neuron, and the corresponding model predictions
B, R, T, N = 4, 10, 999, 1
y_pred = torch.tensor(B, 1, T, N)
y_true = torch.tensor(B, R, T, N) 

import deepSTRF
cc_raw = deepSTRF.metrics.correlation_coefficient(y_pred, y_true) 
cc_norm = deepSTRF.metrics.normalized_correlation_coefficient(y_pred, y_true)
success = deepSTRF.metrics.sahani_success(y_pred, y_true)
```