# How to use the repository

This file illustrates how to use the library, in particular: datasets, models and performance metrics.


## Datasets




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
model = deepSTRF.model_zoo.LinearStrfModel(T=20, F=64)

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