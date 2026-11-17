# Decoding

Most of deepSTRF is about **encoding**: predict neural responses from the
stimulus. **Decoding** goes the other way: predict the sound — its spectrogram —
from the neural activity. It is the natural direction for vocal-motor areas,
whose activity precedes the sound, and the classic stimulus-reconstruction
analysis for sensory areas.

The decoding API reuses the dataset, metric and training machinery unchanged.
A dataset item is the same dict; only the roles of `stims` and `responses` swap.

| | Encoding | Decoding |
|---|---|---|
| Model input | `stims` `(B, 1, F, T)` | `responses` `(B, N, R, T)` |
| Model output / target | `responses` | `stims` `(B, 1, F, T)` |
| Per-unit metrics are over | neurons | frequency bands |
| Base class | `AudioEncodingModel` | `AudioDecodingModel` |


## Models

All decoders subclass `deepSTRF.models.audio.decoding.AudioDecodingModel` and
carry `direction = "decoding"`. They take `(B, N, R, T)` responses (repeats are
averaged, missing neurons count as silent) and return a `(B, 1, F, T)`
spectrogram.

- **`WienerFilter(n_neurons, n_frequency_bands, n_lags=10, lag_offset=0)`** — the
  linear baseline: each band is a weighted sum of every neuron's activity over
  `n_lags` bins. With `lag_offset >= 0` it is causal (reads activity up to the
  current bin), the right choice for motor areas; for sensory areas, whose
  responses lag the sound, use a negative `lag_offset` to read future bins.
  Fit it in closed form with `.fit(loader, alpha=...)` (ridge regression), or
  train it by gradient descent like any model.
- **`MeanTargetBaseline(n_neurons, n_frequency_bands)`** — a control that ignores
  the neural input and predicts the mean training spectrogram, aligned on item
  onset. Report it next to any decoder: a decoder only earns credit for what it
  predicts beyond it.

Your own decoder only needs to implement `decode(x)`, mapping repeat-averaged
activity `(B, N, T)` to `(B, 1, F, T)`.


## Metrics

Both take `(B, 1, F, T)` **natural-log power** spectrograms; NaN target frames
(padding) are ignored.

- `deepSTRF.metrics.spectrogram_corrcoef(pred, target, reduction='mean')` —
  Pearson r per frequency band, over all valid time bins, then averaged over
  bands (`reduction='none'` returns one value per band).
- `deepSTRF.metrics.mel_cepstral_distortion(pred, target)` — distance between
  mel-cepstra, in dB (lower is better).


## Training

`Fitter` reads `model.direction`. For a decoder it feeds `batch['responses']`
to the model, compares the output with `batch['stims']`, masks padded frames,
and reports per-band metrics; the default monitor is `val_cc` (per-band Pearson
r, averaged).

```python
from torch.utils.data import DataLoader
from deepSTRF.datasets.audio import Yang2025Dataset
from deepSTRF.models.audio.decoding import WienerFilter, MeanTargetBaseline, valid_time_mask
from deepSTRF.metrics import spectrogram_corrcoef
from deepSTRF.utils import neural_collate

def split(name):
    ds = Yang2025Dataset(download=True)
    ds.select_pop_by_nrn_attr("bird", "Ti81")      # 49 neurons
    ds.select_stims_by_attr("split", name)
    return DataLoader(ds, batch_size=8, collate_fn=neural_collate)

train, val, test = split("train"), split("val"), split("test")
N, F = train.dataset.n_selected, train.dataset.F

wiener = WienerFilter(N, F, n_lags=40).fit(train, alpha=100.0)
control = MeanTargetBaseline(N, F).fit(train)

batch = next(iter(test))
mask = valid_time_mask(batch["responses"])[:, None, None, :]   # drop batch padding
for model in (wiener, control):
    print(spectrogram_corrcoef(model(batch["responses"]), batch["stims"], mask=mask))
```

To train a non-linear decoder, pass it to `Fitter(model, train, val,
ckpt_path=...)` as usual.

Datasets with a decoding use: [Yang 2025](README_Yang_2025.md) (budgerigar
vocal-motor cortex, the bird's own vocalisations).
