"""``deepSTRF.utils`` — cross-cutting helpers.

For dataset-side helpers (``concat_neural_datasets``, ``neural_collate``,
``hanning_smooth``, ...) see :mod:`deepSTRF.utils.data`. For training
utilities (``Fitter``, ``set_random_seed``) see :mod:`deepSTRF.training`.
For shared notebook plotting (stim+response panels, PSTH-vs-prediction
overlays) see :mod:`deepSTRF.utils.plotting`.
"""

from .data import (
    concat_neural_datasets,
    hanning_smooth,
    neural_collate,
    ResponseSmoothingTransform,
)
from .plotting import (
    plot_psth_vs_pred,
    plot_stim_with_response,
)

__all__ = [
    "concat_neural_datasets",
    "hanning_smooth",
    "neural_collate",
    "ResponseSmoothingTransform",
    "plot_psth_vs_pred",
    "plot_stim_with_response",
]
