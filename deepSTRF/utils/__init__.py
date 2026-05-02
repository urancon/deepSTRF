"""``deepSTRF.utils`` — cross-cutting helpers.

For dataset-side helpers (``concat_neural_datasets``, ``neural_collate``,
``hanning_smooth``, ...) see :mod:`deepSTRF.utils.data`. For training
utilities (``Fitter``, ``set_random_seed``) see :mod:`deepSTRF.training`.
"""

from .data import (
    concat_neural_datasets,
    hanning_smooth,
    neural_collate,
    ResponseSmoothingTransform,
)

__all__ = [
    "concat_neural_datasets",
    "hanning_smooth",
    "neural_collate",
    "ResponseSmoothingTransform",
]
