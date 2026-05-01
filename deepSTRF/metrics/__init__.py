from deepSTRF.metrics.losses import mse_loss, poisson_loss
from deepSTRF.metrics.performance import (
    coherence,
    corrcoef,
    fve,
    noise_power,
    normalized_corrcoef,
    signal_power,
    snr,
    # Deprecated aliases — kept until utils/training*.py is replaced by the
    # forthcoming Fitter (refactor/training-utility).
    correlation_coefficient,
    normalized_correlation_coefficient,
)

__all__ = [
    # losses
    "mse_loss",
    "poisson_loss",
    # prediction-vs-PSTH metrics
    "corrcoef",
    "fve",
    "normalized_corrcoef",
    "coherence",
    # responses-only metrics
    "signal_power",
    "noise_power",
    "snr",
]
