"""``deepSTRF.training`` — opt-in training utilities.

See ``docs/_source/md/fitter.md``.
"""

from deepSTRF.training.fitter import Fitter
from deepSTRF.training.per_neuron_fitter import PerNeuronFitter
from deepSTRF.training.seed import set_random_seed

__all__ = ["Fitter", "PerNeuronFitter", "set_random_seed"]
