"""``deepSTRF.training`` — opt-in training utilities.

See ``docs/_source/md/fitter.md``.
"""

from deepSTRF.training.fitter import Fitter
from deepSTRF.training.multi_seed import fit_multi_seed
from deepSTRF.training.seed import set_random_seed

__all__ = ["Fitter", "fit_multi_seed", "set_random_seed"]
