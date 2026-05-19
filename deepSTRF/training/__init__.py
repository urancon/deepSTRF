"""``deepSTRF.training`` — opt-in training utilities.

See ``docs/_source/md/fitter.md``.
"""

from deepSTRF.training.config import auto_config, slug_from_config
from deepSTRF.training.fitter import Fitter
from deepSTRF.training.multi_seed import fit_multi_seed
from deepSTRF.training.seed import set_random_seed

__all__ = [
    "Fitter", "auto_config", "fit_multi_seed", "set_random_seed",
    "slug_from_config",
]
