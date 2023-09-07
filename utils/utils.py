import os
import io
import random
import torch
import numpy as np
import torch.backends.cudnn


#############
# 	RNG 	#
#############

def set_random_seed(seed):
    random.seed(seed)                           # Python
    np.random.seed(seed)                        # NumPy
    torch.manual_seed(seed)                     # PyTorch
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
