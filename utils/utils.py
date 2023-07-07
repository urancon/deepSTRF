import os
import io
import random
import torch
import numpy as np


#############
# 	RNG 	#
#############

def set_random_seed(seed):
    # Python
    random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # if int(torch.__version__.split('.')[1]) < 8:
    #    torch.set_deterministic(True)  # for pytorch < 1.8
    # else:
    torch.use_deterministic_algorithms(True)

    # NumPy
    np.random.seed(seed)
