import random

import numpy as np
import torch

RANDOM_SEED = 1234

# If you want to fix random seed, please call this function in training scripts.
def fix_seed():
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)  # Numpy module.
    random.seed(RANDOM_SEED)  # Python random module.
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True