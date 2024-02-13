import torch
import random
import numpy as np


def set_seed_everywhere(seed: int, device: str):
    """
    Set the seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
