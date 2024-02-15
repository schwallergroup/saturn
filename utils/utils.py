import torch
import random
import numpy as np

def set_seed_everywhere(seed: int, device: str):
    """Set the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

def to_tensor(array: np.array) -> torch.Tensor:
    """Convert np.array to torch.Tensor."""
    if torch.cuda.is_available():
        return torch.tensor(array, device="cuda")
    return torch.tensor(array, device="cpu")
