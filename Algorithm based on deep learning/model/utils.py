import torch
import numpy as np

def set_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility for PyTorch and NumPy.
    If CUDA is available, sets the CUDA seed as well.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def weighted_pick(options, weights, k=1):
    """
    Select k unique items from options based on the given weights.
    Weights are normalized to probabilities.
    Returns the selected items and their indices.
    """
    probabilities = np.array(weights, dtype=float)
    probabilities /= probabilities.sum()
    indices = np.random.choice(len(options), size=k, replace=False, p=probabilities)
    selected_items = [options[i] for i in indices]
    return selected_items, indices.tolist()
