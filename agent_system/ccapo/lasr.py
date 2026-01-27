import torch
import numpy as np
from typing import List, Union
from .config import LASRConfig

def compute_lasr_weights(
    outcomes: Union[List[bool], torch.Tensor, np.ndarray], 
    lengths: Union[List[int], torch.Tensor, np.ndarray],
    config: LASRConfig
) -> torch.Tensor:
    """
    Computes standard LASR weights for a batch.
    
    W_len(tau) = 
        1.0                        if outcome == False (Failure Invariance)
        1.0 + beta * tanh(z_i)     if outcome == True (Success Gating)
        
    where z_i = (mu_len - len(tau_i)) / sigma_len
    (Note: Paper says "mu_len - len", so shorter is positive Z, leading to W > 1)
    """
    if not isinstance(outcomes, torch.Tensor):
        outcomes = torch.tensor(outcomes, dtype=torch.bool)
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths, dtype=torch.float32)
    else:
        lengths = lengths.float()
        
    device = lengths.device
    batch_size = len(outcomes)
    weights = torch.ones(batch_size, dtype=torch.float32, device=device)
    
    if not config.enable:
        return weights

    # Identify indices
    success_indices = torch.where(outcomes)[0]
    
    if len(success_indices) < 2:
        # Need at least 2 success samples to compute std deviation meaningful, 
        # or just return 1.0 if not enough competition.
        return weights
        
    success_lengths = lengths[success_indices]
    
    mu_len = success_lengths.mean()
    sigma_len = success_lengths.std() + 1e-6
    
    # Calculate Z scores for successful trajectories
    # "Shorter is better" -> (Mean - Len)
    z_scores = (mu_len - success_lengths) / sigma_len
    
    # Calculate weights
    # W = 1.0 + beta * tanh(z)
    modulations = config.beta * torch.tanh(z_scores)
    
    # Apply back
    weights[success_indices] = 1.0 + modulations
    
    return weights
