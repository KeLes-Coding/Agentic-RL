"""
CCAPO v4.1 Dual-Stream Advantage Computation

Computes A_total = A_macro + β · A_micro where:
- A_macro: GRPO-style z-score normalization on R_tau (episode-level)
- A_micro: z-score normalization on (Q_STDB - V̄) per step (with σ_min floor)
"""
import torch
import numpy as np
from collections import defaultdict
from typing import Optional


def compute_ccapo_dual_stream_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    a_micro_raw: np.ndarray,
    beta_micro: float = 0.5,
    sigma_min: float = 0.1,
    norm_adv_by_std: bool = True,
    epsilon: float = 1e-6,
) -> tuple:
    """
    Compute CCAPO v4.1 dual-stream advantage.
    
    Args:
        token_level_rewards: (bs, response_length) - contains R_tau injected at last token
        response_mask: (bs, response_length) - attention mask for response tokens
        index: (bs,) - prompt/uid index for GRPO grouping
        traj_index: (bs,) - trajectory uid
        a_micro_raw: (bs,) - per-sample raw micro advantage from STDB (step-aligned in multi-turn rollout)
        beta_micro: float - fusion weight for A_micro
        sigma_min: float - minimum std for A_micro normalization
        norm_adv_by_std: bool - whether to normalize A_macro by std (True=GRPO, False=DrGRPO)
        epsilon: float - numerical stability
    
    Returns:
        advantages: (bs, response_length) - fused advantages
        returns: (bs, response_length) - same as advantages for compatibility
    """
    bsz, response_length = token_level_rewards.shape
    device = token_level_rewards.device
    
    # =====================================================
    # Step 1: Compute A_macro (GRPO z-score on R_tau)
    # =====================================================
    # R_tau is the sum of token_level_rewards per trajectory
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()
    
    with torch.no_grad():
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            seen_pairs.add((index[i], traj_index[i]))
        
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0, device=device)
                id2std[idx] = torch.tensor(1.0, device=device)
            else:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
                id2std[idx] = torch.std(torch.stack(id2score[idx]))
        
        a_macro = torch.zeros(bsz, device=device)
        for i in range(bsz):
            if norm_adv_by_std:
                a_macro[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                a_macro[i] = scores[i] - id2mean[index[i]]
    
    # =====================================================
    # Step 2: Compute A_micro (z-score on a_micro_raw)
    # =====================================================
    if not isinstance(a_micro_raw, torch.Tensor):
        # non_tensor_batch may carry numpy object arrays; force a stable float array first
        a_micro_raw_np = np.asarray(a_micro_raw, dtype=np.float32)
        a_micro_raw_t = torch.tensor(a_micro_raw_np, dtype=torch.float32, device=device)
    else:
        a_micro_raw_t = a_micro_raw.to(device=device, dtype=torch.float32)
    
    # Batch-level z-score normalization with sigma_min floor
    micro_mean = a_micro_raw_t.mean()
    micro_std = a_micro_raw_t.std()
    micro_std = torch.max(micro_std, torch.tensor(sigma_min, device=device))
    
    a_micro = (a_micro_raw_t - micro_mean) / micro_std
    
    # =====================================================
    # Step 3: Fuse: A_total = A_macro + β · A_micro
    # =====================================================
    a_total = a_macro + beta_micro * a_micro  # (bs,)
    
    # =====================================================
    # Step 4: Broadcast to token level
    # =====================================================
    advantages = a_total.unsqueeze(-1) * response_mask  # (bs, response_length)
    
    return advantages, advantages
