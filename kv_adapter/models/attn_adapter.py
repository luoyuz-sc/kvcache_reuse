import torch
import torch.nn as nn
from typing import Optional, List
import torch.nn.functional as F
from torch.optim import AdamW
import math
import os, time

def layer_group_indices(t_idx: int, L_src: int, L_tgt: int) -> List[int]:
    radius = (L_src -1) // (L_tgt - 1)
    mid = round((t_idx * (L_src-1)) / (L_tgt-1))
    return list(range(max(mid-radius,0), min(mid+radius+1,L_src)))

# -------------------------
# ConvexAttentionAdapter
# - source_attn: [L_src, B, H_src, S, S] (probabilities along last dim)
# - returns adapted/target_attn: [L_tgt, B, H_tgt, S, S]
# Parameters:
#   - alpha_params: per target-layer, raw logits over source layers in that group (softmax -> convex weights)
#   - beta params: per target-layer, per target-head, raw logits over source-heads (softmax -> convex head mapping)
# -------------------------
class LayerAttentionFuser(nn.Module):
    def __init__(self, L_src: int, L_tgt: int, H_src: int, H_tgt: int, share_head_weights: bool = False, eps: float = 1e-12):
        """
        L_src: number of source layers (input)
        L_tgt: number of target layers (output)
        H_src: number of heads in source
        H_tgt: number of heads in target
        """
        super().__init__()
        self.L_src = L_src
        self.L_tgt = L_tgt
        self.H_src = H_src
        self.H_tgt = H_tgt
        self.share_head_weights = share_head_weights
        self.eps = eps

        # per-target-layer alpha logits for its group of source layers.
        # group sizes vary; store one parameter tensor per target-layer with length equal to its group size.
        self.alpha_params = nn.ParameterList()
        self.group_indices = []
        for t in range(L_tgt):
            idxs = layer_group_indices(t, L_src, L_tgt)
            self.group_indices.append(idxs)
            # one raw-logit parameter per source-layer in the group
            p = nn.Parameter(torch.ones(len(idxs)))
            self.alpha_params.append(p)

        # small init: slightly random to break symmetry
        self.reset_parameters()

    def reset_parameters(self):
        # initialize alpha logits small normal so initial softmax ~ uniform
        for p in self.alpha_params:
            with torch.no_grad():
                p.zero_()
                p[p.shape[0]//2].copy_(torch.tensor(1.0))
            
    def forward(self, source_attn: torch.Tensor, layer_index: int) -> torch.Tensor:
        """
        source_attn: [W, B, H_tgt, S, S]
        returns:
            adapted_target_attn: [B, H_tgt, S, S]  (probabilities along last dim)
        """
        W, B, H_tgt, S1, S2 = source_attn.shape
        assert layer_index < self.L_src and H_tgt == self.H_tgt and S1 == S2, "source_attn shape mismatch"


        source_probs = source_attn.permute(1, 2, 3, 4, 0)  # [B, H_tgt, S, S, W]
        # verify non-negativity roughly
        if source_probs.min() < -1e-6:
            raise ValueError("source_attn contains negative values but input_is_logits=False. Provide probabilities or set input_is_logits=True.")

        device = source_probs.device
        dtype = source_probs.dtype
        
        raw_w = self.alpha_params[layer_index]
        weight = F.softmax(raw_w, dim=0).reshape(W, 1).to(dtype=source_attn.dtype)
        
        source_probs = source_probs @ weight.to(source_probs.device)
        out = source_probs.squeeze(dim=-1)
        return out
    
class HeadAttentionFuser(nn.Module):
    def __init__(self, L_src: int, L_tgt: int, H_src: int, H_tgt: int, share_head_weights: bool = False, eps: float = 1e-12):
        """
        L_src: number of source layers (input)
        L_tgt: number of target layers (output)
        H_src: number of heads in source
        H_tgt: number of heads in target
        """
        super().__init__()
        self.L_src = L_src
        self.L_tgt = L_tgt
        self.H_src = H_src
        self.H_tgt = H_tgt
        self.share_head_weights = share_head_weights
        self.eps = eps

        # per-target-layer head-mapping beta: shape [L_tgt, H_tgt, H_src] or shared [H_tgt, H_src]
        self.beta_params = nn.ParameterList()
        if share_head_weights:
            # single matrix: H_tgt x H_src
            self.beta_params.append (nn.Parameter(torch.zeros(H_src, H_tgt)))
        else:
            # per-target-layer
            for _ in range(L_src):
                self.beta_params.append(nn.Parameter(torch.zeros(H_src, H_tgt)))

        # small init: slightly random to break symmetry
        self.reset_parameters()

    def reset_parameters(self):
        # initialize alpha logits small normal so initial softmax ~ uniform
        dim = min(self.H_src, self.H_tgt)
        for p in self.beta_params:
            with torch.no_grad():
                p.zero_()
                p[:dim, :dim].copy_(torch.eye(dim))
            
    def forward(self, source_attn: torch.Tensor, layer_index: int) -> torch.Tensor:
        """
        source_attn: [B, H_src, S, S]
        returns:
            adapted_target_attn: [B, H_tgt, S, S]  (probabilities along last dim)
        """
        B, H_src, S1, S2 = source_attn.shape
        assert layer_index < self.L_src and H_src == self.H_src and S1 == S2, "source_attn shape mismatch"


        source_probs = source_attn.permute(0, 2, 3, 1)  # [B, S, S, H_src]
        # verify non-negativity roughly
        if source_probs.min() < -1e-6:
            raise ValueError("source_attn contains negative values but input_is_logits=False. Provide probabilities or set input_is_logits=True.")

        device = source_probs.device
        dtype = source_probs.dtype
        
        if self.share_head_weights:
            raw_beta = self.beta_params[0]
        else:
            raw_beta = self.beta_params[layer_index]
        weight = F.softmax(raw_beta, dim=0)
        
        weight = weight.to(device=device,dtype=dtype)
        
        source_probs = source_probs @ weight  # [B, S, S, H_tgt]
        out = source_probs.permute(0, 3, 1, 2)
        return out