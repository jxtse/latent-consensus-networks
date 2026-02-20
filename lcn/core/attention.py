# lcn/core/attention.py
"""Cross-Level Attention mechanism for LCN."""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossLevelAttention(nn.Module):
    """
    Cross-Level Attention mechanism for fusing information from
    Local, Group, and Global KV-Cache levels.

    The agent's current state serves as the query, attending to
    representations from all three hierarchy levels.

    Args:
        hidden_dim: Dimension of hidden states
        temperature: Softmax temperature for attention (lower = sharper)
    """

    def __init__(self, hidden_dim: int, temperature: float = 1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature

        # Query projection
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Key projection (shared across levels for simplicity)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Initialize with identity-like weights for stability
        nn.init.eye_(self.query_proj.weight)
        nn.init.eye_(self.key_proj.weight)

    def forward(
        self,
        query_state: torch.Tensor,
        local_repr: Optional[torch.Tensor],
        group_repr: Optional[torch.Tensor],
        global_repr: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cross-level attention and fuse representations.

        Args:
            query_state: Agent's current state [B, D]
            local_repr: Local level representations [B, L1, D] or None
            group_repr: Group level representations [B, L2, D] or None
            global_repr: Global level representations [B, L3, D] or None

        Returns:
            fused_state: Attention-weighted fusion [B, D]
            attn_weights: Attention weights [B, L_total]
        """
        # Collect non-None representations
        reprs = []
        if local_repr is not None:
            reprs.append(local_repr)
        if group_repr is not None:
            reprs.append(group_repr)
        if global_repr is not None:
            reprs.append(global_repr)

        if not reprs:
            # No context available, return query state unchanged
            return query_state, torch.ones(query_state.shape[0], 1, device=query_state.device)

        # Concatenate all levels: [B, L_total, D]
        all_repr = torch.cat(reprs, dim=1)

        # Project query and keys
        query = self.query_proj(query_state).unsqueeze(1)  # [B, 1, D]
        keys = self.key_proj(all_repr)  # [B, L_total, D]

        # Compute attention scores
        scores = torch.bmm(query, keys.transpose(1, 2))  # [B, 1, L_total]
        scores = scores / (self.hidden_dim ** 0.5)  # Scale
        scores = scores / self.temperature  # Temperature

        # Softmax to get weights
        attn_weights = F.softmax(scores, dim=-1).squeeze(1)  # [B, L_total]

        # Weighted sum
        fused_state = torch.bmm(
            attn_weights.unsqueeze(1),  # [B, 1, L_total]
            all_repr  # [B, L_total, D]
        ).squeeze(1)  # [B, D]

        return fused_state, attn_weights

    def get_level_weights(
        self,
        attn_weights: torch.Tensor,
        local_len: int,
        group_len: int,
        global_len: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate attention weights per hierarchy level.

        Args:
            attn_weights: Full attention weights [B, L_total]
            local_len: Number of local tokens
            group_len: Number of group tokens
            global_len: Number of global tokens

        Returns:
            Dict with 'local', 'group', 'global' keys containing
            summed attention weights per level [B]
        """
        idx = 0
        result = {}

        if local_len > 0:
            result["local"] = attn_weights[:, idx:idx + local_len].sum(dim=-1)
            idx += local_len
        else:
            result["local"] = torch.zeros(attn_weights.shape[0], device=attn_weights.device)

        if group_len > 0:
            result["group"] = attn_weights[:, idx:idx + group_len].sum(dim=-1)
            idx += group_len
        else:
            result["group"] = torch.zeros(attn_weights.shape[0], device=attn_weights.device)

        if global_len > 0:
            result["global"] = attn_weights[:, idx:idx + global_len].sum(dim=-1)
        else:
            result["global"] = torch.zeros(attn_weights.shape[0], device=attn_weights.device)

        return result
