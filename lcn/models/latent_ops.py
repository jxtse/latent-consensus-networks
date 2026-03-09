"""Latent space operations shared by model wrappers and consensus."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch

from lcn.core.kv_cache import KVCache


def summarize_kv_cache(kv_cache: KVCache) -> torch.Tensor:
    """Compress a KV cache into a single latent vector per batch item."""
    if not kv_cache:
        raise ValueError("kv_cache must contain at least one layer")

    layer_summaries = []
    for key, value in kv_cache:
        key_summary = _reduce_cache_tensor(key)
        value_summary = _reduce_cache_tensor(value)
        layer_summaries.append((key_summary + value_summary) * 0.5)

    return torch.stack(layer_summaries, dim=0).mean(dim=0)


def stack_kv_cache_summaries(caches: Sequence[KVCache]) -> Optional[torch.Tensor]:
    """Stack summarized caches into `[batch, num_caches, hidden_dim]`."""
    if not caches:
        return None

    summaries = [summarize_kv_cache(cache) for cache in caches]
    return torch.stack(summaries, dim=1)


def cache_to_level_tensor(kv_cache: Optional[KVCache]) -> Optional[torch.Tensor]:
    """Convert an optional cache into `[batch, 1, hidden_dim]`."""
    if kv_cache is None:
        return None
    return summarize_kv_cache(kv_cache).unsqueeze(1)


def state_to_kv_cache(
    state: torch.Tensor,
    *,
    num_layers: int,
    seq_len: int,
    num_heads: int = 1,
) -> KVCache:
    """Project a latent state into a synthetic KV cache shape."""
    if state.ndim != 2:
        raise ValueError(f"state must have shape [batch, hidden_dim], got {tuple(state.shape)}")

    batch_size, hidden_dim = state.shape
    base = state.unsqueeze(1).unsqueeze(1).expand(batch_size, num_heads, seq_len, hidden_dim)

    caches = []
    for layer_idx in range(num_layers):
        offset = (layer_idx + 1) / max(num_layers, 1)
        key = base + offset
        value = base - offset
        caches.append((key.clone(), value.clone()))
    return tuple(caches)


def mean_pool_states(states: Iterable[torch.Tensor]) -> torch.Tensor:
    """Average a sequence of `[batch, hidden_dim]` tensors."""
    state_list = list(states)
    if not state_list:
        raise ValueError("states must contain at least one tensor")
    return torch.stack(state_list, dim=0).mean(dim=0)


def _reduce_cache_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Reduce `[batch, heads, seq, hidden_dim]` tensors to `[batch, hidden_dim]`."""
    if tensor.ndim < 2:
        raise ValueError(f"cache tensor must have at least 2 dims, got {tensor.ndim}")

    reduce_dims = tuple(range(1, tensor.ndim - 1))
    if not reduce_dims:
        return tensor
    return tensor.mean(dim=reduce_dims)
