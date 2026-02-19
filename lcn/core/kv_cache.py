# lcn/core/kv_cache.py
"""Hierarchical KV-Cache management for LCN."""

from typing import Dict, List, Optional, Tuple
import torch


# Type alias for KV-Cache: tuple of (key, value) tensors per layer
KVCache = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]


class HierarchicalKVCache:
    """
    Three-level KV-Cache management for Latent Consensus Networks.

    Levels:
    - Local: Individual agent's KV-Cache
    - Group: Aggregated KV-Cache for a group of agents
    - Global: Aggregated KV-Cache across all groups

    Args:
        num_groups: Number of agent groups
        agents_per_group: Number of agents per group
    """

    def __init__(self, num_groups: int, agents_per_group: int):
        self.num_groups = num_groups
        self.agents_per_group = agents_per_group

        # Local level: agent_id -> KVCache
        self.local_caches: Dict[int, KVCache] = {}

        # Group level: group_id -> KVCache
        self.group_caches: Dict[int, KVCache] = {}

        # Global level: single KVCache
        self.global_cache: Optional[KVCache] = None

        # Mapping: agent_id -> group_id
        self._agent_to_group: Dict[int, int] = {}

    def update_local(
        self,
        agent_id: int,
        group_id: int,
        kv_cache: KVCache
    ) -> None:
        """
        Update an agent's local KV-Cache.

        Args:
            agent_id: Unique identifier for the agent
            group_id: Group the agent belongs to
            kv_cache: The KV-Cache to store
        """
        self.local_caches[agent_id] = kv_cache
        self._agent_to_group[agent_id] = group_id

    def get_local(self, agent_id: int) -> Optional[KVCache]:
        """Get an agent's local KV-Cache."""
        return self.local_caches.get(agent_id)

    def get_group_members(self, group_id: int) -> List[int]:
        """Get all agent IDs in a group."""
        return [
            agent_id
            for agent_id, gid in self._agent_to_group.items()
            if gid == group_id
        ]

    def get_neighbors(self, agent_id: int) -> List[int]:
        """Get neighbor agent IDs (same group, excluding self)."""
        group_id = self._agent_to_group.get(agent_id)
        if group_id is None:
            return []

        return [
            aid for aid in self.get_group_members(group_id)
            if aid != agent_id
        ]

    def get_agent_group(self, agent_id: int) -> Optional[int]:
        """Get the group ID for an agent."""
        return self._agent_to_group.get(agent_id)

    @property
    def group_ids(self) -> List[int]:
        """Get all group IDs that have agents."""
        return list(set(self._agent_to_group.values()))

    @property
    def all_agent_ids(self) -> List[int]:
        """Get all agent IDs."""
        return list(self.local_caches.keys())
