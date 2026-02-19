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

    def aggregate_group(self, group_id: int) -> None:
        """
        Aggregate local caches within a group to create group-level cache.

        Uses mean pooling across all agents in the group.

        Args:
            group_id: The group to aggregate
        """
        members = self.get_group_members(group_id)
        if not members:
            return

        member_caches = [self.local_caches[aid] for aid in members if aid in self.local_caches]
        if not member_caches:
            return

        self.group_caches[group_id] = self._mean_pool_kv_caches(member_caches)

    def aggregate_global(self) -> None:
        """
        Aggregate all group caches to create global-level cache.

        Uses weighted mean pooling based on group sizes.
        """
        if not self.group_caches:
            return

        group_caches = list(self.group_caches.values())
        self.global_cache = self._mean_pool_kv_caches(group_caches)

    def get_all_levels(
        self,
        agent_id: int
    ) -> Tuple[List[KVCache], Optional[KVCache], Optional[KVCache]]:
        """
        Get all three levels of KV-Cache for an agent.

        Args:
            agent_id: The agent requesting caches

        Returns:
            Tuple of (local_neighbors, group_cache, global_cache)
            - local_neighbors: List of KV-Caches from neighbor agents
            - group_cache: Aggregated group KV-Cache
            - global_cache: Aggregated global KV-Cache
        """
        # Local: get neighbor caches
        neighbors = self.get_neighbors(agent_id)
        local_caches = [
            self.local_caches[nid]
            for nid in neighbors
            if nid in self.local_caches
        ]

        # Group: get agent's group cache
        group_id = self._agent_to_group.get(agent_id)
        group_cache = self.group_caches.get(group_id) if group_id is not None else None

        # Global
        global_cache = self.global_cache

        return local_caches, group_cache, global_cache

    @staticmethod
    def _mean_pool_kv_caches(caches: List[KVCache]) -> KVCache:
        """
        Compute mean of multiple KV-Caches.

        Args:
            caches: List of KV-Caches to average

        Returns:
            Mean-pooled KV-Cache
        """
        if len(caches) == 1:
            return caches[0]

        num_layers = len(caches[0])
        result = []

        for layer_idx in range(num_layers):
            # Stack keys and values from all caches
            keys = torch.stack([c[layer_idx][0] for c in caches], dim=0)
            values = torch.stack([c[layer_idx][1] for c in caches], dim=0)

            # Mean along the stack dimension
            mean_key = keys.mean(dim=0)
            mean_value = values.mean(dim=0)

            result.append((mean_key, mean_value))

        return tuple(result)
