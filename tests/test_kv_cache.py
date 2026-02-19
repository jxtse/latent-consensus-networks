# tests/test_kv_cache.py
import pytest
import torch
from lcn.core.kv_cache import HierarchicalKVCache


class TestHierarchicalKVCache:
    """Tests for HierarchicalKVCache data structure."""

    def test_init_creates_empty_caches(self):
        """Cache should initialize with empty local, group, and global caches."""
        cache = HierarchicalKVCache(num_groups=2, agents_per_group=3)

        assert cache.num_groups == 2
        assert cache.agents_per_group == 3
        assert len(cache.local_caches) == 0
        assert len(cache.group_caches) == 0
        assert cache.global_cache is None

    def test_update_local_stores_kv(self):
        """update_local should store KV-Cache for an agent."""
        cache = HierarchicalKVCache(num_groups=2, agents_per_group=3)

        # Create mock KV-Cache (tuple of (key, value) tensors per layer)
        mock_kv = self._create_mock_kv(num_layers=2, seq_len=10, hidden_dim=64)

        cache.update_local(agent_id=0, group_id=0, kv_cache=mock_kv)

        assert 0 in cache.local_caches
        assert cache.local_caches[0] is mock_kv

    def test_get_group_members(self):
        """get_group_members should return all agents in a group."""
        cache = HierarchicalKVCache(num_groups=2, agents_per_group=3)

        # Add agents to groups
        for agent_id in range(6):
            group_id = agent_id // 3
            mock_kv = self._create_mock_kv(num_layers=2, seq_len=10, hidden_dim=64)
            cache.update_local(agent_id=agent_id, group_id=group_id, kv_cache=mock_kv)

        group_0_members = cache.get_group_members(group_id=0)
        group_1_members = cache.get_group_members(group_id=1)

        assert set(group_0_members) == {0, 1, 2}
        assert set(group_1_members) == {3, 4, 5}

    def test_get_neighbors_returns_same_group_agents(self):
        """get_neighbors should return other agents in the same group."""
        cache = HierarchicalKVCache(num_groups=2, agents_per_group=3)

        for agent_id in range(6):
            group_id = agent_id // 3
            mock_kv = self._create_mock_kv(num_layers=2, seq_len=10, hidden_dim=64)
            cache.update_local(agent_id=agent_id, group_id=group_id, kv_cache=mock_kv)

        neighbors = cache.get_neighbors(agent_id=0)

        # Agent 0 should have neighbors 1 and 2 (same group), not itself
        assert set(neighbors) == {1, 2}

    @staticmethod
    def _create_mock_kv(num_layers: int, seq_len: int, hidden_dim: int):
        """Helper to create mock KV-Cache."""
        return tuple(
            (
                torch.randn(1, 1, seq_len, hidden_dim),  # key
                torch.randn(1, 1, seq_len, hidden_dim),  # value
            )
            for _ in range(num_layers)
        )
