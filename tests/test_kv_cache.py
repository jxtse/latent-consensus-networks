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


class TestKVCacheAggregation:
    """Tests for KV-Cache aggregation methods."""

    def test_aggregate_group_creates_group_cache(self):
        """aggregate_group should create a group-level cache from local caches."""
        cache = HierarchicalKVCache(num_groups=2, agents_per_group=2)

        # Add two agents to group 0
        kv1 = self._create_mock_kv(num_layers=2, seq_len=10, hidden_dim=64)
        kv2 = self._create_mock_kv(num_layers=2, seq_len=10, hidden_dim=64)

        cache.update_local(agent_id=0, group_id=0, kv_cache=kv1)
        cache.update_local(agent_id=1, group_id=0, kv_cache=kv2)

        cache.aggregate_group(group_id=0)

        assert 0 in cache.group_caches
        group_kv = cache.group_caches[0]

        # Check shape: should have same structure as input
        assert len(group_kv) == 2  # num_layers
        assert group_kv[0][0].shape == kv1[0][0].shape  # key shape
        assert group_kv[0][1].shape == kv1[0][1].shape  # value shape

    def test_aggregate_group_computes_mean(self):
        """aggregate_group should compute mean of local caches."""
        cache = HierarchicalKVCache(num_groups=1, agents_per_group=2)

        # Create KV-Caches with known values
        kv1 = ((torch.ones(1, 1, 4, 8) * 2, torch.ones(1, 1, 4, 8) * 2),)
        kv2 = ((torch.ones(1, 1, 4, 8) * 4, torch.ones(1, 1, 4, 8) * 4),)

        cache.update_local(agent_id=0, group_id=0, kv_cache=kv1)
        cache.update_local(agent_id=1, group_id=0, kv_cache=kv2)

        cache.aggregate_group(group_id=0)

        group_kv = cache.group_caches[0]

        # Mean of 2 and 4 should be 3
        assert torch.allclose(group_kv[0][0], torch.ones(1, 1, 4, 8) * 3)
        assert torch.allclose(group_kv[0][1], torch.ones(1, 1, 4, 8) * 3)

    def test_aggregate_global_creates_global_cache(self):
        """aggregate_global should create a global cache from group caches."""
        cache = HierarchicalKVCache(num_groups=2, agents_per_group=2)

        # Setup: add agents and aggregate groups
        for agent_id in range(4):
            group_id = agent_id // 2
            kv = self._create_mock_kv(num_layers=2, seq_len=10, hidden_dim=64)
            cache.update_local(agent_id=agent_id, group_id=group_id, kv_cache=kv)

        cache.aggregate_group(group_id=0)
        cache.aggregate_group(group_id=1)
        cache.aggregate_global()

        assert cache.global_cache is not None
        assert len(cache.global_cache) == 2  # num_layers

    def test_get_all_levels_returns_three_levels(self):
        """get_all_levels should return local, group, and global caches."""
        cache = HierarchicalKVCache(num_groups=2, agents_per_group=2)

        # Setup full hierarchy
        for agent_id in range(4):
            group_id = agent_id // 2
            kv = self._create_mock_kv(num_layers=2, seq_len=10, hidden_dim=64)
            cache.update_local(agent_id=agent_id, group_id=group_id, kv_cache=kv)

        cache.aggregate_group(group_id=0)
        cache.aggregate_group(group_id=1)
        cache.aggregate_global()

        local, group, global_ = cache.get_all_levels(agent_id=0)

        # Local should be list of neighbor KV-Caches
        assert isinstance(local, list)
        assert len(local) == 1  # 1 neighbor in group of 2

        # Group and global should be single KV-Caches
        assert group is not None
        assert global_ is not None

    @staticmethod
    def _create_mock_kv(num_layers: int, seq_len: int, hidden_dim: int):
        return tuple(
            (
                torch.randn(1, 1, seq_len, hidden_dim),
                torch.randn(1, 1, seq_len, hidden_dim),
            )
            for _ in range(num_layers)
        )
