# LCN Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement Latent Consensus Networks (LCN) framework for NeurIPS 2026 submission

**Architecture:** Hierarchical KV-Cache (Local/Group/Global) with Cross-Level Attention, built on top of LatentMAS

**Tech Stack:** Python 3.10+, PyTorch, Transformers, Qwen models, pytest

---

## Phase 1: Foundation Framework (Week 1-2)

### Task 1: Project Structure Setup

**Files:**
- Create: `lcn/__init__.py`
- Create: `lcn/core/__init__.py`
- Create: `lcn/models/__init__.py`
- Create: `lcn/environments/__init__.py`
- Create: `lcn/utils/__init__.py`
- Create: `lcn/configs/__init__.py`
- Create: `tests/__init__.py`
- Create: `experiments/__init__.py`
- Create: `pyproject.toml`

**Step 1: Create directory structure**

```bash
mkdir -p lcn/core lcn/models lcn/environments lcn/utils lcn/configs
mkdir -p tests experiments notebooks scripts
```

**Step 2: Create pyproject.toml**

```toml
[project]
name = "lcn"
version = "0.1.0"
description = "Latent Consensus Networks for Multi-Agent Collaboration"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0.0",
    "transformers>=4.35.0",
    "pyyaml>=6.0",
    "numpy>=1.24.0",
    "tqdm>=4.65.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "isort>=5.12.0",
]

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
```

**Step 3: Create lcn/__init__.py**

```python
"""Latent Consensus Networks (LCN) - Hierarchical Multi-Agent Collaboration"""

__version__ = "0.1.0"

from lcn.core.agent import LCNAgent
from lcn.core.kv_cache import HierarchicalKVCache
from lcn.core.attention import CrossLevelAttention
from lcn.core.consensus import ConsensusProtocol

__all__ = [
    "LCNAgent",
    "HierarchicalKVCache",
    "CrossLevelAttention",
    "ConsensusProtocol",
]
```

**Step 4: Commit**

```bash
git add lcn/ tests/ experiments/ pyproject.toml
git commit -m "feat: initialize LCN project structure"
```

---

### Task 2: Implement HierarchicalKVCache - Data Structure

**Files:**
- Create: `lcn/core/kv_cache.py`
- Create: `tests/test_kv_cache.py`

**Step 1: Write the failing test**

```python
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_kv_cache.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'lcn.core.kv_cache'"

**Step 3: Write minimal implementation**

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_kv_cache.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add lcn/core/kv_cache.py tests/test_kv_cache.py
git commit -m "feat(core): implement HierarchicalKVCache data structure"
```

---

### Task 3: Implement HierarchicalKVCache - Aggregation Methods

**Files:**
- Modify: `lcn/core/kv_cache.py`
- Modify: `tests/test_kv_cache.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_kv_cache.py

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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_kv_cache.py::TestKVCacheAggregation -v`
Expected: FAIL with "AttributeError: 'HierarchicalKVCache' object has no attribute 'aggregate_group'"

**Step 3: Write minimal implementation**

```python
# Add to lcn/core/kv_cache.py

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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_kv_cache.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add lcn/core/kv_cache.py tests/test_kv_cache.py
git commit -m "feat(core): implement KV-Cache aggregation methods"
```

---

### Task 4: Implement CrossLevelAttention

**Files:**
- Create: `lcn/core/attention.py`
- Create: `tests/test_attention.py`

**Step 1: Write the failing test**

```python
# tests/test_attention.py
import pytest
import torch
from lcn.core.attention import CrossLevelAttention


class TestCrossLevelAttention:
    """Tests for CrossLevelAttention mechanism."""

    def test_init_with_default_params(self):
        """Attention should initialize with default parameters."""
        attn = CrossLevelAttention(hidden_dim=64)

        assert attn.hidden_dim == 64
        assert attn.temperature == 1.0

    def test_forward_returns_fused_state_and_weights(self):
        """forward should return fused state and attention weights."""
        attn = CrossLevelAttention(hidden_dim=64)

        batch_size = 2
        query_state = torch.randn(batch_size, 64)

        # Create mock representations for each level
        local_repr = torch.randn(batch_size, 5, 64)   # 5 local tokens
        group_repr = torch.randn(batch_size, 3, 64)   # 3 group tokens
        global_repr = torch.randn(batch_size, 2, 64)  # 2 global tokens

        fused, weights = attn(query_state, local_repr, group_repr, global_repr)

        # Fused state should have shape [B, D]
        assert fused.shape == (batch_size, 64)

        # Weights should have shape [B, L_total]
        assert weights.shape == (batch_size, 5 + 3 + 2)

        # Weights should sum to 1
        assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size), atol=1e-5)

    def test_forward_with_missing_levels(self):
        """forward should handle None inputs for missing levels."""
        attn = CrossLevelAttention(hidden_dim=64)

        batch_size = 2
        query_state = torch.randn(batch_size, 64)
        local_repr = torch.randn(batch_size, 5, 64)

        # Group and global are None
        fused, weights = attn(query_state, local_repr, None, None)

        assert fused.shape == (batch_size, 64)
        assert weights.shape == (batch_size, 5)

    def test_temperature_affects_attention_sharpness(self):
        """Lower temperature should produce sharper attention."""
        attn_sharp = CrossLevelAttention(hidden_dim=64, temperature=0.1)
        attn_smooth = CrossLevelAttention(hidden_dim=64, temperature=10.0)

        query_state = torch.randn(1, 64)
        local_repr = torch.randn(1, 10, 64)

        _, weights_sharp = attn_sharp(query_state, local_repr, None, None)
        _, weights_smooth = attn_smooth(query_state, local_repr, None, None)

        # Sharp attention should have higher max weight (more concentrated)
        assert weights_sharp.max() > weights_smooth.max()

    def test_get_level_weights_returns_per_level_summary(self):
        """get_level_weights should return aggregated weights per level."""
        attn = CrossLevelAttention(hidden_dim=64)

        batch_size = 2
        query_state = torch.randn(batch_size, 64)
        local_repr = torch.randn(batch_size, 5, 64)
        group_repr = torch.randn(batch_size, 3, 64)
        global_repr = torch.randn(batch_size, 2, 64)

        fused, weights = attn(query_state, local_repr, group_repr, global_repr)

        level_weights = attn.get_level_weights(
            weights,
            local_len=5,
            group_len=3,
            global_len=2
        )

        # Should return dict with three keys
        assert "local" in level_weights
        assert "group" in level_weights
        assert "global" in level_weights

        # Level weights should sum to 1
        total = level_weights["local"] + level_weights["group"] + level_weights["global"]
        assert torch.allclose(total, torch.ones(batch_size), atol=1e-5)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_attention.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_attention.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add lcn/core/attention.py tests/test_attention.py
git commit -m "feat(core): implement CrossLevelAttention mechanism"
```

---

### Task 5: Implement LCNAgent

**Files:**
- Create: `lcn/core/agent.py`
- Create: `tests/test_agent.py`

**Step 1: Write the failing test**

```python
# tests/test_agent.py
import pytest
import torch
from lcn.core.agent import LCNAgent


class TestLCNAgent:
    """Tests for LCNAgent."""

    def test_init_with_required_params(self):
        """Agent should initialize with required parameters."""
        agent = LCNAgent(
            agent_id=0,
            group_id=0,
            hidden_dim=64,
        )

        assert agent.agent_id == 0
        assert agent.group_id == 0
        assert agent.hidden_dim == 64
        assert agent.state is None

    def test_init_with_persona(self):
        """Agent should accept optional persona."""
        agent = LCNAgent(
            agent_id=0,
            group_id=0,
            hidden_dim=64,
            persona="A skeptical scientist",
        )

        assert agent.persona == "A skeptical scientist"

    def test_set_state(self):
        """set_state should update agent's hidden state."""
        agent = LCNAgent(agent_id=0, group_id=0, hidden_dim=64)

        state = torch.randn(1, 64)
        agent.set_state(state)

        assert agent.state is not None
        assert torch.equal(agent.state, state)

    def test_get_state_returns_clone(self):
        """get_state should return a clone to prevent mutation."""
        agent = LCNAgent(agent_id=0, group_id=0, hidden_dim=64)

        state = torch.randn(1, 64)
        agent.set_state(state)

        retrieved = agent.get_state()
        retrieved[0, 0] = 999.0  # Mutate

        # Original should be unchanged
        assert agent.state[0, 0] != 999.0

    def test_reset_clears_state(self):
        """reset should clear agent's state."""
        agent = LCNAgent(agent_id=0, group_id=0, hidden_dim=64)

        agent.set_state(torch.randn(1, 64))
        agent.reset()

        assert agent.state is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# lcn/core/agent.py
"""LCN Agent definition."""

from typing import Optional
import torch


class LCNAgent:
    """
    Agent in Latent Consensus Networks.

    Each agent belongs to a group and maintains its own hidden state.

    Args:
        agent_id: Unique identifier for the agent
        group_id: Group the agent belongs to
        hidden_dim: Dimension of hidden states
        persona: Optional persona description for the agent
    """

    def __init__(
        self,
        agent_id: int,
        group_id: int,
        hidden_dim: int,
        persona: Optional[str] = None,
    ):
        self.agent_id = agent_id
        self.group_id = group_id
        self.hidden_dim = hidden_dim
        self.persona = persona

        self.state: Optional[torch.Tensor] = None

    def set_state(self, state: torch.Tensor) -> None:
        """Set the agent's hidden state."""
        self.state = state.clone()

    def get_state(self) -> Optional[torch.Tensor]:
        """Get a clone of the agent's hidden state."""
        if self.state is None:
            return None
        return self.state.clone()

    def reset(self) -> None:
        """Reset the agent's state."""
        self.state = None

    def __repr__(self) -> str:
        return f"LCNAgent(id={self.agent_id}, group={self.group_id}, persona={self.persona})"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add lcn/core/agent.py tests/test_agent.py
git commit -m "feat(core): implement LCNAgent"
```

---

### Task 6: Implement ConsensusProtocol - Basic Structure

**Files:**
- Create: `lcn/core/consensus.py`
- Create: `tests/test_consensus.py`

**Step 1: Write the failing test**

```python
# tests/test_consensus.py
import pytest
import torch
from unittest.mock import Mock, MagicMock

from lcn.core.consensus import ConsensusProtocol
from lcn.core.agent import LCNAgent
from lcn.core.kv_cache import HierarchicalKVCache
from lcn.core.attention import CrossLevelAttention


class TestConsensusProtocol:
    """Tests for ConsensusProtocol."""

    def test_init_with_components(self):
        """Protocol should initialize with required components."""
        kv_cache = HierarchicalKVCache(num_groups=2, agents_per_group=2)
        attention = CrossLevelAttention(hidden_dim=64)

        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=3,
            latent_steps=5,
        )

        assert protocol.kv_cache is kv_cache
        assert protocol.attention is attention
        assert protocol.num_rounds == 3
        assert protocol.latent_steps == 5

    def test_register_agents(self):
        """register_agents should add agents to the protocol."""
        kv_cache = HierarchicalKVCache(num_groups=2, agents_per_group=2)
        attention = CrossLevelAttention(hidden_dim=64)

        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=3,
            latent_steps=5,
        )

        agents = [
            LCNAgent(agent_id=0, group_id=0, hidden_dim=64),
            LCNAgent(agent_id=1, group_id=0, hidden_dim=64),
            LCNAgent(agent_id=2, group_id=1, hidden_dim=64),
            LCNAgent(agent_id=3, group_id=1, hidden_dim=64),
        ]

        protocol.register_agents(agents)

        assert len(protocol.agents) == 4
        assert protocol.agents[0].agent_id == 0

    def test_get_agents_by_group(self):
        """get_agents_by_group should return agents in a specific group."""
        kv_cache = HierarchicalKVCache(num_groups=2, agents_per_group=2)
        attention = CrossLevelAttention(hidden_dim=64)

        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=3,
            latent_steps=5,
        )

        agents = [
            LCNAgent(agent_id=0, group_id=0, hidden_dim=64),
            LCNAgent(agent_id=1, group_id=0, hidden_dim=64),
            LCNAgent(agent_id=2, group_id=1, hidden_dim=64),
            LCNAgent(agent_id=3, group_id=1, hidden_dim=64),
        ]

        protocol.register_agents(agents)

        group_0 = protocol.get_agents_by_group(0)
        group_1 = protocol.get_agents_by_group(1)

        assert len(group_0) == 2
        assert len(group_1) == 2
        assert all(a.group_id == 0 for a in group_0)
        assert all(a.group_id == 1 for a in group_1)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_consensus.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# lcn/core/consensus.py
"""Consensus formation protocol for LCN."""

from typing import Dict, List, Optional
import torch

from lcn.core.agent import LCNAgent
from lcn.core.kv_cache import HierarchicalKVCache
from lcn.core.attention import CrossLevelAttention


class ConsensusProtocol:
    """
    Consensus formation protocol for Latent Consensus Networks.

    Orchestrates the multi-round consensus formation process where
    agents iteratively update their states based on cross-level attention.

    Args:
        kv_cache: Hierarchical KV-Cache manager
        attention: Cross-Level Attention module
        num_rounds: Number of consensus formation rounds
        latent_steps: Number of latent reasoning steps per agent per round
    """

    def __init__(
        self,
        kv_cache: HierarchicalKVCache,
        attention: CrossLevelAttention,
        num_rounds: int = 3,
        latent_steps: int = 5,
    ):
        self.kv_cache = kv_cache
        self.attention = attention
        self.num_rounds = num_rounds
        self.latent_steps = latent_steps

        self.agents: List[LCNAgent] = []
        self._agent_map: Dict[int, LCNAgent] = {}

    def register_agents(self, agents: List[LCNAgent]) -> None:
        """
        Register agents with the protocol.

        Args:
            agents: List of agents to register
        """
        self.agents = agents
        self._agent_map = {agent.agent_id: agent for agent in agents}

    def get_agent(self, agent_id: int) -> Optional[LCNAgent]:
        """Get an agent by ID."""
        return self._agent_map.get(agent_id)

    def get_agents_by_group(self, group_id: int) -> List[LCNAgent]:
        """Get all agents in a specific group."""
        return [agent for agent in self.agents if agent.group_id == group_id]

    @property
    def group_ids(self) -> List[int]:
        """Get all unique group IDs."""
        return list(set(agent.group_id for agent in self.agents))
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_consensus.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add lcn/core/consensus.py tests/test_consensus.py
git commit -m "feat(core): implement ConsensusProtocol basic structure"
```

---

### Task 7: Update Core __init__.py

**Files:**
- Modify: `lcn/core/__init__.py`

**Step 1: Update the init file**

```python
# lcn/core/__init__.py
"""Core components for Latent Consensus Networks."""

from lcn.core.agent import LCNAgent
from lcn.core.kv_cache import HierarchicalKVCache, KVCache
from lcn.core.attention import CrossLevelAttention
from lcn.core.consensus import ConsensusProtocol

__all__ = [
    "LCNAgent",
    "HierarchicalKVCache",
    "KVCache",
    "CrossLevelAttention",
    "ConsensusProtocol",
]
```

**Step 2: Run all tests**

Run: `pytest tests/ -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add lcn/core/__init__.py
git commit -m "chore(core): update core module exports"
```

---

## Phase 1 Summary

After completing Phase 1, you will have:

- [x] Project structure with proper Python packaging
- [x] `HierarchicalKVCache` with Local/Group/Global levels and aggregation
- [x] `CrossLevelAttention` mechanism for fusing multi-level information
- [x] `LCNAgent` class for managing agent state
- [x] `ConsensusProtocol` basic structure
- [x] Unit tests for all components

**Next Phase:** Implement model integration and experiment environments.

---

## Phase 2 Preview: Model Integration & Experiments

Phase 2 will cover:

1. **Task 8-10**: Model wrapper integration with LatentMAS
2. **Task 11-13**: Hidden Profile Task environment
3. **Task 14-15**: Asch Conformity environment
4. **Task 16-17**: Wisdom of Crowds environment
5. **Task 18-20**: Baseline comparisons

---

*Plan created: 2026-02-20*
