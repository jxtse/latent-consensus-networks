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
