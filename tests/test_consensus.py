# tests/test_consensus.py
import pytest
import torch
from unittest.mock import Mock, MagicMock

from lcn.core.consensus import ConsensusProtocol
from lcn.core.agent import LCNAgent
from lcn.core.kv_cache import HierarchicalKVCache
from lcn.core.attention import CrossLevelAttention
from lcn.core.results import ConsensusResult


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


class TestConsensusResult:
    """Tests for ConsensusResult dataclass."""

    def test_create_with_all_fields(self):
        """ConsensusResult should be created with all required fields."""
        result = ConsensusResult(
            decision="Candidate C",
            agent_decisions={0: "Candidate A", 1: "Candidate C", 2: "Candidate C"},
            attention_history=[
                {"round": 0, "weights": [0.33, 0.33, 0.34]},
                {"round": 1, "weights": [0.25, 0.35, 0.40]},
            ],
            convergence_round=2,
        )

        assert result.decision == "Candidate C"
        assert result.agent_decisions == {0: "Candidate A", 1: "Candidate C", 2: "Candidate C"}
        assert len(result.attention_history) == 2
        assert result.convergence_round == 2

    def test_create_with_no_convergence(self):
        """ConsensusResult should allow None for convergence_round."""
        result = ConsensusResult(
            decision="No consensus",
            agent_decisions={0: "A", 1: "B", 2: "C"},
            attention_history=[],
            convergence_round=None,
        )

        assert result.decision == "No consensus"
        assert result.convergence_round is None

    def test_create_with_empty_attention_history(self):
        """ConsensusResult should allow empty attention history."""
        result = ConsensusResult(
            decision="Candidate B",
            agent_decisions={0: "B"},
            attention_history=[],
            convergence_round=0,
        )

        assert result.attention_history == []
        assert result.convergence_round == 0

    def test_is_dataclass(self):
        """ConsensusResult should be a proper dataclass."""
        from dataclasses import is_dataclass, fields

        assert is_dataclass(ConsensusResult)
        field_names = {f.name for f in fields(ConsensusResult)}
        assert field_names == {"decision", "agent_decisions", "attention_history", "convergence_round"}

    def test_equality(self):
        """Two ConsensusResults with same values should be equal."""
        result1 = ConsensusResult(
            decision="X",
            agent_decisions={0: "X"},
            attention_history=[{"a": 1}],
            convergence_round=1,
        )
        result2 = ConsensusResult(
            decision="X",
            agent_decisions={0: "X"},
            attention_history=[{"a": 1}],
            convergence_round=1,
        )

        assert result1 == result2

    def test_inequality(self):
        """Two ConsensusResults with different values should not be equal."""
        result1 = ConsensusResult(
            decision="X",
            agent_decisions={0: "X"},
            attention_history=[],
            convergence_round=1,
        )
        result2 = ConsensusResult(
            decision="Y",
            agent_decisions={0: "Y"},
            attention_history=[],
            convergence_round=2,
        )

        assert result1 != result2

    def test_field_types(self):
        """ConsensusResult fields should have correct types in annotations."""
        from typing import get_type_hints, Dict, List, Optional

        hints = get_type_hints(ConsensusResult)

        assert hints["decision"] == str
        assert hints["agent_decisions"] == Dict[int, str]
        assert hints["attention_history"] == List[Dict]
        assert hints["convergence_round"] == Optional[int]
