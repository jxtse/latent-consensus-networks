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


class TestBuildAgentPrompt:
    """Tests for ConsensusProtocol._build_agent_prompt method."""

    def test_build_agent_prompt_returns_list_of_dicts(self):
        """_build_agent_prompt should return a list of message dicts."""
        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=1)
        attention = CrossLevelAttention(hidden_dim=64)

        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=3,
            latent_steps=5,
        )

        agent = LCNAgent(agent_id=0, group_id=0, hidden_dim=64, persona="You are a helpful assistant.")
        task = "What is 2 + 2?"

        messages = protocol._build_agent_prompt(agent, task)

        assert isinstance(messages, list)
        assert len(messages) > 0
        assert all(isinstance(m, dict) for m in messages)

    def test_build_agent_prompt_includes_user_role(self):
        """_build_agent_prompt should include a message with role 'user'."""
        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=1)
        attention = CrossLevelAttention(hidden_dim=64)

        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=3,
            latent_steps=5,
        )

        agent = LCNAgent(agent_id=0, group_id=0, hidden_dim=64, persona="You are a scientist.")
        task = "Explain photosynthesis."

        messages = protocol._build_agent_prompt(agent, task)

        roles = [m.get("role") for m in messages]
        assert "user" in roles

    def test_build_agent_prompt_includes_task_in_content(self):
        """_build_agent_prompt should include the task in message content."""
        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=1)
        attention = CrossLevelAttention(hidden_dim=64)

        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=3,
            latent_steps=5,
        )

        agent = LCNAgent(agent_id=0, group_id=0, hidden_dim=64)
        task = "What is the capital of France?"

        messages = protocol._build_agent_prompt(agent, task)

        # Check that task appears in at least one message content
        all_content = " ".join(m.get("content", "") for m in messages)
        assert task in all_content

    def test_build_agent_prompt_includes_persona_when_present(self):
        """_build_agent_prompt should include persona in messages when agent has one."""
        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=1)
        attention = CrossLevelAttention(hidden_dim=64)

        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=3,
            latent_steps=5,
        )

        persona = "You are a conservative voter who values tradition."
        agent = LCNAgent(agent_id=0, group_id=0, hidden_dim=64, persona=persona)
        task = "Who should win the election?"

        messages = protocol._build_agent_prompt(agent, task)

        all_content = " ".join(m.get("content", "") for m in messages)
        assert persona in all_content

    def test_build_agent_prompt_works_without_persona(self):
        """_build_agent_prompt should work when agent has no persona."""
        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=1)
        attention = CrossLevelAttention(hidden_dim=64)

        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=3,
            latent_steps=5,
        )

        agent = LCNAgent(agent_id=0, group_id=0, hidden_dim=64, persona=None)
        task = "Solve this problem."

        messages = protocol._build_agent_prompt(agent, task)

        # Should still return valid messages
        assert isinstance(messages, list)
        assert len(messages) > 0
        all_content = " ".join(m.get("content", "") for m in messages)
        assert task in all_content

    def test_build_agent_prompt_message_has_role_and_content_keys(self):
        """Each message dict should have 'role' and 'content' keys."""
        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=1)
        attention = CrossLevelAttention(hidden_dim=64)

        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=3,
            latent_steps=5,
        )

        agent = LCNAgent(agent_id=0, group_id=0, hidden_dim=64, persona="A persona")
        task = "A task"

        messages = protocol._build_agent_prompt(agent, task)

        for msg in messages:
            assert "role" in msg
            assert "content" in msg


class TestInitializeAgents:
    """Tests for ConsensusProtocol._initialize_agents method."""

    def test_initialize_agents_sets_agent_states(self):
        """_initialize_agents should set hidden state for each agent."""
        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=2)
        attention = CrossLevelAttention(hidden_dim=64)

        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=3,
            latent_steps=5,
        )

        agents = [
            LCNAgent(agent_id=0, group_id=0, hidden_dim=64, persona="Agent 0"),
            LCNAgent(agent_id=1, group_id=0, hidden_dim=64, persona="Agent 1"),
        ]
        protocol.register_agents(agents)

        # Mock model wrapper
        mock_model = MagicMock()
        mock_model.prepare_input = MagicMock(return_value=(
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[1, 1, 1]])
        ))

        # Mock generate_latent to return kv_cache and hidden_state
        batch_size, hidden_dim = 1, 64
        num_layers, num_heads, seq_len, head_dim = 2, 4, 3, 16
        mock_kv_cache = tuple(
            (
                torch.randn(batch_size, num_heads, seq_len, head_dim),
                torch.randn(batch_size, num_heads, seq_len, head_dim),
            )
            for _ in range(num_layers)
        )
        mock_hidden = torch.randn(batch_size, hidden_dim)
        mock_model.generate_latent = MagicMock(return_value=(mock_kv_cache, mock_hidden))

        protocol._initialize_agents("Test task", mock_model)

        # All agents should have their state set
        for agent in agents:
            assert agent.get_state() is not None
            assert agent.get_state().shape == (batch_size, hidden_dim)

    def test_initialize_agents_stores_kv_cache_in_hierarchical_cache(self):
        """_initialize_agents should store KV-Cache in the hierarchical cache."""
        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=2)
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
        ]
        protocol.register_agents(agents)

        # Mock model wrapper
        mock_model = MagicMock()
        mock_model.prepare_input = MagicMock(return_value=(
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[1, 1, 1]])
        ))

        batch_size, hidden_dim = 1, 64
        num_layers, num_heads, seq_len, head_dim = 2, 4, 3, 16
        mock_kv_cache = tuple(
            (
                torch.randn(batch_size, num_heads, seq_len, head_dim),
                torch.randn(batch_size, num_heads, seq_len, head_dim),
            )
            for _ in range(num_layers)
        )
        mock_hidden = torch.randn(batch_size, hidden_dim)
        mock_model.generate_latent = MagicMock(return_value=(mock_kv_cache, mock_hidden))

        protocol._initialize_agents("Test task", mock_model)

        # KV-Cache should be stored for each agent
        for agent in agents:
            stored_cache = kv_cache.get_local(agent.agent_id)
            assert stored_cache is not None

    def test_initialize_agents_calls_prepare_input_for_each_agent(self):
        """_initialize_agents should call prepare_input once per agent."""
        kv_cache = HierarchicalKVCache(num_groups=2, agents_per_group=2)
        attention = CrossLevelAttention(hidden_dim=64)

        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=3,
            latent_steps=5,
        )

        agents = [
            LCNAgent(agent_id=0, group_id=0, hidden_dim=64, persona="P0"),
            LCNAgent(agent_id=1, group_id=0, hidden_dim=64, persona="P1"),
            LCNAgent(agent_id=2, group_id=1, hidden_dim=64, persona="P2"),
            LCNAgent(agent_id=3, group_id=1, hidden_dim=64, persona="P3"),
        ]
        protocol.register_agents(agents)

        mock_model = MagicMock()
        mock_model.prepare_input = MagicMock(return_value=(
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[1, 1, 1]])
        ))

        batch_size, hidden_dim = 1, 64
        num_layers, num_heads, seq_len, head_dim = 2, 4, 3, 16
        mock_kv_cache = tuple(
            (
                torch.randn(batch_size, num_heads, seq_len, head_dim),
                torch.randn(batch_size, num_heads, seq_len, head_dim),
            )
            for _ in range(num_layers)
        )
        mock_hidden = torch.randn(batch_size, hidden_dim)
        mock_model.generate_latent = MagicMock(return_value=(mock_kv_cache, mock_hidden))

        protocol._initialize_agents("Test task", mock_model)

        assert mock_model.prepare_input.call_count == 4

    def test_initialize_agents_calls_generate_latent_for_each_agent(self):
        """_initialize_agents should call generate_latent once per agent."""
        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=3)
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
            LCNAgent(agent_id=2, group_id=0, hidden_dim=64),
        ]
        protocol.register_agents(agents)

        mock_model = MagicMock()
        mock_model.prepare_input = MagicMock(return_value=(
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[1, 1, 1]])
        ))

        batch_size, hidden_dim = 1, 64
        num_layers, num_heads, seq_len, head_dim = 2, 4, 3, 16
        mock_kv_cache = tuple(
            (
                torch.randn(batch_size, num_heads, seq_len, head_dim),
                torch.randn(batch_size, num_heads, seq_len, head_dim),
            )
            for _ in range(num_layers)
        )
        mock_hidden = torch.randn(batch_size, hidden_dim)
        mock_model.generate_latent = MagicMock(return_value=(mock_kv_cache, mock_hidden))

        protocol._initialize_agents("Test task", mock_model)

        assert mock_model.generate_latent.call_count == 3

    def test_initialize_agents_uses_latent_steps_from_protocol(self):
        """_initialize_agents should pass protocol's latent_steps to generate_latent."""
        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=1)
        attention = CrossLevelAttention(hidden_dim=64)

        latent_steps = 7  # Specific value to check

        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=3,
            latent_steps=latent_steps,
        )

        agents = [LCNAgent(agent_id=0, group_id=0, hidden_dim=64)]
        protocol.register_agents(agents)

        mock_model = MagicMock()
        mock_model.prepare_input = MagicMock(return_value=(
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[1, 1, 1]])
        ))

        batch_size, hidden_dim = 1, 64
        num_layers, num_heads, seq_len, head_dim = 2, 4, 3, 16
        mock_kv_cache = tuple(
            (
                torch.randn(batch_size, num_heads, seq_len, head_dim),
                torch.randn(batch_size, num_heads, seq_len, head_dim),
            )
            for _ in range(num_layers)
        )
        mock_hidden = torch.randn(batch_size, hidden_dim)
        mock_model.generate_latent = MagicMock(return_value=(mock_kv_cache, mock_hidden))

        protocol._initialize_agents("Test task", mock_model)

        # Check that generate_latent was called with the correct latent_steps
        call_kwargs = mock_model.generate_latent.call_args[1]
        assert call_kwargs["latent_steps"] == latent_steps

    def test_initialize_agents_with_multiple_groups(self):
        """_initialize_agents should correctly handle agents from multiple groups."""
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

        mock_model = MagicMock()
        mock_model.prepare_input = MagicMock(return_value=(
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[1, 1, 1]])
        ))

        batch_size, hidden_dim = 1, 64
        num_layers, num_heads, seq_len, head_dim = 2, 4, 3, 16
        mock_kv_cache = tuple(
            (
                torch.randn(batch_size, num_heads, seq_len, head_dim),
                torch.randn(batch_size, num_heads, seq_len, head_dim),
            )
            for _ in range(num_layers)
        )
        mock_hidden = torch.randn(batch_size, hidden_dim)
        mock_model.generate_latent = MagicMock(return_value=(mock_kv_cache, mock_hidden))

        protocol._initialize_agents("Test task", mock_model)

        # Check that KV-Cache has correct group mapping
        assert kv_cache.get_agent_group(0) == 0
        assert kv_cache.get_agent_group(1) == 0
        assert kv_cache.get_agent_group(2) == 1
        assert kv_cache.get_agent_group(3) == 1

    def test_initialize_agents_with_no_agents_does_nothing(self):
        """_initialize_agents should handle empty agent list gracefully."""
        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=1)
        attention = CrossLevelAttention(hidden_dim=64)

        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=3,
            latent_steps=5,
        )

        # No agents registered
        mock_model = MagicMock()

        # Should not raise, just do nothing
        protocol._initialize_agents("Test task", mock_model)

        mock_model.prepare_input.assert_not_called()
        mock_model.generate_latent.assert_not_called()
