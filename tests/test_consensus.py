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


class TestKVToRepr:
    """Tests for ConsensusProtocol._kv_to_repr method."""

    def test_kv_to_repr_returns_tensor_with_correct_shape(self):
        """_kv_to_repr should return tensor with shape [B, L, D]."""
        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=1)
        hidden_dim = 64
        attention = CrossLevelAttention(hidden_dim=hidden_dim)

        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=3,
            latent_steps=5,
        )

        batch_size, num_heads, seq_len, head_dim = 1, 4, 10, 16
        # hidden_dim = num_heads * head_dim = 64
        mock_kv_cache = tuple(
            (
                torch.randn(batch_size, num_heads, seq_len, head_dim),
                torch.randn(batch_size, num_heads, seq_len, head_dim),
            )
            for _ in range(2)  # 2 layers
        )

        repr_tensor = protocol._kv_to_repr(mock_kv_cache)

        assert repr_tensor.shape == (batch_size, seq_len, hidden_dim)

    def test_kv_to_repr_returns_none_for_none_input(self):
        """_kv_to_repr should return None when given None."""
        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=1)
        attention = CrossLevelAttention(hidden_dim=64)

        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=3,
            latent_steps=5,
        )

        assert protocol._kv_to_repr(None) is None

    def test_kv_to_repr_uses_last_layer_values(self):
        """_kv_to_repr should extract representation from the last layer's values."""
        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=1)
        hidden_dim = 64
        attention = CrossLevelAttention(hidden_dim=hidden_dim)

        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=3,
            latent_steps=5,
        )

        batch_size, num_heads, seq_len, head_dim = 1, 4, 5, 16
        # Create a KV-Cache with distinct values in each layer
        layer_0 = (
            torch.zeros(batch_size, num_heads, seq_len, head_dim),
            torch.zeros(batch_size, num_heads, seq_len, head_dim),
        )
        layer_1 = (
            torch.ones(batch_size, num_heads, seq_len, head_dim),
            torch.ones(batch_size, num_heads, seq_len, head_dim) * 2.0,  # Last layer values
        )
        mock_kv_cache = (layer_0, layer_1)

        repr_tensor = protocol._kv_to_repr(mock_kv_cache)

        # The representation should come from the last layer's values (all 2.0s)
        assert torch.allclose(repr_tensor, torch.ones(batch_size, seq_len, hidden_dim) * 2.0)

    def test_kv_to_repr_handles_single_layer(self):
        """_kv_to_repr should work with single-layer KV-Cache."""
        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=1)
        hidden_dim = 64
        attention = CrossLevelAttention(hidden_dim=hidden_dim)

        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=3,
            latent_steps=5,
        )

        batch_size, num_heads, seq_len, head_dim = 1, 4, 3, 16
        mock_kv_cache = (
            (
                torch.randn(batch_size, num_heads, seq_len, head_dim),
                torch.randn(batch_size, num_heads, seq_len, head_dim),
            ),
        )

        repr_tensor = protocol._kv_to_repr(mock_kv_cache)

        assert repr_tensor.shape == (batch_size, seq_len, hidden_dim)


class TestRunMethod:
    """Tests for ConsensusProtocol.run method."""

    def _create_mock_model(self, batch_size=1, hidden_dim=64, num_layers=2, num_heads=4, seq_len=3, head_dim=16):
        """Helper to create a mock model for testing."""
        mock_model = MagicMock()
        mock_model.prepare_input = MagicMock(return_value=(
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[1, 1, 1]])
        ))

        mock_kv_cache = tuple(
            (
                torch.randn(batch_size, num_heads, seq_len, head_dim),
                torch.randn(batch_size, num_heads, seq_len, head_dim),
            )
            for _ in range(num_layers)
        )
        mock_hidden = torch.randn(batch_size, hidden_dim)
        mock_model.generate_latent = MagicMock(return_value=(mock_kv_cache, mock_hidden))

        return mock_model

    def test_run_returns_consensus_result(self):
        """run() should return a ConsensusResult instance."""
        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=2)
        hidden_dim = 64
        attention = CrossLevelAttention(hidden_dim=hidden_dim)

        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=2,
            latent_steps=5,
        )

        agents = [
            LCNAgent(agent_id=0, group_id=0, hidden_dim=hidden_dim),
            LCNAgent(agent_id=1, group_id=0, hidden_dim=hidden_dim),
        ]
        protocol.register_agents(agents)

        mock_model = self._create_mock_model(hidden_dim=hidden_dim)

        result = protocol.run("Test task", mock_model)

        assert isinstance(result, ConsensusResult)

    def test_run_calls_initialize_agents(self):
        """run() should call _initialize_agents with task and model."""
        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=2)
        hidden_dim = 64
        attention = CrossLevelAttention(hidden_dim=hidden_dim)

        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=2,
            latent_steps=5,
        )

        agents = [
            LCNAgent(agent_id=0, group_id=0, hidden_dim=hidden_dim),
            LCNAgent(agent_id=1, group_id=0, hidden_dim=hidden_dim),
        ]
        protocol.register_agents(agents)

        mock_model = self._create_mock_model(hidden_dim=hidden_dim)
        mock_model.generate_text = MagicMock(return_value=["Response"])

        protocol.run("Test task", mock_model)

        # _initialize_agents calls prepare_input and generate_latent for each agent
        # _make_decision also calls prepare_input for each agent (total: 2 init + 2 decision = 4)
        assert mock_model.prepare_input.call_count == 4
        assert mock_model.generate_latent.call_count == 2

    def test_run_aggregates_caches_each_round(self):
        """run() should aggregate group and global caches each round."""
        kv_cache = HierarchicalKVCache(num_groups=2, agents_per_group=2)
        hidden_dim = 64
        attention = CrossLevelAttention(hidden_dim=hidden_dim)

        num_rounds = 3
        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=num_rounds,
            latent_steps=5,
        )

        agents = [
            LCNAgent(agent_id=0, group_id=0, hidden_dim=hidden_dim),
            LCNAgent(agent_id=1, group_id=0, hidden_dim=hidden_dim),
            LCNAgent(agent_id=2, group_id=1, hidden_dim=hidden_dim),
            LCNAgent(agent_id=3, group_id=1, hidden_dim=hidden_dim),
        ]
        protocol.register_agents(agents)

        mock_model = self._create_mock_model(hidden_dim=hidden_dim)

        protocol.run("Test task", mock_model)

        # After run, group and global caches should be populated
        assert 0 in kv_cache.group_caches
        assert 1 in kv_cache.group_caches
        assert kv_cache.global_cache is not None

    def test_run_updates_agent_states_each_round(self):
        """run() should update agent states in each round."""
        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=2)
        hidden_dim = 64
        attention = CrossLevelAttention(hidden_dim=hidden_dim)

        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=2,
            latent_steps=5,
        )

        agents = [
            LCNAgent(agent_id=0, group_id=0, hidden_dim=hidden_dim),
            LCNAgent(agent_id=1, group_id=0, hidden_dim=hidden_dim),
        ]
        protocol.register_agents(agents)

        mock_model = self._create_mock_model(hidden_dim=hidden_dim)

        # Run consensus
        protocol.run("Test task", mock_model)

        # All agents should have states
        for agent in agents:
            assert agent.get_state() is not None
            assert agent.get_state().shape[-1] == hidden_dim

    def test_run_tracks_attention_history(self):
        """run() should track attention weights in attention_history."""
        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=2)
        hidden_dim = 64
        attention = CrossLevelAttention(hidden_dim=hidden_dim)

        num_rounds = 3
        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=num_rounds,
            latent_steps=5,
        )

        agents = [
            LCNAgent(agent_id=0, group_id=0, hidden_dim=hidden_dim),
            LCNAgent(agent_id=1, group_id=0, hidden_dim=hidden_dim),
        ]
        protocol.register_agents(agents)

        mock_model = self._create_mock_model(hidden_dim=hidden_dim)

        result = protocol.run("Test task", mock_model)

        # attention_history should have entries
        assert isinstance(result.attention_history, list)
        # Should have entries for each round
        assert len(result.attention_history) == num_rounds

    def test_run_returns_real_decision(self):
        """run() should return a real decision from agents (not a placeholder)."""
        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=2)
        hidden_dim = 64
        attention = CrossLevelAttention(hidden_dim=hidden_dim)

        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=2,
            latent_steps=5,
        )

        agents = [
            LCNAgent(agent_id=0, group_id=0, hidden_dim=hidden_dim),
            LCNAgent(agent_id=1, group_id=0, hidden_dim=hidden_dim),
        ]
        protocol.register_agents(agents)

        mock_model = self._create_mock_model(hidden_dim=hidden_dim)
        mock_model.generate_text = MagicMock(return_value=["Decision A"])

        result = protocol.run("Test task", mock_model)

        # Should be a real decision, not a placeholder
        assert result.decision is not None
        assert isinstance(result.decision, str)
        assert result.decision == "Decision A"

    def test_run_with_multiple_groups(self):
        """run() should handle multiple groups correctly."""
        kv_cache = HierarchicalKVCache(num_groups=2, agents_per_group=2)
        hidden_dim = 64
        attention = CrossLevelAttention(hidden_dim=hidden_dim)

        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=2,
            latent_steps=5,
        )

        agents = [
            LCNAgent(agent_id=0, group_id=0, hidden_dim=hidden_dim),
            LCNAgent(agent_id=1, group_id=0, hidden_dim=hidden_dim),
            LCNAgent(agent_id=2, group_id=1, hidden_dim=hidden_dim),
            LCNAgent(agent_id=3, group_id=1, hidden_dim=hidden_dim),
        ]
        protocol.register_agents(agents)

        mock_model = self._create_mock_model(hidden_dim=hidden_dim)

        result = protocol.run("Test task", mock_model)

        assert isinstance(result, ConsensusResult)
        # All 4 agents should be processed
        for agent in agents:
            assert agent.get_state() is not None

    def test_run_with_zero_rounds(self):
        """run() with zero rounds should still initialize agents and return result."""
        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=2)
        hidden_dim = 64
        attention = CrossLevelAttention(hidden_dim=hidden_dim)

        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=0,  # Zero rounds
            latent_steps=5,
        )

        agents = [
            LCNAgent(agent_id=0, group_id=0, hidden_dim=hidden_dim),
            LCNAgent(agent_id=1, group_id=0, hidden_dim=hidden_dim),
        ]
        protocol.register_agents(agents)

        mock_model = self._create_mock_model(hidden_dim=hidden_dim)

        result = protocol.run("Test task", mock_model)

        assert isinstance(result, ConsensusResult)
        assert result.attention_history == []

    def test_run_uses_cross_level_attention(self):
        """run() should use cross-level attention to fuse information."""
        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=2)
        hidden_dim = 64

        # Create a real attention module
        attention = CrossLevelAttention(hidden_dim=hidden_dim)

        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=1,
            latent_steps=5,
        )

        agents = [
            LCNAgent(agent_id=0, group_id=0, hidden_dim=hidden_dim),
            LCNAgent(agent_id=1, group_id=0, hidden_dim=hidden_dim),
        ]
        protocol.register_agents(agents)

        mock_model = self._create_mock_model(hidden_dim=hidden_dim)

        # Run should complete without error
        result = protocol.run("Test task", mock_model)

        assert isinstance(result, ConsensusResult)

    def test_run_hidden_states_change_across_rounds(self):
        """run() should evolve hidden states across consensus rounds via attention fusion."""
        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=2)
        hidden_dim = 64

        # Create a real attention module
        attention = CrossLevelAttention(hidden_dim=hidden_dim)

        num_rounds = 3
        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=num_rounds,
            latent_steps=5,
        )

        agents = [
            LCNAgent(agent_id=0, group_id=0, hidden_dim=hidden_dim),
            LCNAgent(agent_id=1, group_id=0, hidden_dim=hidden_dim),
        ]
        protocol.register_agents(agents)

        mock_model = self._create_mock_model(hidden_dim=hidden_dim)

        # Initialize agents manually to capture initial states
        protocol._initialize_agents("Test task", mock_model)

        # Capture states after initialization (before any consensus rounds)
        initial_state_0 = agents[0].get_state().clone()
        initial_state_1 = agents[1].get_state().clone()

        # Manually run one consensus round to see state changes
        # Step 2a: Aggregate caches
        for group_id in protocol.group_ids:
            kv_cache.aggregate_group(group_id)
        kv_cache.aggregate_global()

        # Step 2b-c: For agent 0, fuse and update
        local_caches, group_cache, global_cache = kv_cache.get_all_levels(0)
        local_repr = protocol._kv_list_to_repr(local_caches)
        group_repr = protocol._kv_to_repr(group_cache)
        global_repr = protocol._kv_to_repr(global_cache)

        query_state = agents[0].get_state()
        fused_state, _ = attention.forward(
            query_state=query_state,
            local_repr=local_repr,
            group_repr=group_repr,
            global_repr=global_repr,
        )
        agents[0].set_state(fused_state)

        # State after one round of fusion
        state_after_round_0 = agents[0].get_state()

        # Verify that the state actually changed
        # The attention fusion should produce a different state because:
        # 1. We have group and global representations available
        # 2. The fused state is a weighted combination of query and context
        assert state_after_round_0 is not None
        assert initial_state_0 is not None
        assert state_after_round_0.shape == initial_state_0.shape

        # The states should be different after attention fusion
        # (unless the attention weights are exactly [1, 0, 0, ...] which is unlikely)
        state_diff = (state_after_round_0 - initial_state_0).abs().sum().item()
        assert state_diff > 0, "Hidden state should change after attention fusion"


class TestMakeDecision:
    """Tests for ConsensusProtocol._make_decision method."""

    def _create_mock_model(self, batch_size=1, hidden_dim=64, num_layers=2, num_heads=4, seq_len=3, head_dim=16):
        """Helper to create a mock model for testing."""
        mock_model = MagicMock()
        mock_model.prepare_input = MagicMock(return_value=(
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[1, 1, 1]])
        ))

        mock_kv_cache = tuple(
            (
                torch.randn(batch_size, num_heads, seq_len, head_dim),
                torch.randn(batch_size, num_heads, seq_len, head_dim),
            )
            for _ in range(num_layers)
        )
        mock_hidden = torch.randn(batch_size, hidden_dim)
        mock_model.generate_latent = MagicMock(return_value=(mock_kv_cache, mock_hidden))
        mock_model.generate_text = MagicMock(return_value=["Test response"])

        return mock_model

    def test_make_decision_returns_consensus_result(self):
        """_make_decision should return a ConsensusResult instance."""
        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=2)
        hidden_dim = 64
        attention = CrossLevelAttention(hidden_dim=hidden_dim)

        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=2,
            latent_steps=5,
        )

        agents = [
            LCNAgent(agent_id=0, group_id=0, hidden_dim=hidden_dim),
            LCNAgent(agent_id=1, group_id=0, hidden_dim=hidden_dim),
        ]
        protocol.register_agents(agents)

        mock_model = self._create_mock_model(hidden_dim=hidden_dim)
        mock_model.generate_text = MagicMock(return_value=["Response A"])

        # Initialize agents first so they have KV-Caches
        protocol._initialize_agents("Test task", mock_model)

        attention_history = [{"round": 0, "agent_weights": {}}]
        result = protocol._make_decision("Test task", mock_model, attention_history)

        assert isinstance(result, ConsensusResult)

    def test_make_decision_calls_generate_text_for_each_agent(self):
        """_make_decision should call generate_text once per agent."""
        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=3)
        hidden_dim = 64
        attention = CrossLevelAttention(hidden_dim=hidden_dim)

        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=2,
            latent_steps=5,
        )

        agents = [
            LCNAgent(agent_id=0, group_id=0, hidden_dim=hidden_dim),
            LCNAgent(agent_id=1, group_id=0, hidden_dim=hidden_dim),
            LCNAgent(agent_id=2, group_id=0, hidden_dim=hidden_dim),
        ]
        protocol.register_agents(agents)

        mock_model = self._create_mock_model(hidden_dim=hidden_dim)

        # Initialize agents first
        protocol._initialize_agents("Test task", mock_model)

        # Reset call count for generate_text
        mock_model.generate_text.reset_mock()
        mock_model.generate_text.return_value = ["Response"]

        attention_history = []
        protocol._make_decision("Test task", mock_model, attention_history)

        # Should call generate_text once per agent
        assert mock_model.generate_text.call_count == 3

    def test_make_decision_populates_agent_decisions(self):
        """_make_decision should populate agent_decisions dict with each agent's response."""
        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=2)
        hidden_dim = 64
        attention = CrossLevelAttention(hidden_dim=hidden_dim)

        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=2,
            latent_steps=5,
        )

        agents = [
            LCNAgent(agent_id=0, group_id=0, hidden_dim=hidden_dim),
            LCNAgent(agent_id=1, group_id=0, hidden_dim=hidden_dim),
        ]
        protocol.register_agents(agents)

        mock_model = self._create_mock_model(hidden_dim=hidden_dim)

        # Initialize agents
        protocol._initialize_agents("Test task", mock_model)

        # Make generate_text return different values for each call
        mock_model.generate_text.side_effect = [["Response A"], ["Response B"]]

        attention_history = []
        result = protocol._make_decision("Test task", mock_model, attention_history)

        # agent_decisions should map agent_id to their response
        assert 0 in result.agent_decisions
        assert 1 in result.agent_decisions
        assert result.agent_decisions[0] == "Response A"
        assert result.agent_decisions[1] == "Response B"

    def test_make_decision_majority_vote(self):
        """_make_decision should use majority vote to determine group decision."""
        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=3)
        hidden_dim = 64
        attention = CrossLevelAttention(hidden_dim=hidden_dim)

        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=2,
            latent_steps=5,
        )

        agents = [
            LCNAgent(agent_id=0, group_id=0, hidden_dim=hidden_dim),
            LCNAgent(agent_id=1, group_id=0, hidden_dim=hidden_dim),
            LCNAgent(agent_id=2, group_id=0, hidden_dim=hidden_dim),
        ]
        protocol.register_agents(agents)

        mock_model = self._create_mock_model(hidden_dim=hidden_dim)

        # Initialize agents
        protocol._initialize_agents("Test task", mock_model)

        # Majority is "Yes" (2 out of 3)
        mock_model.generate_text.side_effect = [["Yes"], ["No"], ["Yes"]]

        attention_history = []
        result = protocol._make_decision("Test task", mock_model, attention_history)

        assert result.decision == "Yes"

    def test_make_decision_sets_convergence_round(self):
        """_make_decision should set convergence_round to num_rounds."""
        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=2)
        hidden_dim = 64
        attention = CrossLevelAttention(hidden_dim=hidden_dim)

        num_rounds = 5
        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=num_rounds,
            latent_steps=5,
        )

        agents = [
            LCNAgent(agent_id=0, group_id=0, hidden_dim=hidden_dim),
            LCNAgent(agent_id=1, group_id=0, hidden_dim=hidden_dim),
        ]
        protocol.register_agents(agents)

        mock_model = self._create_mock_model(hidden_dim=hidden_dim)

        # Initialize agents
        protocol._initialize_agents("Test task", mock_model)

        mock_model.generate_text.return_value = ["Response"]

        attention_history = []
        result = protocol._make_decision("Test task", mock_model, attention_history)

        assert result.convergence_round == num_rounds

    def test_make_decision_preserves_attention_history(self):
        """_make_decision should include the passed attention_history in result."""
        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=2)
        hidden_dim = 64
        attention = CrossLevelAttention(hidden_dim=hidden_dim)

        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=2,
            latent_steps=5,
        )

        agents = [
            LCNAgent(agent_id=0, group_id=0, hidden_dim=hidden_dim),
            LCNAgent(agent_id=1, group_id=0, hidden_dim=hidden_dim),
        ]
        protocol.register_agents(agents)

        mock_model = self._create_mock_model(hidden_dim=hidden_dim)

        # Initialize agents
        protocol._initialize_agents("Test task", mock_model)

        mock_model.generate_text.return_value = ["Response"]

        attention_history = [
            {"round": 0, "agent_weights": {0: [0.5, 0.5]}},
            {"round": 1, "agent_weights": {0: [0.6, 0.4]}},
        ]
        result = protocol._make_decision("Test task", mock_model, attention_history)

        assert result.attention_history == attention_history

    def test_make_decision_uses_kv_cache_for_generation(self):
        """_make_decision should pass KV-Cache to generate_text for context."""
        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=1)
        hidden_dim = 64
        attention = CrossLevelAttention(hidden_dim=hidden_dim)

        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=2,
            latent_steps=5,
        )

        agents = [LCNAgent(agent_id=0, group_id=0, hidden_dim=hidden_dim)]
        protocol.register_agents(agents)

        mock_model = self._create_mock_model(hidden_dim=hidden_dim)

        # Initialize agents
        protocol._initialize_agents("Test task", mock_model)

        mock_model.generate_text.return_value = ["Response"]

        attention_history = []
        protocol._make_decision("Test task", mock_model, attention_history)

        # Verify generate_text was called with past_key_values
        call_kwargs = mock_model.generate_text.call_args[1]
        assert "past_key_values" in call_kwargs
        assert call_kwargs["past_key_values"] is not None

    def test_make_decision_with_tie_picks_first(self):
        """_make_decision should pick the first most common response on tie."""
        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=4)
        hidden_dim = 64
        attention = CrossLevelAttention(hidden_dim=hidden_dim)

        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=2,
            latent_steps=5,
        )

        agents = [
            LCNAgent(agent_id=0, group_id=0, hidden_dim=hidden_dim),
            LCNAgent(agent_id=1, group_id=0, hidden_dim=hidden_dim),
            LCNAgent(agent_id=2, group_id=0, hidden_dim=hidden_dim),
            LCNAgent(agent_id=3, group_id=0, hidden_dim=hidden_dim),
        ]
        protocol.register_agents(agents)

        mock_model = self._create_mock_model(hidden_dim=hidden_dim)

        # Initialize agents
        protocol._initialize_agents("Test task", mock_model)

        # Tie: 2 "A" and 2 "B"
        mock_model.generate_text.side_effect = [["A"], ["B"], ["A"], ["B"]]

        attention_history = []
        result = protocol._make_decision("Test task", mock_model, attention_history)

        # Should pick the first most common - "A" appears first
        assert result.decision == "A"


class TestExtractDecision:
    """Tests for ConsensusProtocol._extract_decision method."""

    def test_extract_decision_returns_response_as_is(self):
        """_extract_decision should return the response unchanged (placeholder behavior)."""
        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=1)
        attention = CrossLevelAttention(hidden_dim=64)

        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=2,
            latent_steps=5,
        )

        response = "After considering all factors, I believe Candidate C is the best choice."
        decision = protocol._extract_decision(response)

        # Placeholder implementation returns the full response
        assert decision == response

    def test_extract_decision_handles_empty_string(self):
        """_extract_decision should handle empty string input."""
        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=1)
        attention = CrossLevelAttention(hidden_dim=64)

        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=2,
            latent_steps=5,
        )

        decision = protocol._extract_decision("")
        assert decision == ""

    def test_extract_decision_preserves_whitespace(self):
        """_extract_decision should preserve whitespace in responses."""
        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=1)
        attention = CrossLevelAttention(hidden_dim=64)

        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=2,
            latent_steps=5,
        )

        response = "  Yes  "
        decision = protocol._extract_decision(response)
        assert decision == response


class TestRunWithDecisionMaking:
    """Tests for run() method integration with _make_decision."""

    def _create_mock_model(self, batch_size=1, hidden_dim=64, num_layers=2, num_heads=4, seq_len=3, head_dim=16):
        """Helper to create a mock model for testing."""
        mock_model = MagicMock()
        mock_model.prepare_input = MagicMock(return_value=(
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[1, 1, 1]])
        ))

        mock_kv_cache = tuple(
            (
                torch.randn(batch_size, num_heads, seq_len, head_dim),
                torch.randn(batch_size, num_heads, seq_len, head_dim),
            )
            for _ in range(num_layers)
        )
        mock_hidden = torch.randn(batch_size, hidden_dim)
        mock_model.generate_latent = MagicMock(return_value=(mock_kv_cache, mock_hidden))
        mock_model.generate_text = MagicMock(return_value=["Test response"])

        return mock_model

    def test_run_calls_make_decision(self):
        """run() should call _make_decision and return its result."""
        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=2)
        hidden_dim = 64
        attention = CrossLevelAttention(hidden_dim=hidden_dim)

        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=2,
            latent_steps=5,
        )

        agents = [
            LCNAgent(agent_id=0, group_id=0, hidden_dim=hidden_dim),
            LCNAgent(agent_id=1, group_id=0, hidden_dim=hidden_dim),
        ]
        protocol.register_agents(agents)

        mock_model = self._create_mock_model(hidden_dim=hidden_dim)
        mock_model.generate_text.side_effect = [["Decision A"], ["Decision B"]]

        result = protocol.run("Test task", mock_model)

        # Should no longer be a placeholder
        assert result.decision != "[placeholder - decision making not yet implemented]"
        assert result.decision in ["Decision A", "Decision B"]

    def test_run_returns_real_agent_decisions(self):
        """run() should return real agent decisions, not empty strings."""
        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=2)
        hidden_dim = 64
        attention = CrossLevelAttention(hidden_dim=hidden_dim)

        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=2,
            latent_steps=5,
        )

        agents = [
            LCNAgent(agent_id=0, group_id=0, hidden_dim=hidden_dim),
            LCNAgent(agent_id=1, group_id=0, hidden_dim=hidden_dim),
        ]
        protocol.register_agents(agents)

        mock_model = self._create_mock_model(hidden_dim=hidden_dim)
        mock_model.generate_text.side_effect = [["Response 1"], ["Response 2"]]

        result = protocol.run("Test task", mock_model)

        # Agent decisions should contain actual responses
        assert result.agent_decisions[0] == "Response 1"
        assert result.agent_decisions[1] == "Response 2"

    def test_run_sets_convergence_round_to_num_rounds(self):
        """run() should set convergence_round to num_rounds."""
        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=2)
        hidden_dim = 64
        attention = CrossLevelAttention(hidden_dim=hidden_dim)

        num_rounds = 4
        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=num_rounds,
            latent_steps=5,
        )

        agents = [
            LCNAgent(agent_id=0, group_id=0, hidden_dim=hidden_dim),
            LCNAgent(agent_id=1, group_id=0, hidden_dim=hidden_dim),
        ]
        protocol.register_agents(agents)

        mock_model = self._create_mock_model(hidden_dim=hidden_dim)
        mock_model.generate_text.return_value = ["Response"]

        result = protocol.run("Test task", mock_model)

        assert result.convergence_round == num_rounds
