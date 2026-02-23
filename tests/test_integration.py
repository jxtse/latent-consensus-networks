# tests/test_integration.py
"""Integration tests for LCN framework - end-to-end tests using mock model."""

import pytest
import torch
from unittest.mock import MagicMock, patch
from dataclasses import is_dataclass, fields

from lcn.core.agent import LCNAgent
from lcn.core.kv_cache import HierarchicalKVCache
from lcn.core.attention import CrossLevelAttention
from lcn.core.consensus import ConsensusProtocol
from lcn.core.results import ConsensusResult
from lcn.environments.hidden_profile import (
    HiddenProfileEnvironment,
    HiddenProfileMetrics,
    HiddenProfileScenario,
)


class MockLCNModelWrapper:
    """
    Mock LCNModelWrapper for integration testing without loading a real model.

    Provides predictable mock responses for all model operations.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        seq_len: int = 10,
        correct_answer: str = "C",
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = hidden_dim // num_heads
        self.correct_answer = correct_answer

        # Track call counts for verification
        self.prepare_input_calls = 0
        self.generate_latent_calls = 0
        self.generate_text_calls = 0

    def prepare_input(self, messages, add_generation_prompt=True):
        """Return mock input tensors."""
        self.prepare_input_calls += 1
        # Return mock input_ids and attention_mask
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.ones_like(input_ids)
        return input_ids, attention_mask

    def generate_latent(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        latent_steps: int,
        past_key_values=None,
    ):
        """Return mock KV-Cache and hidden state."""
        self.generate_latent_calls += 1
        batch_size = input_ids.shape[0]

        # Create mock KV-Cache
        kv_cache = tuple(
            (
                torch.randn(batch_size, self.num_heads, self.seq_len, self.head_dim),
                torch.randn(batch_size, self.num_heads, self.seq_len, self.head_dim),
            )
            for _ in range(self.num_layers)
        )

        # Create mock hidden state
        hidden_state = torch.randn(batch_size, self.hidden_dim)

        return kv_cache, hidden_state

    def generate_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values=None,
        max_new_tokens: int = 256,
    ):
        """Return mock text response with the correct answer."""
        self.generate_text_calls += 1
        # Return the correct answer to ensure integration test passes
        return [self.correct_answer]


class TestEndToEndWithMockModel:
    """End-to-end integration tests using mock model."""

    @pytest.fixture
    def hidden_dim(self):
        return 64

    @pytest.fixture
    def num_agents(self):
        return 6

    @pytest.fixture
    def num_groups(self):
        return 2

    @pytest.fixture
    def environment(self, num_agents, num_groups, hidden_dim):
        """Create HiddenProfileEnvironment with default settings."""
        return HiddenProfileEnvironment(
            num_agents=num_agents,
            num_groups=num_groups,
            hidden_dim=hidden_dim,
        )

    @pytest.fixture
    def mock_model(self, hidden_dim):
        """Create mock model wrapper."""
        return MockLCNModelWrapper(hidden_dim=hidden_dim)

    @pytest.fixture
    def kv_cache(self, num_groups, num_agents):
        """Create hierarchical KV-Cache."""
        agents_per_group = num_agents // num_groups
        return HierarchicalKVCache(num_groups=num_groups, agents_per_group=agents_per_group)

    @pytest.fixture
    def attention(self, hidden_dim):
        """Create cross-level attention module."""
        return CrossLevelAttention(hidden_dim=hidden_dim)

    @pytest.fixture
    def protocol(self, kv_cache, attention):
        """Create consensus protocol."""
        return ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=3,
            latent_steps=5,
        )

    def test_full_integration_flow(self, environment, protocol, mock_model):
        """
        Test the complete integration flow:
        1. Create environment and agents
        2. Register agents with protocol
        3. Run consensus on a scenario
        4. Verify ConsensusResult
        5. Evaluate with HiddenProfileMetrics
        """
        # Step 1: Create agents
        agents = environment.create_agents()
        assert len(agents) == 6

        # Step 2: Register agents with protocol
        protocol.register_agents(agents)
        assert len(protocol.agents) == 6

        # Step 3: Run consensus on a scenario
        scenario = environment.scenarios[0]
        task = environment.get_agent_prompt(agents[0], scenario)

        result = protocol.run(task, mock_model)

        # Step 4: Verify ConsensusResult structure
        assert isinstance(result, ConsensusResult)
        assert isinstance(result.decision, str)
        assert len(result.decision) > 0
        assert isinstance(result.agent_decisions, dict)
        assert len(result.agent_decisions) == 6
        assert isinstance(result.attention_history, list)
        assert isinstance(result.convergence_round, int)

        # Step 5: Evaluate with HiddenProfileMetrics
        metrics = environment.evaluate(result, scenario)
        assert isinstance(metrics, HiddenProfileMetrics)

    def test_consensus_result_structure(self, environment, protocol, mock_model):
        """Verify ConsensusResult has all required fields with correct types."""
        agents = environment.create_agents()
        protocol.register_agents(agents)
        scenario = environment.scenarios[0]
        task = environment.get_agent_prompt(agents[0], scenario)

        result = protocol.run(task, mock_model)

        # Verify it's a dataclass
        assert is_dataclass(result)

        # Verify all fields are present
        field_names = {f.name for f in fields(ConsensusResult)}
        assert field_names == {"decision", "agent_decisions", "attention_history", "convergence_round"}

        # Verify decision is a string
        assert isinstance(result.decision, str)

        # Verify agent_decisions maps agent_id (int) to decision (str)
        for agent_id, decision in result.agent_decisions.items():
            assert isinstance(agent_id, int)
            assert isinstance(decision, str)

        # Verify attention_history is a list of dicts
        for entry in result.attention_history:
            assert isinstance(entry, dict)
            assert "round" in entry
            assert "agent_weights" in entry

        # Verify convergence_round
        assert result.convergence_round is None or isinstance(result.convergence_round, int)

    def test_hidden_profile_metrics_structure(self, environment, protocol, mock_model):
        """Verify HiddenProfileMetrics has all required fields."""
        agents = environment.create_agents()
        protocol.register_agents(agents)
        scenario = environment.scenarios[0]
        task = environment.get_agent_prompt(agents[0], scenario)

        result = protocol.run(task, mock_model)
        metrics = environment.evaluate(result, scenario)

        # Verify it's a dataclass
        assert is_dataclass(metrics)

        # Verify all fields are present
        field_names = {f.name for f in fields(HiddenProfileMetrics)}
        assert field_names == {"accuracy", "information_integration", "individual_accuracy", "decision_distribution"}

        # Verify accuracy is a float between 0 and 1
        assert isinstance(metrics.accuracy, float)
        assert 0.0 <= metrics.accuracy <= 1.0

        # Verify information_integration is a float
        assert isinstance(metrics.information_integration, float)

        # Verify individual_accuracy maps agent_id to float
        for agent_id, acc in metrics.individual_accuracy.items():
            assert isinstance(agent_id, int)
            assert isinstance(acc, float)
            assert 0.0 <= acc <= 1.0

        # Verify decision_distribution maps option to count
        for option, count in metrics.decision_distribution.items():
            assert isinstance(option, str)
            assert isinstance(count, int)
            assert count >= 0

    def test_all_agents_participate(self, environment, protocol, mock_model):
        """Verify all registered agents participate in consensus."""
        agents = environment.create_agents()
        protocol.register_agents(agents)
        scenario = environment.scenarios[0]
        task = environment.get_agent_prompt(agents[0], scenario)

        result = protocol.run(task, mock_model)

        # All agent IDs should be in agent_decisions
        expected_agent_ids = {agent.agent_id for agent in agents}
        actual_agent_ids = set(result.agent_decisions.keys())
        assert expected_agent_ids == actual_agent_ids

    def test_attention_history_recorded(self, environment, protocol, mock_model, num_agents):
        """Verify attention history is recorded for all rounds and agents."""
        agents = environment.create_agents()
        protocol.register_agents(agents)
        scenario = environment.scenarios[0]
        task = environment.get_agent_prompt(agents[0], scenario)

        result = protocol.run(task, mock_model)

        # Should have one entry per round
        assert len(result.attention_history) == protocol.num_rounds

        # Each entry should have agent weights
        for round_entry in result.attention_history:
            assert "round" in round_entry
            assert "agent_weights" in round_entry
            # Agent weights should have entries for each agent
            assert len(round_entry["agent_weights"]) == num_agents

    def test_correct_answer_yields_high_accuracy(self, hidden_dim):
        """When all agents answer correctly, accuracy should be 1.0."""
        env = HiddenProfileEnvironment(num_agents=4, num_groups=2, hidden_dim=hidden_dim)
        agents = env.create_agents()

        kv_cache = HierarchicalKVCache(num_groups=2, agents_per_group=2)
        attention = CrossLevelAttention(hidden_dim=hidden_dim)
        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=2,
            latent_steps=3,
        )
        protocol.register_agents(agents)

        # Mock model that always returns the correct answer for scenario 0
        mock_model = MockLCNModelWrapper(hidden_dim=hidden_dim, correct_answer="C")

        scenario = env.scenarios[0]  # correct_answer is "C"
        task = env.get_agent_prompt(agents[0], scenario)

        result = protocol.run(task, mock_model)
        metrics = env.evaluate(result, scenario)

        # With all agents answering "C", accuracy should be 1.0
        assert metrics.accuracy == 1.0
        assert all(acc == 1.0 for acc in metrics.individual_accuracy.values())

    def test_incorrect_answer_yields_low_accuracy(self, hidden_dim):
        """When all agents answer incorrectly, accuracy should be 0.0."""
        env = HiddenProfileEnvironment(num_agents=4, num_groups=2, hidden_dim=hidden_dim)
        agents = env.create_agents()

        kv_cache = HierarchicalKVCache(num_groups=2, agents_per_group=2)
        attention = CrossLevelAttention(hidden_dim=hidden_dim)
        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=2,
            latent_steps=3,
        )
        protocol.register_agents(agents)

        # Mock model that returns wrong answer ("A" instead of "C")
        mock_model = MockLCNModelWrapper(hidden_dim=hidden_dim, correct_answer="A")

        scenario = env.scenarios[0]  # correct_answer is "C"
        task = env.get_agent_prompt(agents[0], scenario)

        result = protocol.run(task, mock_model)
        metrics = env.evaluate(result, scenario)

        # With all agents answering "A", accuracy should be 0.0
        assert metrics.accuracy == 0.0
        assert all(acc == 0.0 for acc in metrics.individual_accuracy.values())


class TestModelWrapperCalls:
    """Tests to verify mock model is called correctly."""

    def test_prepare_input_called_for_initialization_and_decision(self):
        """prepare_input should be called for each agent during init and decision."""
        hidden_dim = 64
        num_agents = 4

        env = HiddenProfileEnvironment(num_agents=num_agents, num_groups=2, hidden_dim=hidden_dim)
        agents = env.create_agents()

        kv_cache = HierarchicalKVCache(num_groups=2, agents_per_group=2)
        attention = CrossLevelAttention(hidden_dim=hidden_dim)
        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=2,
            latent_steps=3,
        )
        protocol.register_agents(agents)

        mock_model = MockLCNModelWrapper(hidden_dim=hidden_dim)
        scenario = env.scenarios[0]
        task = env.get_agent_prompt(agents[0], scenario)

        protocol.run(task, mock_model)

        # prepare_input called twice per agent: once in _initialize_agents, once in _make_decision
        assert mock_model.prepare_input_calls == num_agents * 2

    def test_generate_latent_called_for_each_agent(self):
        """generate_latent should be called once per agent during initialization."""
        hidden_dim = 64
        num_agents = 4

        env = HiddenProfileEnvironment(num_agents=num_agents, num_groups=2, hidden_dim=hidden_dim)
        agents = env.create_agents()

        kv_cache = HierarchicalKVCache(num_groups=2, agents_per_group=2)
        attention = CrossLevelAttention(hidden_dim=hidden_dim)
        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=2,
            latent_steps=3,
        )
        protocol.register_agents(agents)

        mock_model = MockLCNModelWrapper(hidden_dim=hidden_dim)
        scenario = env.scenarios[0]
        task = env.get_agent_prompt(agents[0], scenario)

        protocol.run(task, mock_model)

        # generate_latent called once per agent during initialization
        assert mock_model.generate_latent_calls == num_agents

    def test_generate_text_called_for_each_agent(self):
        """generate_text should be called once per agent during decision making."""
        hidden_dim = 64
        num_agents = 4

        env = HiddenProfileEnvironment(num_agents=num_agents, num_groups=2, hidden_dim=hidden_dim)
        agents = env.create_agents()

        kv_cache = HierarchicalKVCache(num_groups=2, agents_per_group=2)
        attention = CrossLevelAttention(hidden_dim=hidden_dim)
        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=2,
            latent_steps=3,
        )
        protocol.register_agents(agents)

        mock_model = MockLCNModelWrapper(hidden_dim=hidden_dim)
        scenario = env.scenarios[0]
        task = env.get_agent_prompt(agents[0], scenario)

        protocol.run(task, mock_model)

        # generate_text called once per agent during decision making
        assert mock_model.generate_text_calls == num_agents


class TestMultipleScenarios:
    """Tests with multiple scenarios."""

    def test_scenario_1_correct_answer_is_c(self):
        """Scenario 1 should have correct_answer 'C'."""
        env = HiddenProfileEnvironment()
        assert env.scenarios[0].correct_answer == "C"

    def test_scenario_2_correct_answer_is_z(self):
        """Scenario 2 should have correct_answer 'Z'."""
        env = HiddenProfileEnvironment()
        assert env.scenarios[1].correct_answer == "Z"

    def test_run_with_scenario_2(self):
        """Integration test with second scenario (product choice)."""
        hidden_dim = 64

        env = HiddenProfileEnvironment(num_agents=6, num_groups=2, hidden_dim=hidden_dim)
        agents = env.create_agents()

        kv_cache = HierarchicalKVCache(num_groups=2, agents_per_group=3)
        attention = CrossLevelAttention(hidden_dim=hidden_dim)
        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=2,
            latent_steps=3,
        )
        protocol.register_agents(agents)

        # Mock model returns "Z" (correct answer for scenario 2)
        mock_model = MockLCNModelWrapper(hidden_dim=hidden_dim, correct_answer="Z")

        scenario = env.scenarios[1]  # correct_answer is "Z"
        task = env.get_agent_prompt(agents[0], scenario)

        result = protocol.run(task, mock_model)
        metrics = env.evaluate(result, scenario)

        assert metrics.accuracy == 1.0
        assert result.decision == "Z"


class TestKVCacheIntegration:
    """Tests for KV-Cache integration during consensus."""

    def test_kv_caches_populated_after_initialization(self):
        """KV-Caches should be populated for all agents after initialization."""
        hidden_dim = 64
        num_agents = 4

        env = HiddenProfileEnvironment(num_agents=num_agents, num_groups=2, hidden_dim=hidden_dim)
        agents = env.create_agents()

        kv_cache = HierarchicalKVCache(num_groups=2, agents_per_group=2)
        attention = CrossLevelAttention(hidden_dim=hidden_dim)
        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=2,
            latent_steps=3,
        )
        protocol.register_agents(agents)

        mock_model = MockLCNModelWrapper(hidden_dim=hidden_dim)
        scenario = env.scenarios[0]
        task = env.get_agent_prompt(agents[0], scenario)

        # Run consensus
        protocol.run(task, mock_model)

        # All agents should have local KV-Caches
        for agent in agents:
            local_cache = kv_cache.get_local(agent.agent_id)
            assert local_cache is not None

    def test_group_caches_populated_after_consensus(self):
        """Group caches should be populated after consensus rounds."""
        hidden_dim = 64

        env = HiddenProfileEnvironment(num_agents=4, num_groups=2, hidden_dim=hidden_dim)
        agents = env.create_agents()

        kv_cache = HierarchicalKVCache(num_groups=2, agents_per_group=2)
        attention = CrossLevelAttention(hidden_dim=hidden_dim)
        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=2,
            latent_steps=3,
        )
        protocol.register_agents(agents)

        mock_model = MockLCNModelWrapper(hidden_dim=hidden_dim)
        scenario = env.scenarios[0]
        task = env.get_agent_prompt(agents[0], scenario)

        protocol.run(task, mock_model)

        # Both groups should have aggregated caches
        assert 0 in kv_cache.group_caches
        assert 1 in kv_cache.group_caches

    def test_global_cache_populated_after_consensus(self):
        """Global cache should be populated after consensus rounds."""
        hidden_dim = 64

        env = HiddenProfileEnvironment(num_agents=4, num_groups=2, hidden_dim=hidden_dim)
        agents = env.create_agents()

        kv_cache = HierarchicalKVCache(num_groups=2, agents_per_group=2)
        attention = CrossLevelAttention(hidden_dim=hidden_dim)
        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=2,
            latent_steps=3,
        )
        protocol.register_agents(agents)

        mock_model = MockLCNModelWrapper(hidden_dim=hidden_dim)
        scenario = env.scenarios[0]
        task = env.get_agent_prompt(agents[0], scenario)

        protocol.run(task, mock_model)

        assert kv_cache.global_cache is not None


class TestAgentStateEvolution:
    """Tests for agent state evolution during consensus."""

    def test_agents_have_states_after_consensus(self):
        """All agents should have hidden states after consensus."""
        hidden_dim = 64

        env = HiddenProfileEnvironment(num_agents=4, num_groups=2, hidden_dim=hidden_dim)
        agents = env.create_agents()

        kv_cache = HierarchicalKVCache(num_groups=2, agents_per_group=2)
        attention = CrossLevelAttention(hidden_dim=hidden_dim)
        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=2,
            latent_steps=3,
        )
        protocol.register_agents(agents)

        mock_model = MockLCNModelWrapper(hidden_dim=hidden_dim)
        scenario = env.scenarios[0]
        task = env.get_agent_prompt(agents[0], scenario)

        protocol.run(task, mock_model)

        # All agents should have states
        for agent in agents:
            state = agent.get_state()
            assert state is not None
            assert state.shape[-1] == hidden_dim

    def test_agent_states_have_correct_shape(self):
        """Agent states should have shape [B, hidden_dim]."""
        hidden_dim = 128

        env = HiddenProfileEnvironment(num_agents=2, num_groups=1, hidden_dim=hidden_dim)
        agents = env.create_agents()

        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=2)
        attention = CrossLevelAttention(hidden_dim=hidden_dim)
        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=2,
            latent_steps=3,
        )
        protocol.register_agents(agents)

        mock_model = MockLCNModelWrapper(hidden_dim=hidden_dim)
        scenario = env.scenarios[0]
        task = env.get_agent_prompt(agents[0], scenario)

        protocol.run(task, mock_model)

        for agent in agents:
            state = agent.get_state()
            assert state.dim() == 2
            assert state.shape[0] == 1  # batch size
            assert state.shape[1] == hidden_dim


class TestDecisionDistribution:
    """Tests for decision distribution in metrics."""

    def test_unanimous_decision_distribution(self):
        """When all agents agree, decision_distribution should have one entry."""
        hidden_dim = 64

        env = HiddenProfileEnvironment(num_agents=4, num_groups=2, hidden_dim=hidden_dim)
        agents = env.create_agents()

        kv_cache = HierarchicalKVCache(num_groups=2, agents_per_group=2)
        attention = CrossLevelAttention(hidden_dim=hidden_dim)
        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=2,
            latent_steps=3,
        )
        protocol.register_agents(agents)

        mock_model = MockLCNModelWrapper(hidden_dim=hidden_dim, correct_answer="C")
        scenario = env.scenarios[0]
        task = env.get_agent_prompt(agents[0], scenario)

        result = protocol.run(task, mock_model)
        metrics = env.evaluate(result, scenario)

        # All agents answer "C"
        assert metrics.decision_distribution == {"C": 4}

    def test_decision_count_matches_agent_count(self):
        """Sum of decision_distribution counts should equal number of agents."""
        hidden_dim = 64
        num_agents = 6

        env = HiddenProfileEnvironment(num_agents=num_agents, num_groups=2, hidden_dim=hidden_dim)
        agents = env.create_agents()

        kv_cache = HierarchicalKVCache(num_groups=2, agents_per_group=3)
        attention = CrossLevelAttention(hidden_dim=hidden_dim)
        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=2,
            latent_steps=3,
        )
        protocol.register_agents(agents)

        mock_model = MockLCNModelWrapper(hidden_dim=hidden_dim)
        scenario = env.scenarios[0]
        task = env.get_agent_prompt(agents[0], scenario)

        result = protocol.run(task, mock_model)
        metrics = env.evaluate(result, scenario)

        total_decisions = sum(metrics.decision_distribution.values())
        assert total_decisions == num_agents


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_agent(self):
        """Integration should work with a single agent."""
        hidden_dim = 64

        env = HiddenProfileEnvironment(num_agents=1, num_groups=1, hidden_dim=hidden_dim)
        agents = env.create_agents()
        assert len(agents) == 1

        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=1)
        attention = CrossLevelAttention(hidden_dim=hidden_dim)
        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=2,
            latent_steps=3,
        )
        protocol.register_agents(agents)

        mock_model = MockLCNModelWrapper(hidden_dim=hidden_dim, correct_answer="C")
        scenario = env.scenarios[0]
        task = env.get_agent_prompt(agents[0], scenario)

        result = protocol.run(task, mock_model)
        metrics = env.evaluate(result, scenario)

        assert isinstance(result, ConsensusResult)
        assert isinstance(metrics, HiddenProfileMetrics)
        assert len(result.agent_decisions) == 1

    def test_single_group(self):
        """Integration should work with all agents in one group."""
        hidden_dim = 64

        env = HiddenProfileEnvironment(num_agents=4, num_groups=1, hidden_dim=hidden_dim)
        agents = env.create_agents()

        # All agents should be in group 0
        assert all(agent.group_id == 0 for agent in agents)

        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=4)
        attention = CrossLevelAttention(hidden_dim=hidden_dim)
        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=2,
            latent_steps=3,
        )
        protocol.register_agents(agents)

        mock_model = MockLCNModelWrapper(hidden_dim=hidden_dim)
        scenario = env.scenarios[0]
        task = env.get_agent_prompt(agents[0], scenario)

        result = protocol.run(task, mock_model)

        assert isinstance(result, ConsensusResult)
        assert len(result.agent_decisions) == 4

    def test_zero_rounds(self):
        """Integration should work with zero consensus rounds."""
        hidden_dim = 64

        env = HiddenProfileEnvironment(num_agents=4, num_groups=2, hidden_dim=hidden_dim)
        agents = env.create_agents()

        kv_cache = HierarchicalKVCache(num_groups=2, agents_per_group=2)
        attention = CrossLevelAttention(hidden_dim=hidden_dim)
        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=0,  # Zero rounds
            latent_steps=3,
        )
        protocol.register_agents(agents)

        mock_model = MockLCNModelWrapper(hidden_dim=hidden_dim)
        scenario = env.scenarios[0]
        task = env.get_agent_prompt(agents[0], scenario)

        result = protocol.run(task, mock_model)

        assert isinstance(result, ConsensusResult)
        assert result.attention_history == []
        assert len(result.agent_decisions) == 4

    def test_many_rounds(self):
        """Integration should work with many consensus rounds."""
        hidden_dim = 64

        env = HiddenProfileEnvironment(num_agents=4, num_groups=2, hidden_dim=hidden_dim)
        agents = env.create_agents()

        kv_cache = HierarchicalKVCache(num_groups=2, agents_per_group=2)
        attention = CrossLevelAttention(hidden_dim=hidden_dim)
        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=10,  # Many rounds
            latent_steps=3,
        )
        protocol.register_agents(agents)

        mock_model = MockLCNModelWrapper(hidden_dim=hidden_dim)
        scenario = env.scenarios[0]
        task = env.get_agent_prompt(agents[0], scenario)

        result = protocol.run(task, mock_model)

        assert isinstance(result, ConsensusResult)
        assert len(result.attention_history) == 10
        assert result.convergence_round == 10


class TestPromptConstruction:
    """Tests for prompt construction with scenarios."""

    def test_each_agent_gets_unique_prompt(self):
        """Each agent should get a unique prompt with their hidden info."""
        env = HiddenProfileEnvironment(num_agents=6, num_groups=2)
        agents = env.create_agents()
        scenario = env.scenarios[0]

        prompts = [env.get_agent_prompt(agent, scenario) for agent in agents]

        # Agents with different hidden info should get different prompts
        # Agents 0, 1, 2 have negative info about A/B
        # Agent 3 has positive info about C
        prompt_0 = prompts[0]
        prompt_3 = prompts[3]

        # Agent 0 sees "missed deadlines twice"
        assert "missed deadlines twice" in prompt_0
        # Agent 3 sees "exceptional problem solver"
        assert "exceptional problem solver" in prompt_3

    def test_all_prompts_contain_shared_info(self):
        """All agents should see the shared information."""
        env = HiddenProfileEnvironment(num_agents=4, num_groups=2)
        agents = env.create_agents()
        scenario = env.scenarios[0]

        for agent in agents:
            prompt = env.get_agent_prompt(agent, scenario)
            # All agents should see shared traits
            assert "good communicator" in prompt
            assert "team player" in prompt
            assert "strong leader" in prompt
            assert "reliable" in prompt
