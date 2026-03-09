# tests/test_consensus.py
import pytest
import torch
from unittest.mock import Mock, MagicMock

from lcn.core.consensus import ConsensusProtocol
from lcn.core.agent import LCNAgent
from lcn.core.kv_cache import HierarchicalKVCache
from lcn.core.attention import CrossLevelAttention
from lcn.environments.hidden_profile import HiddenProfileEnvironment
from lcn.models import MockModelWrapper


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

    def test_run_executes_multi_round_consensus(self):
        """run should initialize agents, update caches, and return decisions."""
        environment = HiddenProfileEnvironment(hidden_dim=16)
        agents, task = environment.setup_episode()
        kv_cache = HierarchicalKVCache(num_groups=2, agents_per_group=2)
        attention = CrossLevelAttention(hidden_dim=16)
        model = MockModelWrapper(hidden_dim=16)
        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=2,
            latent_steps=1,
            model_wrapper=model,
        )
        protocol.register_agents(agents)

        result = protocol.run(task)

        assert result["decision"] in task["options"]
        assert len(result["agent_decisions"]) == 4
        assert len(result["history"]) == 2
        assert kv_cache.global_cache is not None

    def test_fuse_agent_context_aligns_attention_to_state_device(self):
        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=1)
        attention = CrossLevelAttention(hidden_dim=16)
        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=1,
            latent_steps=1,
        )

        agent = LCNAgent(agent_id=0, group_id=0, hidden_dim=16)
        state = torch.randn(1, 16)
        agent.set_state(state)
        protocol.register_agents([agent])
        kv_cache.update_local(
            agent_id=0,
            group_id=0,
            kv_cache=((state.unsqueeze(1).unsqueeze(1), state.unsqueeze(1).unsqueeze(1)),),
        )

        attention.to = Mock(return_value=attention)
        protocol._fuse_agent_context(agent)

        attention.to.assert_called_once_with(device=state.device, dtype=state.dtype)

    def test_initialize_and_decide_route_to_agent_specific_wrappers(self):
        agents = [
            LCNAgent(agent_id=0, group_id=0, hidden_dim=4, metadata={"observation": "obs0"}),
            LCNAgent(agent_id=1, group_id=0, hidden_dim=4, metadata={"observation": "obs1"}),
        ]
        task = {"prompt": "pick", "options": ["A", "B"], "option_hints": {"A": "hint a", "B": "hint b"}}
        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=2)
        attention = CrossLevelAttention(hidden_dim=4)

        class StubWrapper:
            def __init__(self, name: str):
                self.name = name
                self.hidden_dim = 4
                self.initialize_prompts = []
                self.decision_calls = 0

            def initialize_state(self, prompt, *, persona=None, metadata=None):
                self.initialize_prompts.append(prompt)
                value = 1.0 if self.name == "left" else 2.0
                return torch.full((1, 4), value)

            def build_kv_cache(self, state):
                return ((state.unsqueeze(1).unsqueeze(1), state.unsqueeze(1).unsqueeze(1)),)

            def latent_step(self, agent_state, fused_state, *, prompt=None, metadata=None, step_idx=0):
                return agent_state

            def choose_option(self, agent_state, options, *, metadata=None):
                self.decision_calls += 1
                return "A" if self.name == "left" else "B"

        wrappers = {
            0: StubWrapper("left"),
            1: StubWrapper("right"),
        }
        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=1,
            latent_steps=1,
            model_wrapper=wrappers,
        )
        protocol.register_agents(agents)

        result = protocol.run(task)

        assert wrappers[0].initialize_prompts == ["pick\nobs0"]
        assert wrappers[1].initialize_prompts == ["pick\nobs1"]
        assert wrappers[0].decision_calls == 1
        assert wrappers[1].decision_calls == 1
        assert result["agent_decisions"] == {0: "A", 1: "B"}
