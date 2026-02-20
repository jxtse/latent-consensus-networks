# tests/test_hidden_profile.py
"""Tests for HiddenProfileEnvironment."""

import pytest
from dataclasses import fields

from lcn.environments.base import BaseEnvironment
from lcn.environments.hidden_profile import (
    HiddenProfileEnvironment,
    HiddenProfileScenario,
)
from lcn.core.agent import LCNAgent
from lcn.core.results import ConsensusResult


class TestHiddenProfileScenario:
    """Tests for HiddenProfileScenario dataclass."""

    def test_is_dataclass(self):
        """HiddenProfileScenario should be a dataclass."""
        # Check it has __dataclass_fields__ attribute
        assert hasattr(HiddenProfileScenario, "__dataclass_fields__")

    def test_has_shared_info_field(self):
        """HiddenProfileScenario should have shared_info field."""
        field_names = [f.name for f in fields(HiddenProfileScenario)]
        assert "shared_info" in field_names

    def test_has_hidden_info_field(self):
        """HiddenProfileScenario should have hidden_info field."""
        field_names = [f.name for f in fields(HiddenProfileScenario)]
        assert "hidden_info" in field_names

    def test_has_correct_answer_field(self):
        """HiddenProfileScenario should have correct_answer field."""
        field_names = [f.name for f in fields(HiddenProfileScenario)]
        assert "correct_answer" in field_names

    def test_can_instantiate_with_valid_data(self):
        """HiddenProfileScenario should be instantiable with valid data."""
        shared_info = {
            "A": ["positive trait 1", "positive trait 2"],
            "B": ["positive trait 1", "positive trait 2"],
            "C": ["positive trait 1", "positive trait 2"],
        }
        hidden_info = {
            0: {"A": ["negative trait 1"]},
            1: {"A": ["negative trait 2"]},
            2: {"B": ["negative trait 1"]},
            3: {"C": ["positive trait 3", "positive trait 4", "positive trait 5"]},
        }
        correct_answer = "C"

        scenario = HiddenProfileScenario(
            shared_info=shared_info,
            hidden_info=hidden_info,
            correct_answer=correct_answer,
        )

        assert scenario.shared_info == shared_info
        assert scenario.hidden_info == hidden_info
        assert scenario.correct_answer == correct_answer


class TestHiddenProfileEnvironmentInit:
    """Tests for HiddenProfileEnvironment initialization."""

    def test_is_subclass_of_base_environment(self):
        """HiddenProfileEnvironment should extend BaseEnvironment."""
        assert issubclass(HiddenProfileEnvironment, BaseEnvironment)

    def test_default_initialization(self):
        """HiddenProfileEnvironment should initialize with default values."""
        env = HiddenProfileEnvironment()

        assert env.num_agents == 6
        assert env.num_groups == 2

    def test_custom_num_agents(self):
        """HiddenProfileEnvironment should accept custom num_agents."""
        env = HiddenProfileEnvironment(num_agents=4)

        assert env.num_agents == 4

    def test_custom_num_groups(self):
        """HiddenProfileEnvironment should accept custom num_groups."""
        env = HiddenProfileEnvironment(num_groups=3)

        assert env.num_groups == 3

    def test_custom_hidden_dim(self):
        """HiddenProfileEnvironment should accept custom hidden_dim."""
        env = HiddenProfileEnvironment(hidden_dim=128)

        assert env.hidden_dim == 128

    def test_default_hidden_dim(self):
        """HiddenProfileEnvironment should have default hidden_dim."""
        env = HiddenProfileEnvironment()

        assert env.hidden_dim == 64

    def test_scenarios_loaded_on_init(self):
        """HiddenProfileEnvironment should load scenarios on init."""
        env = HiddenProfileEnvironment()

        assert hasattr(env, "scenarios")
        assert isinstance(env.scenarios, list)
        assert len(env.scenarios) > 0


class TestHiddenProfileEnvironmentLoadScenarios:
    """Tests for _load_scenarios method."""

    def test_returns_list_of_scenarios(self):
        """_load_scenarios should return a list of HiddenProfileScenario."""
        env = HiddenProfileEnvironment()
        scenarios = env._load_scenarios()

        assert isinstance(scenarios, list)
        assert all(isinstance(s, HiddenProfileScenario) for s in scenarios)

    def test_scenarios_have_valid_shared_info(self):
        """Each scenario should have valid shared_info structure."""
        env = HiddenProfileEnvironment()
        scenarios = env._load_scenarios()

        for scenario in scenarios:
            assert isinstance(scenario.shared_info, dict)
            # Each candidate should map to a list of traits
            for candidate, traits in scenario.shared_info.items():
                assert isinstance(candidate, str)
                assert isinstance(traits, list)
                assert all(isinstance(t, str) for t in traits)

    def test_scenarios_have_valid_hidden_info(self):
        """Each scenario should have valid hidden_info structure."""
        env = HiddenProfileEnvironment()
        scenarios = env._load_scenarios()

        for scenario in scenarios:
            assert isinstance(scenario.hidden_info, dict)
            # Each agent_id should map to a dict of candidate -> traits
            for agent_id, info in scenario.hidden_info.items():
                assert isinstance(agent_id, int)
                assert isinstance(info, dict)
                for candidate, traits in info.items():
                    assert isinstance(candidate, str)
                    assert isinstance(traits, list)

    def test_scenarios_have_valid_correct_answer(self):
        """Each scenario should have a valid correct_answer."""
        env = HiddenProfileEnvironment()
        scenarios = env._load_scenarios()

        for scenario in scenarios:
            assert isinstance(scenario.correct_answer, str)
            assert len(scenario.correct_answer) > 0
            # correct_answer should be one of the candidates
            assert scenario.correct_answer in scenario.shared_info

    def test_at_least_one_scenario_exists(self):
        """_load_scenarios should return at least one scenario."""
        env = HiddenProfileEnvironment()
        scenarios = env._load_scenarios()

        assert len(scenarios) >= 1


class TestHiddenProfileEnvironmentCreateAgents:
    """Tests for create_agents method."""

    def test_returns_list_of_lcn_agents(self):
        """create_agents should return a list of LCNAgent instances."""
        env = HiddenProfileEnvironment(num_agents=4)
        agents = env.create_agents()

        assert isinstance(agents, list)
        assert len(agents) == 4
        assert all(isinstance(a, LCNAgent) for a in agents)

    def test_agents_have_sequential_ids(self):
        """Agents should have sequential agent_ids starting from 0."""
        env = HiddenProfileEnvironment(num_agents=6)
        agents = env.create_agents()

        agent_ids = [a.agent_id for a in agents]
        assert agent_ids == [0, 1, 2, 3, 4, 5]

    def test_agents_distributed_across_groups(self):
        """Agents should be distributed across groups."""
        env = HiddenProfileEnvironment(num_agents=6, num_groups=2)
        agents = env.create_agents()

        group_ids = [a.group_id for a in agents]
        # With 6 agents and 2 groups, expect 3 agents per group
        assert group_ids.count(0) == 3
        assert group_ids.count(1) == 3

    def test_agents_have_hidden_dim(self):
        """Agents should have the correct hidden_dim."""
        env = HiddenProfileEnvironment(num_agents=2, hidden_dim=128)
        agents = env.create_agents()

        assert all(a.hidden_dim == 128 for a in agents)

    def test_agents_have_personas(self):
        """Agents should have persona strings."""
        env = HiddenProfileEnvironment(num_agents=4)
        agents = env.create_agents()

        for agent in agents:
            assert agent.persona is not None
            assert isinstance(agent.persona, str)
            assert len(agent.persona) > 0

    def test_agents_distributed_with_uneven_groups(self):
        """Agents should distribute evenly when num_agents not divisible by num_groups."""
        env = HiddenProfileEnvironment(num_agents=5, num_groups=2)
        agents = env.create_agents()

        group_ids = [a.group_id for a in agents]
        # With 5 agents and 2 groups, expect approximately even distribution
        assert group_ids.count(0) in [2, 3]
        assert group_ids.count(1) in [2, 3]
        assert group_ids.count(0) + group_ids.count(1) == 5


class TestHiddenProfileEnvironmentGetAgentPrompt:
    """Tests for get_agent_prompt method (stub for now)."""

    def test_get_agent_prompt_exists(self):
        """get_agent_prompt method should exist."""
        env = HiddenProfileEnvironment()
        assert hasattr(env, "get_agent_prompt")
        assert callable(env.get_agent_prompt)

    def test_get_agent_prompt_returns_string(self):
        """get_agent_prompt should return a string."""
        env = HiddenProfileEnvironment()
        agents = env.create_agents()
        scenario = env.scenarios[0]

        prompt = env.get_agent_prompt(agents[0], scenario)

        assert isinstance(prompt, str)


class TestHiddenProfileEnvironmentEvaluate:
    """Tests for evaluate method (stub for now)."""

    def test_evaluate_exists(self):
        """evaluate method should exist."""
        env = HiddenProfileEnvironment()
        assert hasattr(env, "evaluate")
        assert callable(env.evaluate)

    def test_evaluate_accepts_result_and_scenario(self):
        """evaluate should accept ConsensusResult and scenario."""
        env = HiddenProfileEnvironment()
        scenario = env.scenarios[0]
        result = ConsensusResult(
            decision="C",
            agent_decisions={0: "C", 1: "C"},
            attention_history=[],
            convergence_round=1,
        )

        # Should not raise - just verify it runs
        evaluation = env.evaluate(result, scenario)

        # Stub can return anything, just verify it doesn't crash
        assert evaluation is not None


class TestHiddenProfileEnvironmentImport:
    """Tests for HiddenProfileEnvironment imports."""

    def test_import_from_environments_module(self):
        """HiddenProfileEnvironment should be importable from lcn.environments."""
        from lcn.environments import HiddenProfileEnvironment as ImportedEnv

        assert ImportedEnv is HiddenProfileEnvironment

    def test_import_scenario_from_environments_module(self):
        """HiddenProfileScenario should be importable from lcn.environments."""
        from lcn.environments import HiddenProfileScenario as ImportedScenario

        assert ImportedScenario is HiddenProfileScenario
