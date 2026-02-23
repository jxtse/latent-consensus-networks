# tests/test_hidden_profile.py
"""Tests for HiddenProfileEnvironment."""

import pytest
from dataclasses import fields

from lcn.environments.base import BaseEnvironment
from lcn.environments.hidden_profile import (
    HiddenProfileEnvironment,
    HiddenProfileMetrics,
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
    """Tests for get_agent_prompt method."""

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

    def test_get_agent_prompt_contains_persona(self):
        """get_agent_prompt should include the agent's persona."""
        env = HiddenProfileEnvironment()
        agents = env.create_agents()
        scenario = env.scenarios[0]

        prompt = env.get_agent_prompt(agents[0], scenario)

        assert agents[0].persona in prompt

    def test_get_agent_prompt_contains_all_shared_info(self):
        """get_agent_prompt should include all shared info for all candidates."""
        env = HiddenProfileEnvironment()
        agents = env.create_agents()
        scenario = env.scenarios[0]

        prompt = env.get_agent_prompt(agents[0], scenario)

        # Check that all candidate names are in the prompt
        for candidate in scenario.shared_info:
            assert candidate in prompt

        # Check that all shared traits are present
        for candidate, traits in scenario.shared_info.items():
            for trait in traits:
                assert trait in prompt

    def test_get_agent_prompt_contains_agents_hidden_info(self):
        """get_agent_prompt should include the agent's own hidden info."""
        env = HiddenProfileEnvironment()
        agents = env.create_agents()
        scenario = env.scenarios[0]

        prompt = env.get_agent_prompt(agents[0], scenario)

        # Agent 0 has hidden info about candidate A
        assert agents[0].agent_id in scenario.hidden_info
        hidden_info = scenario.hidden_info[agents[0].agent_id]
        for candidate, traits in hidden_info.items():
            for trait in traits:
                assert trait in prompt

    def test_get_agent_prompt_excludes_other_agents_hidden_info(self):
        """get_agent_prompt should NOT include other agents' hidden info."""
        env = HiddenProfileEnvironment()
        agents = env.create_agents()
        scenario = env.scenarios[0]

        prompt = env.get_agent_prompt(agents[0], scenario)

        # Check that hidden info from other agents is NOT in the prompt
        for agent_id, hidden_info in scenario.hidden_info.items():
            if agent_id != agents[0].agent_id:
                for candidate, traits in hidden_info.items():
                    for trait in traits:
                        assert trait not in prompt, f"Found other agent's hidden info: {trait}"

    def test_get_agent_prompt_different_agents_get_different_hidden_info(self):
        """Different agents should receive different hidden info in their prompts."""
        env = HiddenProfileEnvironment()
        agents = env.create_agents()
        scenario = env.scenarios[0]

        prompt_0 = env.get_agent_prompt(agents[0], scenario)
        prompt_3 = env.get_agent_prompt(agents[3], scenario)

        # Agent 0 and 3 have different hidden info
        # Agent 0: {"A": ["missed deadlines twice"]}
        # Agent 3: {"C": ["exceptional problem solver", "mentored junior staff", "led successful project"]}

        # Verify each agent sees their own hidden info
        assert "missed deadlines twice" in prompt_0
        assert "exceptional problem solver" in prompt_3

        # Verify they don't see each other's hidden info
        assert "exceptional problem solver" not in prompt_0
        assert "missed deadlines twice" not in prompt_3

    def test_get_agent_prompt_includes_decision_instructions(self):
        """get_agent_prompt should include instructions to make a decision."""
        env = HiddenProfileEnvironment()
        agents = env.create_agents()
        scenario = env.scenarios[0]

        prompt = env.get_agent_prompt(agents[0], scenario)

        # Should ask for a decision/recommendation
        prompt_lower = prompt.lower()
        assert any(word in prompt_lower for word in ["recommend", "choose", "decision", "decide", "which candidate"])

    def test_get_agent_prompt_agent_with_no_hidden_info(self):
        """get_agent_prompt should work for agents with no hidden info."""
        # Create a scenario where agent 10 has no hidden info
        scenario = HiddenProfileScenario(
            shared_info={"A": ["trait1"], "B": ["trait2"]},
            hidden_info={0: {"A": ["secret"]}},
            correct_answer="A",
        )
        env = HiddenProfileEnvironment(num_agents=2)
        agents = env.create_agents()

        # Agent 1 has no hidden info in this scenario
        prompt = env.get_agent_prompt(agents[1], scenario)

        # Should still be a valid prompt
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        # Should still contain shared info
        assert "trait1" in prompt
        assert "trait2" in prompt
        # Should not crash or have hidden info section issues

    def test_get_agent_prompt_has_shared_info_section(self):
        """get_agent_prompt should clearly label shared information section."""
        env = HiddenProfileEnvironment()
        agents = env.create_agents()
        scenario = env.scenarios[0]

        prompt = env.get_agent_prompt(agents[0], scenario)

        # Should have a section header for shared info
        prompt_lower = prompt.lower()
        assert "shared" in prompt_lower

    def test_get_agent_prompt_has_private_info_section(self):
        """get_agent_prompt should clearly label private information section."""
        env = HiddenProfileEnvironment()
        agents = env.create_agents()
        scenario = env.scenarios[0]

        prompt = env.get_agent_prompt(agents[0], scenario)

        # Should have a section header for private info
        prompt_lower = prompt.lower()
        assert any(word in prompt_lower for word in ["private", "your", "unique", "only you"])


class TestHiddenProfileMetrics:
    """Tests for HiddenProfileMetrics dataclass."""

    def test_is_dataclass(self):
        """HiddenProfileMetrics should be a dataclass."""
        assert hasattr(HiddenProfileMetrics, "__dataclass_fields__")

    def test_has_accuracy_field(self):
        """HiddenProfileMetrics should have accuracy field."""
        field_names = [f.name for f in fields(HiddenProfileMetrics)]
        assert "accuracy" in field_names

    def test_has_information_integration_field(self):
        """HiddenProfileMetrics should have information_integration field."""
        field_names = [f.name for f in fields(HiddenProfileMetrics)]
        assert "information_integration" in field_names

    def test_has_individual_accuracy_field(self):
        """HiddenProfileMetrics should have individual_accuracy field."""
        field_names = [f.name for f in fields(HiddenProfileMetrics)]
        assert "individual_accuracy" in field_names

    def test_has_decision_distribution_field(self):
        """HiddenProfileMetrics should have decision_distribution field."""
        field_names = [f.name for f in fields(HiddenProfileMetrics)]
        assert "decision_distribution" in field_names

    def test_can_instantiate_with_valid_data(self):
        """HiddenProfileMetrics should be instantiable with valid data."""
        metrics = HiddenProfileMetrics(
            accuracy=1.0,
            information_integration=0.5,
            individual_accuracy={0: 1.0, 1: 0.0},
            decision_distribution={"A": 2, "B": 1},
        )

        assert metrics.accuracy == 1.0
        assert metrics.information_integration == 0.5
        assert metrics.individual_accuracy == {0: 1.0, 1: 0.0}
        assert metrics.decision_distribution == {"A": 2, "B": 1}


class TestHiddenProfileEnvironmentEvaluate:
    """Tests for evaluate method."""

    def test_evaluate_exists(self):
        """evaluate method should exist."""
        env = HiddenProfileEnvironment()
        assert hasattr(env, "evaluate")
        assert callable(env.evaluate)

    def test_evaluate_returns_hidden_profile_metrics(self):
        """evaluate should return HiddenProfileMetrics."""
        env = HiddenProfileEnvironment()
        scenario = env.scenarios[0]
        result = ConsensusResult(
            decision="C",
            agent_decisions={0: "C", 1: "C"},
            attention_history=[],
            convergence_round=1,
        )

        metrics = env.evaluate(result, scenario)

        assert isinstance(metrics, HiddenProfileMetrics)

    def test_evaluate_accuracy_correct_decision(self):
        """evaluate should return accuracy=1.0 when decision matches correct_answer."""
        env = HiddenProfileEnvironment()
        scenario = env.scenarios[0]  # correct_answer is "C"
        result = ConsensusResult(
            decision="C",
            agent_decisions={0: "C", 1: "A"},
            attention_history=[],
            convergence_round=1,
        )

        metrics = env.evaluate(result, scenario)

        assert metrics.accuracy == 1.0

    def test_evaluate_accuracy_incorrect_decision(self):
        """evaluate should return accuracy=0.0 when decision does not match correct_answer."""
        env = HiddenProfileEnvironment()
        scenario = env.scenarios[0]  # correct_answer is "C"
        result = ConsensusResult(
            decision="A",
            agent_decisions={0: "A", 1: "A"},
            attention_history=[],
            convergence_round=1,
        )

        metrics = env.evaluate(result, scenario)

        assert metrics.accuracy == 0.0

    def test_evaluate_information_integration_placeholder(self):
        """evaluate should return information_integration=0.0 as placeholder."""
        env = HiddenProfileEnvironment()
        scenario = env.scenarios[0]
        result = ConsensusResult(
            decision="C",
            agent_decisions={0: "C", 1: "C"},
            attention_history=[],
            convergence_round=1,
        )

        metrics = env.evaluate(result, scenario)

        # Placeholder implementation returns 0.0
        assert metrics.information_integration == 0.0

    def test_evaluate_individual_accuracy_all_correct(self):
        """evaluate should compute individual_accuracy for each agent."""
        env = HiddenProfileEnvironment()
        scenario = env.scenarios[0]  # correct_answer is "C"
        result = ConsensusResult(
            decision="C",
            agent_decisions={0: "C", 1: "C", 2: "C"},
            attention_history=[],
            convergence_round=1,
        )

        metrics = env.evaluate(result, scenario)

        assert metrics.individual_accuracy == {0: 1.0, 1: 1.0, 2: 1.0}

    def test_evaluate_individual_accuracy_mixed(self):
        """evaluate should compute individual_accuracy with mixed results."""
        env = HiddenProfileEnvironment()
        scenario = env.scenarios[0]  # correct_answer is "C"
        result = ConsensusResult(
            decision="C",
            agent_decisions={0: "C", 1: "A", 2: "B", 3: "C"},
            attention_history=[],
            convergence_round=1,
        )

        metrics = env.evaluate(result, scenario)

        assert metrics.individual_accuracy == {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0}

    def test_evaluate_individual_accuracy_all_incorrect(self):
        """evaluate should compute individual_accuracy when all agents are wrong."""
        env = HiddenProfileEnvironment()
        scenario = env.scenarios[0]  # correct_answer is "C"
        result = ConsensusResult(
            decision="A",
            agent_decisions={0: "A", 1: "B", 2: "A"},
            attention_history=[],
            convergence_round=1,
        )

        metrics = env.evaluate(result, scenario)

        assert metrics.individual_accuracy == {0: 0.0, 1: 0.0, 2: 0.0}

    def test_evaluate_decision_distribution(self):
        """evaluate should count decisions per option."""
        env = HiddenProfileEnvironment()
        scenario = env.scenarios[0]
        result = ConsensusResult(
            decision="C",
            agent_decisions={0: "A", 1: "A", 2: "B", 3: "C", 4: "C", 5: "C"},
            attention_history=[],
            convergence_round=1,
        )

        metrics = env.evaluate(result, scenario)

        assert metrics.decision_distribution == {"A": 2, "B": 1, "C": 3}

    def test_evaluate_decision_distribution_single_option(self):
        """evaluate should handle case where all agents choose the same option."""
        env = HiddenProfileEnvironment()
        scenario = env.scenarios[0]
        result = ConsensusResult(
            decision="C",
            agent_decisions={0: "C", 1: "C", 2: "C"},
            attention_history=[],
            convergence_round=1,
        )

        metrics = env.evaluate(result, scenario)

        assert metrics.decision_distribution == {"C": 3}

    def test_evaluate_decision_distribution_empty_agents(self):
        """evaluate should handle empty agent_decisions."""
        env = HiddenProfileEnvironment()
        scenario = env.scenarios[0]
        result = ConsensusResult(
            decision="C",
            agent_decisions={},
            attention_history=[],
            convergence_round=1,
        )

        metrics = env.evaluate(result, scenario)

        assert metrics.decision_distribution == {}
        assert metrics.individual_accuracy == {}

    def test_evaluate_with_scenario2(self):
        """evaluate should work with second scenario (correct_answer is Z)."""
        env = HiddenProfileEnvironment()
        scenario = env.scenarios[1]  # correct_answer is "Z"
        result = ConsensusResult(
            decision="Z",
            agent_decisions={0: "Z", 1: "X", 2: "Z"},
            attention_history=[],
            convergence_round=2,
        )

        metrics = env.evaluate(result, scenario)

        assert metrics.accuracy == 1.0
        assert metrics.individual_accuracy == {0: 1.0, 1: 0.0, 2: 1.0}
        assert metrics.decision_distribution == {"Z": 2, "X": 1}


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

    def test_import_metrics_from_environments_module(self):
        """HiddenProfileMetrics should be importable from lcn.environments."""
        from lcn.environments import HiddenProfileMetrics as ImportedMetrics

        assert ImportedMetrics is HiddenProfileMetrics
