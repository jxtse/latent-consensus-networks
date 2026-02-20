# tests/test_environments.py
"""Tests for LCN experiment environments."""

import pytest
from abc import ABC
from typing import List, Any

from lcn.environments.base import BaseEnvironment
from lcn.core.agent import LCNAgent
from lcn.core.results import ConsensusResult


class TestBaseEnvironmentABC:
    """Tests for BaseEnvironment abstract base class."""

    def test_is_abstract_base_class(self):
        """BaseEnvironment should be an abstract base class."""
        assert issubclass(BaseEnvironment, ABC)

    def test_cannot_instantiate_directly(self):
        """BaseEnvironment should not be instantiable directly."""
        with pytest.raises(TypeError):
            BaseEnvironment()

    def test_has_create_agents_abstract_method(self):
        """BaseEnvironment should have an abstract create_agents method."""
        assert hasattr(BaseEnvironment, "create_agents")
        # Check it's abstract
        assert "create_agents" in BaseEnvironment.__abstractmethods__

    def test_has_get_agent_prompt_abstract_method(self):
        """BaseEnvironment should have an abstract get_agent_prompt method."""
        assert hasattr(BaseEnvironment, "get_agent_prompt")
        assert "get_agent_prompt" in BaseEnvironment.__abstractmethods__

    def test_has_evaluate_abstract_method(self):
        """BaseEnvironment should have an abstract evaluate method."""
        assert hasattr(BaseEnvironment, "evaluate")
        assert "evaluate" in BaseEnvironment.__abstractmethods__


class ConcreteEnvironment(BaseEnvironment):
    """Concrete implementation of BaseEnvironment for testing."""

    def __init__(self, num_agents: int = 3, hidden_dim: int = 64):
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim

    def create_agents(self) -> List[LCNAgent]:
        """Create test agents."""
        return [
            LCNAgent(
                agent_id=i,
                group_id=0,
                hidden_dim=self.hidden_dim,
                persona=f"Agent {i}",
            )
            for i in range(self.num_agents)
        ]

    def get_agent_prompt(self, agent: LCNAgent, scenario: Any) -> str:
        """Get prompt for agent based on scenario."""
        return f"Agent {agent.agent_id}: {scenario}"

    def evaluate(self, result: ConsensusResult, scenario: Any) -> Any:
        """Evaluate the consensus result against the scenario."""
        return {
            "decision": result.decision,
            "correct": result.decision == scenario.get("correct_answer", ""),
        }


class TestConcreteEnvironment:
    """Tests verifying concrete implementations work correctly."""

    def test_concrete_environment_can_be_instantiated(self):
        """A concrete subclass should be instantiable."""
        env = ConcreteEnvironment()
        assert env is not None

    def test_create_agents_returns_list_of_lcn_agents(self):
        """create_agents should return a list of LCNAgent instances."""
        env = ConcreteEnvironment(num_agents=4)
        agents = env.create_agents()

        assert isinstance(agents, list)
        assert len(agents) == 4
        assert all(isinstance(a, LCNAgent) for a in agents)

    def test_create_agents_returns_agents_with_correct_ids(self):
        """create_agents should return agents with sequential IDs."""
        env = ConcreteEnvironment(num_agents=3)
        agents = env.create_agents()

        agent_ids = [a.agent_id for a in agents]
        assert agent_ids == [0, 1, 2]

    def test_get_agent_prompt_returns_string(self):
        """get_agent_prompt should return a string."""
        env = ConcreteEnvironment()
        agent = LCNAgent(agent_id=0, group_id=0, hidden_dim=64)
        scenario = {"task": "test task"}

        prompt = env.get_agent_prompt(agent, scenario)

        assert isinstance(prompt, str)

    def test_get_agent_prompt_includes_scenario_info(self):
        """get_agent_prompt should incorporate scenario information."""
        env = ConcreteEnvironment()
        agent = LCNAgent(agent_id=1, group_id=0, hidden_dim=64)
        scenario = "What is the capital of France?"

        prompt = env.get_agent_prompt(agent, scenario)

        assert "What is the capital of France?" in prompt

    def test_evaluate_returns_evaluation_result(self):
        """evaluate should return an evaluation of the consensus result."""
        env = ConcreteEnvironment()
        result = ConsensusResult(
            decision="Paris",
            agent_decisions={0: "Paris", 1: "Paris", 2: "Lyon"},
            attention_history=[],
            convergence_round=2,
        )
        scenario = {"correct_answer": "Paris"}

        evaluation = env.evaluate(result, scenario)

        assert evaluation["correct"] is True
        assert evaluation["decision"] == "Paris"

    def test_evaluate_with_incorrect_answer(self):
        """evaluate should correctly identify incorrect answers."""
        env = ConcreteEnvironment()
        result = ConsensusResult(
            decision="Lyon",
            agent_decisions={0: "Lyon", 1: "Lyon", 2: "Paris"},
            attention_history=[],
            convergence_round=3,
        )
        scenario = {"correct_answer": "Paris"}

        evaluation = env.evaluate(result, scenario)

        assert evaluation["correct"] is False
        assert evaluation["decision"] == "Lyon"


class TestBaseEnvironmentImport:
    """Tests for BaseEnvironment imports."""

    def test_import_from_environments_module(self):
        """BaseEnvironment should be importable from lcn.environments."""
        from lcn.environments import BaseEnvironment as ImportedBase

        assert ImportedBase is BaseEnvironment

    def test_import_from_base_module(self):
        """BaseEnvironment should be importable from lcn.environments.base."""
        from lcn.environments.base import BaseEnvironment as ImportedBase

        assert ImportedBase is not None
        assert issubclass(ImportedBase, ABC)
