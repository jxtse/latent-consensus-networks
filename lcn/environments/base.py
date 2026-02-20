# lcn/environments/base.py
"""Base class for LCN experiment environments."""

from abc import ABC, abstractmethod
from typing import Any, List

from lcn.core.agent import LCNAgent
from lcn.core.results import ConsensusResult


class BaseEnvironment(ABC):
    """
    Abstract base class for LCN experiment environments.

    An environment defines how agents are created, how prompts are generated
    for a given scenario, and how the consensus result is evaluated.

    Subclasses must implement:
        - create_agents(): Create the agents for the experiment
        - get_agent_prompt(agent, scenario): Generate a prompt for an agent
        - evaluate(result, scenario): Evaluate the consensus result
    """

    @abstractmethod
    def create_agents(self) -> List[LCNAgent]:
        """
        Create and return a list of LCNAgent instances for the experiment.

        Returns:
            List of LCNAgent instances configured for the environment.
        """
        ...

    @abstractmethod
    def get_agent_prompt(self, agent: LCNAgent, scenario: Any) -> str:
        """
        Get the prompt for an agent given a scenario.

        Args:
            agent: The LCNAgent for which to generate the prompt.
            scenario: The scenario/task to be presented to the agent.

        Returns:
            A string prompt for the agent.
        """
        ...

    @abstractmethod
    def evaluate(self, result: ConsensusResult, scenario: Any) -> Any:
        """
        Evaluate a ConsensusResult against the scenario.

        Args:
            result: The ConsensusResult from the consensus protocol.
            scenario: The scenario that was presented to the agents.

        Returns:
            Evaluation metrics/results (type depends on the specific environment).
        """
        ...
