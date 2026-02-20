# lcn/environments/hidden_profile.py
"""Hidden Profile Task experiment environment."""

from dataclasses import dataclass
from typing import Any, Dict, List

from lcn.core.agent import LCNAgent
from lcn.core.results import ConsensusResult
from lcn.environments.base import BaseEnvironment


@dataclass
class HiddenProfileScenario:
    """
    A single hidden profile scenario.

    In the hidden profile paradigm, agents must integrate partial information
    to make optimal group decisions. Each agent sees shared information plus
    their unique hidden information.

    Attributes:
        shared_info: Mapping from candidate name to list of shared traits
                    that all agents can see.
        hidden_info: Mapping from agent_id to dict of candidate -> hidden traits
                    that only that agent can see.
        correct_answer: The candidate that is optimal when all information
                       is integrated.
    """

    shared_info: Dict[str, List[str]]
    hidden_info: Dict[int, Dict[str, List[str]]]
    correct_answer: str


class HiddenProfileEnvironment(BaseEnvironment):
    """
    Hidden Profile Task experiment environment.

    Tests whether LCN can integrate distributed information
    to make optimal group decisions.

    In the hidden profile paradigm:
    - Each agent sees only partial information
    - Some information is shared by all agents
    - Some information is unique to each agent (hidden from others)
    - The correct answer requires integrating information from all agents

    Args:
        num_agents: Number of agents in the environment. Default is 6.
        num_groups: Number of groups to organize agents into. Default is 2.
        hidden_dim: Dimension of agent hidden states. Default is 64.
    """

    def __init__(
        self,
        num_agents: int = 6,
        num_groups: int = 2,
        hidden_dim: int = 64,
    ):
        self.num_agents = num_agents
        self.num_groups = num_groups
        self.hidden_dim = hidden_dim
        self.scenarios = self._load_scenarios()

    def _load_scenarios(self) -> List[HiddenProfileScenario]:
        """
        Load or generate hidden profile scenarios.

        Returns:
            List of HiddenProfileScenario instances.
        """
        # Hardcoded scenario based on the design doc example
        # Scenario: Choose best candidate (A, B, C)
        scenario1 = HiddenProfileScenario(
            shared_info={
                "A": ["good communicator", "team player"],
                "B": ["strong leader", "creative thinker"],
                "C": ["reliable", "organized"],
            },
            hidden_info={
                0: {"A": ["missed deadlines twice"]},
                1: {"A": ["conflict with previous team"]},
                2: {"B": ["lacks technical depth"]},
                3: {
                    "C": [
                        "exceptional problem solver",
                        "mentored junior staff",
                        "led successful project",
                    ]
                },
                4: {"B": ["requires close supervision"]},
                5: {"A": ["overpromises on deliverables"]},
            },
            correct_answer="C",
        )

        # Second scenario with different structure
        scenario2 = HiddenProfileScenario(
            shared_info={
                "X": ["affordable", "good warranty"],
                "Y": ["popular brand", "fast delivery"],
                "Z": ["eco-friendly", "durable"],
            },
            hidden_info={
                0: {"X": ["poor customer reviews"]},
                1: {"Y": ["known defects in batch"]},
                2: {"X": ["competitor's inferior model"]},
                3: {"Z": ["award-winning design", "best value rating"]},
                4: {"Y": ["frequent returns"]},
                5: {"Z": ["recommended by experts"]},
            },
            correct_answer="Z",
        )

        return [scenario1, scenario2]

    def create_agents(self) -> List[LCNAgent]:
        """
        Create agents with hidden profile personas.

        Agents are distributed evenly across groups. Each agent
        gets a unique persona describing their role.

        Returns:
            List of LCNAgent instances configured for the hidden profile task.
        """
        agents = []
        agents_per_group = self.num_agents // self.num_groups
        remainder = self.num_agents % self.num_groups

        # Persona templates for variety
        persona_templates = [
            "Analyst who focuses on data-driven decisions",
            "Experienced manager with intuition for people",
            "Detail-oriented researcher who values evidence",
            "Strategic thinker who considers long-term impact",
            "Pragmatic decision-maker focused on outcomes",
            "Collaborative team member who seeks consensus",
            "Critical evaluator who questions assumptions",
            "Innovative thinker who values creativity",
        ]

        for i in range(self.num_agents):
            # Distribute agents across groups evenly
            # First 'remainder' groups get one extra agent
            if remainder > 0:
                # Groups 0 to (remainder-1) get (agents_per_group + 1) agents
                cumulative = 0
                group_id = 0
                for g in range(self.num_groups):
                    group_size = agents_per_group + (1 if g < remainder else 0)
                    if i < cumulative + group_size:
                        group_id = g
                        break
                    cumulative += group_size
            else:
                group_id = i // agents_per_group

            # Cycle through personas if we have more agents than templates
            persona = persona_templates[i % len(persona_templates)]

            agent = LCNAgent(
                agent_id=i,
                group_id=group_id,
                hidden_dim=self.hidden_dim,
                persona=persona,
            )
            agents.append(agent)

        return agents

    def get_agent_prompt(self, agent: LCNAgent, scenario: Any) -> str:
        """
        Get agent's prompt with their unique information.

        Note: This is a stub implementation for Task 16.
        Full implementation in Task 17.

        Args:
            agent: The LCNAgent for which to generate the prompt.
            scenario: The HiddenProfileScenario to be presented.

        Returns:
            A string prompt for the agent (stub returns placeholder).
        """
        # Stub implementation - full implementation in Task 17
        return f"Agent {agent.agent_id}: Please evaluate the candidates."

    def evaluate(self, result: ConsensusResult, scenario: Any) -> Any:
        """
        Evaluate consensus result.

        Note: This is a stub implementation for Task 16.
        Full implementation in Task 18.

        Args:
            result: The ConsensusResult from the consensus protocol.
            scenario: The HiddenProfileScenario that was presented.

        Returns:
            Evaluation dict (stub returns basic structure).
        """
        # Stub implementation - full implementation in Task 18
        return {
            "decision": result.decision,
            "correct_answer": scenario.correct_answer,
            "is_correct": result.decision == scenario.correct_answer,
        }
