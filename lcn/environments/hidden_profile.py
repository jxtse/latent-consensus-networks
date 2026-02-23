# lcn/environments/hidden_profile.py
"""Hidden Profile Task experiment environment."""

from dataclasses import dataclass
from typing import Dict, List

from lcn.core.agent import LCNAgent
from lcn.core.results import ConsensusResult
from lcn.environments.base import BaseEnvironment


@dataclass
class HiddenProfileMetrics:
    """
    Evaluation metrics for a hidden profile task.

    Captures how well the group and individual agents performed
    in integrating hidden information and reaching the correct decision.

    Attributes:
        accuracy: 1.0 if the group chose the correct answer, 0.0 otherwise.
        information_integration: Placeholder for measuring how much hidden
                                info influenced the decision (currently 0.0).
        individual_accuracy: Per-agent accuracy, mapping agent_id to 1.0/0.0
                            based on whether they chose the correct answer.
        decision_distribution: Count of how many agents chose each option.
    """

    accuracy: float
    information_integration: float
    individual_accuracy: Dict[int, float]
    decision_distribution: Dict[str, int]


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

    def get_agent_prompt(self, agent: LCNAgent, scenario: HiddenProfileScenario) -> str:
        """
        Build prompt with shared info + agent's unique hidden info.

        In the hidden profile paradigm, each agent sees:
        - All shared information (known to everyone)
        - Only their own private/hidden information

        Args:
            agent: The LCNAgent for which to generate the prompt.
            scenario: The HiddenProfileScenario to be presented.

        Returns:
            A formatted prompt string for the agent.
        """
        # Build persona section
        persona_section = f"You are a {agent.persona}."

        # Build task introduction
        candidates = list(scenario.shared_info.keys())
        candidates_str = ", ".join(candidates[:-1]) + f", or {candidates[-1]}" if len(candidates) > 1 else candidates[0]
        task_intro = f"You are part of a team making a hiring decision. Choose the best candidate ({candidates_str})."

        # Build shared information section
        shared_lines = ["SHARED INFORMATION (known to all):"]
        for candidate, traits in scenario.shared_info.items():
            shared_lines.append(f"Candidate {candidate}:")
            for trait in traits:
                shared_lines.append(f"- {trait}")
            shared_lines.append("")  # Empty line between candidates
        shared_section = "\n".join(shared_lines).rstrip()

        # Build private information section
        private_lines = ["YOUR PRIVATE INFORMATION (only you know this):"]
        agent_hidden = scenario.hidden_info.get(agent.agent_id, {})
        if agent_hidden:
            for candidate, traits in agent_hidden.items():
                private_lines.append(f"Candidate {candidate}:")
                for trait in traits:
                    private_lines.append(f"- {trait}")
        else:
            private_lines.append("(You have no additional private information.)")
        private_section = "\n".join(private_lines)

        # Build closing instruction
        closing = "Based on all available information, which candidate would you recommend? Explain your reasoning."

        # Combine all sections
        prompt = f"""{persona_section}

{task_intro}

{shared_section}

{private_section}

{closing}"""

        return prompt

    def evaluate(
        self, result: ConsensusResult, scenario: HiddenProfileScenario
    ) -> HiddenProfileMetrics:
        """
        Evaluate consensus result against the scenario.

        Computes evaluation metrics including group accuracy, individual
        accuracy per agent, and decision distribution.

        Args:
            result: The ConsensusResult from the consensus protocol.
            scenario: The HiddenProfileScenario that was presented.

        Returns:
            HiddenProfileMetrics with computed evaluation metrics.
        """
        # Compute group accuracy
        accuracy = 1.0 if result.decision == scenario.correct_answer else 0.0

        # Placeholder for information integration
        # TODO: Implement proper analysis of how hidden info influenced decisions
        information_integration = 0.0

        # Compute individual accuracy
        individual_accuracy: Dict[int, float] = {}
        for agent_id, decision in result.agent_decisions.items():
            individual_accuracy[agent_id] = (
                1.0 if decision == scenario.correct_answer else 0.0
            )

        # Compute decision distribution
        decision_distribution: Dict[str, int] = {}
        for decision in result.agent_decisions.values():
            decision_distribution[decision] = decision_distribution.get(decision, 0) + 1

        return HiddenProfileMetrics(
            accuracy=accuracy,
            information_integration=information_integration,
            individual_accuracy=individual_accuracy,
            decision_distribution=decision_distribution,
        )
