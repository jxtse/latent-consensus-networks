"""Hidden Profile environment for information integration experiments."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Mapping, Optional, Tuple

from lcn.core.agent import LCNAgent
from lcn.environments.base import BaseEnvironment


class HiddenProfileEnvironment(BaseEnvironment):
    """Classic hidden-profile setup where the best answer requires pooling unique clues."""

    def __init__(self, *, hidden_dim: int = 32, num_groups: int = 2):
        super().__init__("hidden_profile", num_agents=4, num_groups=num_groups)
        self.hidden_dim = hidden_dim

    def setup_episode(self, seed: Optional[int] = None) -> Tuple[list[LCNAgent], Dict[str, Any]]:
        shared_info = (
            "All candidates have two visible strengths. "
            "The best choice depends on integrating the private clues."
        )
        observations = [
            f"{shared_info} Private clue: Candidate A has one hidden drawback.",
            f"{shared_info} Private clue: Candidate A has another hidden drawback.",
            f"{shared_info} Private clue: Candidate B has one hidden drawback.",
            f"{shared_info} Private clue: Candidate C has three additional hidden strengths.",
        ]
        personas = [
            "detail-oriented analyst",
            "cautious reviewer",
            "process auditor",
            "opportunity scout",
        ]
        agents = self._make_agents(
            personas,
            observations,
            hidden_dim=self.hidden_dim,
            num_groups=self.num_groups,
        )
        task = {
            "task_name": self.name,
            "prompt": "Choose the best candidate after combining the team's evidence.",
            "options": ["A", "B", "C"],
            "correct_option": "C",
            "option_hints": {
                "A": "Candidate A has visible strengths but multiple hidden drawbacks.",
                "B": "Candidate B has visible strengths but one hidden drawback.",
                "C": "Candidate C has visible strengths and three hidden extra strengths.",
            },
        }
        return agents, task

    def evaluate_episode(
        self,
        result: Mapping[str, Any],
        task: Mapping[str, Any],
    ) -> Dict[str, Any]:
        decision = result["decision"]
        agent_decisions = result["agent_decisions"]
        counts = Counter(agent_decisions.values())
        return {
            "accuracy": float(decision == task["correct_option"]),
            "consensus_strength": counts[decision] / len(agent_decisions),
            "num_correct_agents": sum(
                1 for choice in agent_decisions.values() if choice == task["correct_option"]
            ),
        }
