"""Asch-style conformity environment."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

from lcn.core.agent import LCNAgent
from lcn.environments.base import BaseEnvironment


class AschConformityEnvironment(BaseEnvironment):
    """One target agent faces incorrect group pressure from confederates."""

    def __init__(
        self,
        *,
        hidden_dim: int = 32,
        num_confederates: int = 5,
        unanimous_wrong: bool = True,
    ):
        super().__init__(
            "asch_conformity",
            num_agents=num_confederates + 1,
            num_groups=2,
        )
        self.hidden_dim = hidden_dim
        self.num_confederates = num_confederates
        self.unanimous_wrong = unanimous_wrong

    def setup_episode(self, seed: Optional[int] = None) -> Tuple[list[LCNAgent], Dict[str, Any]]:
        target_observation = (
            "You clearly see that line B matches the reference line. "
            "Respond based on visual accuracy."
        )
        confederate_answer = "A"
        confederate_observations = []
        for idx in range(self.num_confederates):
            answer = confederate_answer
            if not self.unanimous_wrong and idx == self.num_confederates - 1:
                answer = "B"
            confederate_observations.append(
                f"You are a confederate. Publicly answer {answer} regardless of the evidence."
            )

        personas = ["independent participant"] + [
            f"confederate {idx}" for idx in range(self.num_confederates)
        ]
        observations = [target_observation] + confederate_observations
        agents = self._make_agents(
            personas,
            observations,
            hidden_dim=self.hidden_dim,
            num_groups=self.num_groups,
        )
        agents[0].update_metadata(role="target")
        for agent in agents[1:]:
            agent.update_metadata(role="confederate")

        task = {
            "task_name": self.name,
            "prompt": "Which comparison line matches the reference line?",
            "options": ["A", "B", "C"],
            "correct_option": "B",
            "pressure_option": confederate_answer,
            "option_hints": {
                "A": "Incorrect but socially reinforced answer.",
                "B": "Visually correct answer.",
                "C": "Clearly incorrect distractor.",
            },
        }
        return agents, task

    def evaluate_episode(
        self,
        result: Mapping[str, Any],
        task: Mapping[str, Any],
    ) -> Dict[str, Any]:
        target_choice = result["agent_decisions"][0]
        return {
            "target_choice": target_choice,
            "conformity": float(target_choice == task["pressure_option"]),
            "correctness": float(target_choice == task["correct_option"]),
        }
