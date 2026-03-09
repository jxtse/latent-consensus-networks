"""Wisdom of Crowds environment."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

from lcn.core.agent import LCNAgent
from lcn.environments.base import BaseEnvironment


class WisdomOfCrowdsEnvironment(BaseEnvironment):
    """Collective estimation task with multiple plausible numerical choices."""

    def __init__(self, *, hidden_dim: int = 32, num_agents: int = 6, num_groups: int = 3):
        super().__init__("wisdom_of_crowds", num_agents=num_agents, num_groups=num_groups)
        self.hidden_dim = hidden_dim

    def setup_episode(self, seed: Optional[int] = None) -> Tuple[list[LCNAgent], Dict[str, Any]]:
        target_value = 150
        candidate_estimates = [90, 120, 150, 180, 210]
        personas = [f"forecaster {idx}" for idx in range(self.num_agents)]
        observations = [
            "Your prior estimate is around 120 because you expect a conservative outcome.",
            "Your prior estimate is around 180 because you expect a growth scenario.",
            "Your prior estimate is around 150 based on balanced evidence.",
            "Your prior estimate is around 90 because you discount tail events.",
            "Your prior estimate is around 210 because you overweight momentum.",
            "Your prior estimate is around 150 after averaging multiple heuristics.",
        ][: self.num_agents]
        agents = self._make_agents(
            personas,
            observations,
            hidden_dim=self.hidden_dim,
            num_groups=self.num_groups,
        )
        task = {
            "task_name": self.name,
            "prompt": "Estimate the final quantity as accurately as possible.",
            "options": [str(value) for value in candidate_estimates],
            "correct_option": str(target_value),
            "target_value": target_value,
            "option_hints": {
                str(value): f"Estimate option {value}." for value in candidate_estimates
            },
        }
        return agents, task

    def evaluate_episode(
        self,
        result: Mapping[str, Any],
        task: Mapping[str, Any],
    ) -> Dict[str, Any]:
        group_estimate = int(result["decision"])
        target_value = int(task["target_value"])
        return {
            "group_error": abs(group_estimate - target_value),
            "is_exact": float(group_estimate == target_value),
        }
