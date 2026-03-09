"""Base environment abstractions for LCN experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from lcn.core.agent import LCNAgent


class BaseEnvironment(ABC):
    """Common interface for experiment environments."""

    def __init__(self, name: str, *, num_agents: int, num_groups: int):
        self.name = name
        self.num_agents = num_agents
        self.num_groups = num_groups

    @abstractmethod
    def setup_episode(self, seed: Optional[int] = None) -> Tuple[List[LCNAgent], Dict[str, Any]]:
        """Create agents and a task specification for one episode."""

    @abstractmethod
    def evaluate_episode(
        self,
        result: Mapping[str, Any],
        task: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Compute environment-specific metrics from the protocol result."""

    @staticmethod
    def _assign_group(agent_idx: int, *, num_agents: int, num_groups: int) -> int:
        group_size = max(num_agents // max(num_groups, 1), 1)
        return min(agent_idx // group_size, max(num_groups - 1, 0))

    @staticmethod
    def _make_agents(
        personas: Sequence[str],
        observations: Sequence[str],
        *,
        hidden_dim: int,
        num_groups: int,
    ) -> List[LCNAgent]:
        agents = []
        for idx, (persona, observation) in enumerate(zip(personas, observations)):
            agents.append(
                LCNAgent(
                    agent_id=idx,
                    group_id=BaseEnvironment._assign_group(
                        idx,
                        num_agents=len(personas),
                        num_groups=num_groups,
                    ),
                    hidden_dim=hidden_dim,
                    persona=persona,
                    metadata={"observation": observation},
                )
            )
        return agents
