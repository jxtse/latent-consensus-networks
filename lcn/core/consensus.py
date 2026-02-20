# lcn/core/consensus.py
"""Consensus formation protocol for LCN."""

from typing import Dict, List, Optional
import torch

from lcn.core.agent import LCNAgent
from lcn.core.kv_cache import HierarchicalKVCache
from lcn.core.attention import CrossLevelAttention


class ConsensusProtocol:
    """
    Consensus formation protocol for Latent Consensus Networks.

    Orchestrates the multi-round consensus formation process where
    agents iteratively update their states based on cross-level attention.

    Args:
        kv_cache: Hierarchical KV-Cache manager
        attention: Cross-Level Attention module
        num_rounds: Number of consensus formation rounds
        latent_steps: Number of latent reasoning steps per agent per round
    """

    def __init__(
        self,
        kv_cache: HierarchicalKVCache,
        attention: CrossLevelAttention,
        num_rounds: int = 3,
        latent_steps: int = 5,
    ):
        self.kv_cache = kv_cache
        self.attention = attention
        self.num_rounds = num_rounds
        self.latent_steps = latent_steps

        self.agents: List[LCNAgent] = []
        self._agent_map: Dict[int, LCNAgent] = {}

    def register_agents(self, agents: List[LCNAgent]) -> None:
        """
        Register agents with the protocol.

        Args:
            agents: List of agents to register
        """
        self.agents = agents
        self._agent_map = {agent.agent_id: agent for agent in agents}

    def get_agent(self, agent_id: int) -> Optional[LCNAgent]:
        """Get an agent by ID."""
        return self._agent_map.get(agent_id)

    def get_agents_by_group(self, group_id: int) -> List[LCNAgent]:
        """Get all agents in a specific group."""
        return [agent for agent in self.agents if agent.group_id == group_id]

    @property
    def group_ids(self) -> List[int]:
        """Get all unique group IDs."""
        return list(set(agent.group_id for agent in self.agents))
