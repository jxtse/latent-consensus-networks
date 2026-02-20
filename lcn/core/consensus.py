# lcn/core/consensus.py
"""Consensus formation protocol for LCN."""

from typing import Dict, List, Optional, TYPE_CHECKING
import torch

from lcn.core.agent import LCNAgent
from lcn.core.kv_cache import HierarchicalKVCache
from lcn.core.attention import CrossLevelAttention

if TYPE_CHECKING:
    from lcn.models.model_wrapper import LCNModelWrapper


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

    def _build_agent_prompt(self, agent: LCNAgent, task: str) -> List[Dict]:
        """
        Build chat messages for an agent.

        Creates a message list that includes the agent's persona (if present)
        and the task/question to be answered.

        Args:
            agent: The agent for whom to build the prompt
            task: The task or question to be answered

        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        content_parts = []

        # Include persona if agent has one
        if agent.persona:
            content_parts.append(agent.persona)

        # Include the task
        content_parts.append(task)

        content = "\n\n".join(content_parts)

        return [{"role": "user", "content": content}]

    def _initialize_agents(self, task: str, model: "LCNModelWrapper") -> None:
        """
        Initialize all agents with task, generate initial latent states.

        For each registered agent:
        1. Build their prompt using _build_agent_prompt
        2. Prepare input using model.prepare_input()
        3. Generate initial latent state using model.generate_latent()
        4. Store the hidden state in the agent via agent.set_state()
        5. Store the KV-Cache in self.kv_cache.update_local()

        Args:
            task: The task/question for agents to consider
            model: The LCNModelWrapper to use for generation
        """
        for agent in self.agents:
            # Build prompt for this agent
            messages = self._build_agent_prompt(agent, task)

            # Prepare input tokens
            input_ids, attention_mask = model.prepare_input(messages)

            # Generate latent state
            kv_cache, hidden_state = model.generate_latent(
                input_ids=input_ids,
                attention_mask=attention_mask,
                latent_steps=self.latent_steps,
            )

            # Store hidden state in agent
            agent.set_state(hidden_state)

            # Store KV-Cache in hierarchical cache
            self.kv_cache.update_local(agent.agent_id, agent.group_id, kv_cache)
