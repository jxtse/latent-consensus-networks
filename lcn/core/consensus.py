# lcn/core/consensus.py
"""Consensus formation protocol for LCN."""

from typing import Dict, List, Optional, TYPE_CHECKING
import torch

from lcn.core.agent import LCNAgent
from lcn.core.kv_cache import HierarchicalKVCache, KVCache
from lcn.core.attention import CrossLevelAttention
from lcn.core.results import ConsensusResult

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

    def _kv_to_repr(self, kv_cache: Optional[KVCache]) -> Optional[torch.Tensor]:
        """
        Convert a KV-Cache to a representation tensor for attention.

        Extracts a representation from the KV-Cache by reshaping the value
        tensors from the last layer. The values have shape [B, num_heads, L, head_dim]
        and are reshaped to [B, L, hidden_dim] where hidden_dim = num_heads * head_dim.

        Args:
            kv_cache: KV-Cache tuple of (key, value) per layer, or None

        Returns:
            Representation tensor with shape [B, L, D], or None if input is None
        """
        if kv_cache is None:
            return None

        # Get the last layer's value tensor
        # KV-Cache structure: ((k0, v0), (k1, v1), ...) where each k,v has shape [B, H, L, D_head]
        last_layer_values = kv_cache[-1][1]  # [B, num_heads, seq_len, head_dim]

        # Reshape from [B, num_heads, seq_len, head_dim] to [B, seq_len, hidden_dim]
        batch_size, num_heads, seq_len, head_dim = last_layer_values.shape
        hidden_dim = num_heads * head_dim

        # Transpose to [B, seq_len, num_heads, head_dim] then reshape to [B, seq_len, hidden_dim]
        repr_tensor = last_layer_values.transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)

        return repr_tensor

    def _kv_list_to_repr(self, kv_caches: List[KVCache]) -> Optional[torch.Tensor]:
        """
        Convert a list of KV-Caches to a combined representation tensor.

        Mean-pools across the list of KV-Caches, then converts to representation.

        Args:
            kv_caches: List of KV-Cache tuples

        Returns:
            Combined representation tensor with shape [B, L, D], or None if list is empty
        """
        if not kv_caches:
            return None

        # Mean-pool the KV-Caches first
        pooled = HierarchicalKVCache._mean_pool_kv_caches(kv_caches)
        return self._kv_to_repr(pooled)

    def run(self, task: str, model: "LCNModelWrapper") -> ConsensusResult:
        """
        Execute the full consensus formation loop.

        This method orchestrates the multi-round consensus formation process:
        1. Initialize agents with the task
        2. For each round:
           a. Aggregate group and global caches
           b. Each agent: get cross-level attention fusion
           c. Each agent: update state with fused representation
        3. Return placeholder ConsensusResult (decision making is Task 14)

        Note on KV-Cache vs Hidden State evolution:
        - KV-Caches are generated once during initialization and remain static
        - Hidden states evolve through attention fusion across rounds
        - The consensus process uses KV-Caches as contextual representations
          while the hidden states capture the evolving agent "opinions"
        - New KV-Caches are not generated in the loop (model calls happen
          only in initialization and final decision phases)

        Args:
            task: The task/question for agents to consider
            model: The LCNModelWrapper to use for generation

        Returns:
            ConsensusResult containing placeholder decision and attention history
        """
        # Step 1: Initialize agents
        self._initialize_agents(task, model)

        # Track attention history across rounds
        attention_history: List[Dict] = []

        # Step 2: Consensus rounds
        for round_idx in range(self.num_rounds):
            round_attention: Dict = {"round": round_idx, "agent_weights": {}}

            # Step 2a: Aggregate caches at group and global levels
            for group_id in self.group_ids:
                self.kv_cache.aggregate_group(group_id)
            self.kv_cache.aggregate_global()

            # Step 2b-d: For each agent, fuse and update
            for agent in self.agents:
                # Get all levels of KV-Cache for this agent
                local_caches, group_cache, global_cache = self.kv_cache.get_all_levels(
                    agent.agent_id
                )

                # Convert to representations
                local_repr = self._kv_list_to_repr(local_caches)
                group_repr = self._kv_to_repr(group_cache)
                global_repr = self._kv_to_repr(global_cache)

                # Get agent's current state as query
                query_state = agent.get_state()  # [B, D]

                # Fuse via cross-level attention
                fused_state, attn_weights = self.attention.forward(
                    query_state=query_state,
                    local_repr=local_repr,
                    group_repr=group_repr,
                    global_repr=global_repr,
                )

                # Store attention weights for this agent
                round_attention["agent_weights"][agent.agent_id] = attn_weights.detach().cpu().tolist()

                # Update agent state with fused representation
                # Note: KV-Caches remain static; only hidden states evolve through
                # attention fusion. This is by design - the KV-Caches provide stable
                # contextual representations while states capture evolving opinions.
                agent.set_state(fused_state)

            # Record this round's attention history
            attention_history.append(round_attention)

        # Step 3: Return placeholder ConsensusResult
        # (Task 14 will implement proper decision making)
        return ConsensusResult(
            decision="[placeholder - decision making not yet implemented]",
            agent_decisions={agent.agent_id: "" for agent in self.agents},
            attention_history=attention_history,
            convergence_round=None,
        )
