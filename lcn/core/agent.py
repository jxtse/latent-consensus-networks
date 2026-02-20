# lcn/core/agent.py
"""LCN Agent definition."""

from typing import Optional
import torch


class LCNAgent:
    """
    Agent in Latent Consensus Networks.

    Each agent belongs to a group and maintains its own hidden state.

    Args:
        agent_id: Unique identifier for the agent
        group_id: Group the agent belongs to
        hidden_dim: Dimension of hidden states
        persona: Optional persona description for the agent
    """

    def __init__(
        self,
        agent_id: int,
        group_id: int,
        hidden_dim: int,
        persona: Optional[str] = None,
    ):
        self.agent_id = agent_id
        self.group_id = group_id
        self.hidden_dim = hidden_dim
        self.persona = persona

        self.state: Optional[torch.Tensor] = None

    def set_state(self, state: torch.Tensor) -> None:
        """Set the agent's hidden state."""
        self.state = state.clone()

    def get_state(self) -> Optional[torch.Tensor]:
        """Get a clone of the agent's hidden state."""
        if self.state is None:
            return None
        return self.state.clone()

    def reset(self) -> None:
        """Reset the agent's state."""
        self.state = None

    def __repr__(self) -> str:
        return f"LCNAgent(id={self.agent_id}, group={self.group_id}, persona={self.persona})"
