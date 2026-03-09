"""LCN Agent definition."""

from typing import Any, Dict, Optional
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
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.agent_id = agent_id
        self.group_id = group_id
        self.hidden_dim = hidden_dim
        self.persona = persona
        self.metadata: Dict[str, Any] = metadata.copy() if metadata else {}

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

    def update_metadata(self, **kwargs: Any) -> None:
        """Update agent metadata used by environments and runners."""
        self.metadata.update(kwargs)

    def __repr__(self) -> str:
        return f"LCNAgent(id={self.agent_id}, group={self.group_id}, persona={self.persona})"
