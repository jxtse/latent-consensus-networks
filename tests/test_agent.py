# tests/test_agent.py
import pytest
import torch
from lcn.core.agent import LCNAgent


class TestLCNAgent:
    """Tests for LCNAgent."""

    def test_init_with_required_params(self):
        """Agent should initialize with required parameters."""
        agent = LCNAgent(
            agent_id=0,
            group_id=0,
            hidden_dim=64,
        )

        assert agent.agent_id == 0
        assert agent.group_id == 0
        assert agent.hidden_dim == 64
        assert agent.state is None

    def test_init_with_persona(self):
        """Agent should accept optional persona."""
        agent = LCNAgent(
            agent_id=0,
            group_id=0,
            hidden_dim=64,
            persona="A skeptical scientist",
        )

        assert agent.persona == "A skeptical scientist"

    def test_set_state(self):
        """set_state should update agent's hidden state."""
        agent = LCNAgent(agent_id=0, group_id=0, hidden_dim=64)

        state = torch.randn(1, 64)
        agent.set_state(state)

        assert agent.state is not None
        assert torch.equal(agent.state, state)

    def test_get_state_returns_clone(self):
        """get_state should return a clone to prevent mutation."""
        agent = LCNAgent(agent_id=0, group_id=0, hidden_dim=64)

        state = torch.randn(1, 64)
        agent.set_state(state)

        retrieved = agent.get_state()
        retrieved[0, 0] = 999.0  # Mutate

        # Original should be unchanged
        assert agent.state[0, 0] != 999.0

    def test_reset_clears_state(self):
        """reset should clear agent's state."""
        agent = LCNAgent(agent_id=0, group_id=0, hidden_dim=64)

        agent.set_state(torch.randn(1, 64))
        agent.reset()

        assert agent.state is None
