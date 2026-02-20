# tests/test_attention.py
import pytest
import torch
from lcn.core.attention import CrossLevelAttention


class TestCrossLevelAttention:
    """Tests for CrossLevelAttention mechanism."""

    def test_init_with_default_params(self):
        """Attention should initialize with default parameters."""
        attn = CrossLevelAttention(hidden_dim=64)

        assert attn.hidden_dim == 64
        assert attn.temperature == 1.0

    def test_forward_returns_fused_state_and_weights(self):
        """forward should return fused state and attention weights."""
        attn = CrossLevelAttention(hidden_dim=64)

        batch_size = 2
        query_state = torch.randn(batch_size, 64)

        # Create mock representations for each level
        local_repr = torch.randn(batch_size, 5, 64)   # 5 local tokens
        group_repr = torch.randn(batch_size, 3, 64)   # 3 group tokens
        global_repr = torch.randn(batch_size, 2, 64)  # 2 global tokens

        fused, weights = attn(query_state, local_repr, group_repr, global_repr)

        # Fused state should have shape [B, D]
        assert fused.shape == (batch_size, 64)

        # Weights should have shape [B, L_total]
        assert weights.shape == (batch_size, 5 + 3 + 2)

        # Weights should sum to 1
        assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size), atol=1e-5)

    def test_forward_with_missing_levels(self):
        """forward should handle None inputs for missing levels."""
        attn = CrossLevelAttention(hidden_dim=64)

        batch_size = 2
        query_state = torch.randn(batch_size, 64)
        local_repr = torch.randn(batch_size, 5, 64)

        # Group and global are None
        fused, weights = attn(query_state, local_repr, None, None)

        assert fused.shape == (batch_size, 64)
        assert weights.shape == (batch_size, 5)

    def test_temperature_affects_attention_sharpness(self):
        """Lower temperature should produce sharper attention."""
        attn_sharp = CrossLevelAttention(hidden_dim=64, temperature=0.1)
        attn_smooth = CrossLevelAttention(hidden_dim=64, temperature=10.0)

        query_state = torch.randn(1, 64)
        local_repr = torch.randn(1, 10, 64)

        _, weights_sharp = attn_sharp(query_state, local_repr, None, None)
        _, weights_smooth = attn_smooth(query_state, local_repr, None, None)

        # Sharp attention should have higher max weight (more concentrated)
        assert weights_sharp.max() > weights_smooth.max()

    def test_get_level_weights_returns_per_level_summary(self):
        """get_level_weights should return aggregated weights per level."""
        attn = CrossLevelAttention(hidden_dim=64)

        batch_size = 2
        query_state = torch.randn(batch_size, 64)
        local_repr = torch.randn(batch_size, 5, 64)
        group_repr = torch.randn(batch_size, 3, 64)
        global_repr = torch.randn(batch_size, 2, 64)

        fused, weights = attn(query_state, local_repr, group_repr, global_repr)

        level_weights = attn.get_level_weights(
            weights,
            local_len=5,
            group_len=3,
            global_len=2
        )

        # Should return dict with three keys
        assert "local" in level_weights
        assert "group" in level_weights
        assert "global" in level_weights

        # Level weights should sum to 1
        total = level_weights["local"] + level_weights["group"] + level_weights["global"]
        assert torch.allclose(total, torch.ones(batch_size), atol=1e-5)
