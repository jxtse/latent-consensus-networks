# tests/test_model_wrapper.py
"""Tests for LCNModelWrapper."""

import pytest
import torch
from unittest.mock import MagicMock, patch


class TestLCNModelWrapperInit:
    """Tests for LCNModelWrapper initialization."""

    def test_init_stores_model_name(self):
        """Should store the model name."""
        from lcn.models.model_wrapper import LCNModelWrapper

        wrapper = LCNModelWrapper(
            model_name="test-model",
            device=torch.device("cpu"),
        )

        assert wrapper.model_name == "test-model"

    def test_init_stores_device(self):
        """Should store the device."""
        from lcn.models.model_wrapper import LCNModelWrapper

        device = torch.device("cpu")
        wrapper = LCNModelWrapper(
            model_name="test-model",
            device=device,
        )

        assert wrapper.device == device

    def test_init_latent_space_realign_default_false(self):
        """latent_space_realign should default to False."""
        from lcn.models.model_wrapper import LCNModelWrapper

        wrapper = LCNModelWrapper(
            model_name="test-model",
            device=torch.device("cpu"),
        )

        assert wrapper.latent_space_realign is False

    def test_init_latent_space_realign_can_be_true(self):
        """latent_space_realign should accept True."""
        from lcn.models.model_wrapper import LCNModelWrapper

        wrapper = LCNModelWrapper(
            model_name="test-model",
            device=torch.device("cpu"),
            latent_space_realign=True,
        )

        assert wrapper.latent_space_realign is True

    def test_init_tokenizer_is_none(self):
        """Tokenizer should be None before loading."""
        from lcn.models.model_wrapper import LCNModelWrapper

        wrapper = LCNModelWrapper(
            model_name="test-model",
            device=torch.device("cpu"),
        )

        assert wrapper.tokenizer is None

    def test_init_model_is_none(self):
        """Model should be None before loading."""
        from lcn.models.model_wrapper import LCNModelWrapper

        wrapper = LCNModelWrapper(
            model_name="test-model",
            device=torch.device("cpu"),
        )

        assert wrapper.model is None


class TestLCNModelWrapperPrepareInput:
    """Tests for LCNModelWrapper.prepare_input method."""

    def test_prepare_input_returns_tuple_of_two_tensors(self):
        """prepare_input should return (input_ids, attention_mask)."""
        from lcn.models.model_wrapper import LCNModelWrapper

        wrapper = LCNModelWrapper(
            model_name="test-model",
            device=torch.device("cpu"),
        )

        # Mock the tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None
        mock_tokenizer.apply_chat_template = MagicMock(
            return_value="<|user|>\nHello\n</|user|>\n<|assistant|>"
        )
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        wrapper.tokenizer = mock_tokenizer

        messages = [{"role": "user", "content": "Hello"}]
        input_ids, attention_mask = wrapper.prepare_input(messages)

        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(attention_mask, torch.Tensor)

    def test_prepare_input_moves_tensors_to_device(self):
        """prepare_input should move tensors to the wrapper's device."""
        from lcn.models.model_wrapper import LCNModelWrapper

        device = torch.device("cpu")
        wrapper = LCNModelWrapper(
            model_name="test-model",
            device=device,
        )

        # Mock the tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        wrapper.tokenizer = mock_tokenizer

        messages = [{"role": "user", "content": "Hello"}]
        input_ids, attention_mask = wrapper.prepare_input(messages)

        assert input_ids.device == device
        assert attention_mask.device == device

    def test_prepare_input_handles_multiple_messages(self):
        """prepare_input should handle conversation with multiple messages."""
        from lcn.models.model_wrapper import LCNModelWrapper

        wrapper = LCNModelWrapper(
            model_name="test-model",
            device=torch.device("cpu"),
        )

        # Mock the tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = "template"
        mock_tokenizer.apply_chat_template = MagicMock(
            return_value="User: Hello\nAssistant: Hi\nUser: How are you?"
        )
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }
        wrapper.tokenizer = mock_tokenizer

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "How are you?"},
        ]
        input_ids, attention_mask = wrapper.prepare_input(messages)

        assert input_ids.shape[1] == 5
        assert attention_mask.shape[1] == 5

    def test_prepare_input_uses_chat_template_when_available(self):
        """prepare_input should use chat_template if tokenizer has one."""
        from lcn.models.model_wrapper import LCNModelWrapper

        wrapper = LCNModelWrapper(
            model_name="test-model",
            device=torch.device("cpu"),
        )

        # Mock the tokenizer with chat_template
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = "some template"
        mock_tokenizer.apply_chat_template = MagicMock(
            return_value="formatted prompt"
        )
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        wrapper.tokenizer = mock_tokenizer

        messages = [{"role": "user", "content": "Hello"}]
        wrapper.prepare_input(messages)

        mock_tokenizer.apply_chat_template.assert_called_once()

    def test_prepare_input_fallback_without_chat_template(self):
        """prepare_input should fallback to manual formatting without chat_template."""
        from lcn.models.model_wrapper import LCNModelWrapper

        wrapper = LCNModelWrapper(
            model_name="test-model",
            device=torch.device("cpu"),
        )

        # Mock the tokenizer without chat_template
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        wrapper.tokenizer = mock_tokenizer

        messages = [{"role": "user", "content": "Hello"}]
        wrapper.prepare_input(messages)

        # Should be called with some manually formatted string
        mock_tokenizer.assert_called_once()
        call_args = mock_tokenizer.call_args
        # The first positional arg should be a string containing the message content
        prompt = call_args[0][0]
        assert "Hello" in prompt


class TestLCNModelWrapperRenderChat:
    """Tests for LCNModelWrapper.render_chat method."""

    def test_render_chat_uses_template_when_available(self):
        """render_chat should use tokenizer's chat_template when available."""
        from lcn.models.model_wrapper import LCNModelWrapper

        wrapper = LCNModelWrapper(
            model_name="test-model",
            device=torch.device("cpu"),
        )

        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = "template"
        mock_tokenizer.apply_chat_template = MagicMock(return_value="formatted")
        wrapper.tokenizer = mock_tokenizer

        messages = [{"role": "user", "content": "Hello"}]
        result = wrapper.render_chat(messages)

        assert result == "formatted"
        mock_tokenizer.apply_chat_template.assert_called_once_with(
            messages, tokenize=False, add_generation_prompt=True
        )

    def test_render_chat_fallback_format(self):
        """render_chat should use fallback format without chat_template."""
        from lcn.models.model_wrapper import LCNModelWrapper

        wrapper = LCNModelWrapper(
            model_name="test-model",
            device=torch.device("cpu"),
        )

        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None
        wrapper.tokenizer = mock_tokenizer

        messages = [{"role": "user", "content": "Hello"}]
        result = wrapper.render_chat(messages)

        assert "<|user|>" in result
        assert "Hello" in result
        assert "<|assistant|>" in result

    def test_render_chat_add_generation_prompt_false(self):
        """render_chat should respect add_generation_prompt=False."""
        from lcn.models.model_wrapper import LCNModelWrapper

        wrapper = LCNModelWrapper(
            model_name="test-model",
            device=torch.device("cpu"),
        )

        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None
        wrapper.tokenizer = mock_tokenizer

        messages = [{"role": "user", "content": "Hello"}]
        result = wrapper.render_chat(messages, add_generation_prompt=False)

        # Should not end with assistant prompt
        assert not result.endswith("<|assistant|>")


class TestLCNModelWrapperExport:
    """Tests for LCNModelWrapper export from lcn.models."""

    def test_can_import_from_models_package(self):
        """LCNModelWrapper should be importable from lcn.models."""
        from lcn.models import LCNModelWrapper

        assert LCNModelWrapper is not None


class TestPastLength:
    """Tests for _past_length static method."""

    def test_past_length_with_none_returns_zero(self):
        """_past_length should return 0 for None."""
        from lcn.models.model_wrapper import LCNModelWrapper

        result = LCNModelWrapper._past_length(None)
        assert result == 0

    def test_past_length_with_empty_tuple_returns_zero(self):
        """_past_length should return 0 for empty tuple."""
        from lcn.models.model_wrapper import LCNModelWrapper

        result = LCNModelWrapper._past_length(())
        assert result == 0

    def test_past_length_extracts_sequence_length_from_kv_cache(self):
        """_past_length should extract seq_len from KV-Cache structure."""
        from lcn.models.model_wrapper import LCNModelWrapper

        # KV-Cache structure: tuple of (key, value) per layer
        # key/value shape: [batch, num_heads, seq_len, head_dim]
        batch, num_heads, seq_len, head_dim = 1, 8, 10, 64
        key = torch.randn(batch, num_heads, seq_len, head_dim)
        value = torch.randn(batch, num_heads, seq_len, head_dim)
        kv_cache = ((key, value),)  # Single layer

        result = LCNModelWrapper._past_length(kv_cache)
        assert result == seq_len

    def test_past_length_works_with_multi_layer_cache(self):
        """_past_length should work with multi-layer KV-Cache."""
        from lcn.models.model_wrapper import LCNModelWrapper

        batch, num_heads, seq_len, head_dim = 2, 12, 20, 64
        layer1 = (
            torch.randn(batch, num_heads, seq_len, head_dim),
            torch.randn(batch, num_heads, seq_len, head_dim),
        )
        layer2 = (
            torch.randn(batch, num_heads, seq_len, head_dim),
            torch.randn(batch, num_heads, seq_len, head_dim),
        )
        kv_cache = (layer1, layer2)

        result = LCNModelWrapper._past_length(kv_cache)
        assert result == seq_len


class TestGenerateLatent:
    """Tests for LCNModelWrapper.generate_latent method."""

    def test_generate_latent_raises_error_when_model_not_loaded(self):
        """generate_latent should raise RuntimeError if model is None."""
        from lcn.models.model_wrapper import LCNModelWrapper

        wrapper = LCNModelWrapper(
            model_name="test-model",
            device=torch.device("cpu"),
        )
        # model is None by default

        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.tensor([[1, 1, 1]])

        with pytest.raises(RuntimeError, match="[Mm]odel.*not.*loaded"):
            wrapper.generate_latent(
                input_ids=input_ids,
                attention_mask=attention_mask,
                latent_steps=0,
            )

    def test_generate_latent_raises_error_for_non_2d_input(self):
        """generate_latent should raise ValueError for non-2D input_ids."""
        from lcn.models.model_wrapper import LCNModelWrapper

        wrapper = LCNModelWrapper(
            model_name="test-model",
            device=torch.device("cpu"),
        )
        wrapper.model = MagicMock()  # Pretend model is loaded

        # 1D input instead of 2D
        input_ids = torch.tensor([1, 2, 3])
        attention_mask = torch.tensor([1, 1, 1])

        with pytest.raises(ValueError, match="2D"):
            wrapper.generate_latent(
                input_ids=input_ids,
                attention_mask=attention_mask,
                latent_steps=0,
            )

    def test_generate_latent_returns_tuple_of_kv_cache_and_hidden_state(self):
        """generate_latent should return (kv_cache, hidden_state)."""
        from lcn.models.model_wrapper import LCNModelWrapper

        wrapper = LCNModelWrapper(
            model_name="test-model",
            device=torch.device("cpu"),
        )

        # Mock model
        batch_size, seq_len, hidden_dim = 1, 5, 128
        num_layers, num_heads, head_dim = 2, 4, 32

        mock_kv_cache = tuple(
            (
                torch.randn(batch_size, num_heads, seq_len, head_dim),
                torch.randn(batch_size, num_heads, seq_len, head_dim),
            )
            for _ in range(num_layers)
        )
        mock_hidden_states = tuple(
            torch.randn(batch_size, seq_len, hidden_dim) for _ in range(num_layers + 1)
        )

        mock_outputs = MagicMock()
        mock_outputs.past_key_values = mock_kv_cache
        mock_outputs.hidden_states = mock_hidden_states

        mock_model = MagicMock()
        mock_model.return_value = mock_outputs
        wrapper.model = mock_model

        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 1]])

        result = wrapper.generate_latent(
            input_ids=input_ids,
            attention_mask=attention_mask,
            latent_steps=0,
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        kv_cache, hidden_state = result
        # KV-Cache is a tuple of tuples
        assert isinstance(kv_cache, tuple)
        # Hidden state is a tensor
        assert isinstance(hidden_state, torch.Tensor)

    def test_generate_latent_hidden_state_shape_is_batch_by_hidden_dim(self):
        """generate_latent should return hidden_state with shape [B, D]."""
        from lcn.models.model_wrapper import LCNModelWrapper

        wrapper = LCNModelWrapper(
            model_name="test-model",
            device=torch.device("cpu"),
        )

        batch_size, seq_len, hidden_dim = 2, 5, 128
        num_layers, num_heads, head_dim = 2, 4, 32

        mock_kv_cache = tuple(
            (
                torch.randn(batch_size, num_heads, seq_len, head_dim),
                torch.randn(batch_size, num_heads, seq_len, head_dim),
            )
            for _ in range(num_layers)
        )
        mock_hidden_states = tuple(
            torch.randn(batch_size, seq_len, hidden_dim) for _ in range(num_layers + 1)
        )

        mock_outputs = MagicMock()
        mock_outputs.past_key_values = mock_kv_cache
        mock_outputs.hidden_states = mock_hidden_states

        mock_model = MagicMock()
        mock_model.return_value = mock_outputs
        wrapper.model = mock_model

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        kv_cache, hidden_state = wrapper.generate_latent(
            input_ids=input_ids,
            attention_mask=attention_mask,
            latent_steps=0,
        )

        assert hidden_state.shape == (batch_size, hidden_dim)

    def test_generate_latent_calls_model_with_correct_flags(self):
        """generate_latent should call model with use_cache=True, output_hidden_states=True."""
        from lcn.models.model_wrapper import LCNModelWrapper

        wrapper = LCNModelWrapper(
            model_name="test-model",
            device=torch.device("cpu"),
        )

        batch_size, seq_len, hidden_dim = 1, 3, 64
        num_layers, num_heads, head_dim = 1, 2, 32

        mock_kv_cache = tuple(
            (
                torch.randn(batch_size, num_heads, seq_len, head_dim),
                torch.randn(batch_size, num_heads, seq_len, head_dim),
            )
            for _ in range(num_layers)
        )
        mock_hidden_states = tuple(
            torch.randn(batch_size, seq_len, hidden_dim) for _ in range(num_layers + 1)
        )

        mock_outputs = MagicMock()
        mock_outputs.past_key_values = mock_kv_cache
        mock_outputs.hidden_states = mock_hidden_states

        mock_model = MagicMock()
        mock_model.return_value = mock_outputs
        wrapper.model = mock_model

        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.tensor([[1, 1, 1]])

        wrapper.generate_latent(
            input_ids=input_ids,
            attention_mask=attention_mask,
            latent_steps=0,
        )

        # Verify model was called with correct flags
        mock_model.assert_called_once()
        call_kwargs = mock_model.call_args[1]
        assert call_kwargs["use_cache"] is True
        assert call_kwargs["output_hidden_states"] is True
        assert call_kwargs["return_dict"] is True

    def test_generate_latent_creates_attention_mask_if_none(self):
        """generate_latent should create attention_mask of ones if None provided."""
        from lcn.models.model_wrapper import LCNModelWrapper

        wrapper = LCNModelWrapper(
            model_name="test-model",
            device=torch.device("cpu"),
        )

        batch_size, seq_len, hidden_dim = 1, 4, 64
        num_layers, num_heads, head_dim = 1, 2, 32

        mock_kv_cache = tuple(
            (
                torch.randn(batch_size, num_heads, seq_len, head_dim),
                torch.randn(batch_size, num_heads, seq_len, head_dim),
            )
            for _ in range(num_layers)
        )
        mock_hidden_states = tuple(
            torch.randn(batch_size, seq_len, hidden_dim) for _ in range(num_layers + 1)
        )

        mock_outputs = MagicMock()
        mock_outputs.past_key_values = mock_kv_cache
        mock_outputs.hidden_states = mock_hidden_states

        mock_model = MagicMock()
        mock_model.return_value = mock_outputs
        wrapper.model = mock_model

        input_ids = torch.tensor([[1, 2, 3, 4]])

        wrapper.generate_latent(
            input_ids=input_ids,
            attention_mask=None,  # None provided
            latent_steps=0,
        )

        call_kwargs = mock_model.call_args[1]
        attention_mask = call_kwargs["attention_mask"]
        assert attention_mask.shape == (batch_size, seq_len)
        assert (attention_mask == 1).all()


class TestGenerateLatentWithLatentSteps:
    """Tests for generate_latent with latent_steps > 0."""

    def _create_mock_wrapper_with_model(self, batch_size, hidden_dim, num_layers, num_heads, head_dim):
        """Helper to create a wrapper with mocked model."""
        from lcn.models.model_wrapper import LCNModelWrapper

        wrapper = LCNModelWrapper(
            model_name="test-model",
            device=torch.device("cpu"),
        )

        seq_len = 1  # For latent steps, we process 1 embedding at a time

        def create_outputs(current_seq_len):
            mock_kv_cache = tuple(
                (
                    torch.randn(batch_size, num_heads, current_seq_len, head_dim),
                    torch.randn(batch_size, num_heads, current_seq_len, head_dim),
                )
                for _ in range(num_layers)
            )
            mock_hidden_states = tuple(
                torch.randn(batch_size, seq_len, hidden_dim) for _ in range(num_layers + 1)
            )
            mock_outputs = MagicMock()
            mock_outputs.past_key_values = mock_kv_cache
            mock_outputs.hidden_states = mock_hidden_states
            return mock_outputs

        return wrapper, create_outputs

    def test_generate_latent_performs_latent_steps(self):
        """generate_latent with latent_steps > 0 should call model multiple times."""
        from lcn.models.model_wrapper import LCNModelWrapper

        wrapper = LCNModelWrapper(
            model_name="test-model",
            device=torch.device("cpu"),
        )

        batch_size, initial_seq_len, hidden_dim = 1, 3, 64
        num_layers, num_heads, head_dim = 1, 2, 32
        latent_steps = 2

        call_count = [0]
        accumulated_seq_len = [initial_seq_len]

        def mock_model_call(**kwargs):
            call_count[0] += 1
            # Each call adds 1 to sequence length in KV-cache
            current_seq = accumulated_seq_len[0]
            mock_kv_cache = tuple(
                (
                    torch.randn(batch_size, num_heads, current_seq, head_dim),
                    torch.randn(batch_size, num_heads, current_seq, head_dim),
                )
                for _ in range(num_layers)
            )
            # Output 1 position for latent step
            out_seq = 1 if call_count[0] > 1 else initial_seq_len
            mock_hidden_states = tuple(
                torch.randn(batch_size, out_seq, hidden_dim)
                for _ in range(num_layers + 1)
            )
            accumulated_seq_len[0] += 1  # Next call will have longer cache
            mock_outputs = MagicMock()
            mock_outputs.past_key_values = mock_kv_cache
            mock_outputs.hidden_states = mock_hidden_states
            return mock_outputs

        mock_model = MagicMock(side_effect=mock_model_call)
        wrapper.model = mock_model

        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.tensor([[1, 1, 1]])

        wrapper.generate_latent(
            input_ids=input_ids,
            attention_mask=attention_mask,
            latent_steps=latent_steps,
        )

        # 1 initial call + latent_steps calls
        assert mock_model.call_count == 1 + latent_steps

    def test_generate_latent_uses_inputs_embeds_for_latent_steps(self):
        """Latent steps should use inputs_embeds, not input_ids."""
        from lcn.models.model_wrapper import LCNModelWrapper

        wrapper = LCNModelWrapper(
            model_name="test-model",
            device=torch.device("cpu"),
        )

        batch_size, initial_seq_len, hidden_dim = 1, 3, 64
        num_layers, num_heads, head_dim = 1, 2, 32
        latent_steps = 1

        calls = []

        def mock_model_call(**kwargs):
            calls.append(kwargs)
            current_seq = initial_seq_len + len(calls) - 1
            mock_kv_cache = tuple(
                (
                    torch.randn(batch_size, num_heads, current_seq, head_dim),
                    torch.randn(batch_size, num_heads, current_seq, head_dim),
                )
                for _ in range(num_layers)
            )
            out_seq = 1 if len(calls) > 1 else initial_seq_len
            mock_hidden_states = tuple(
                torch.randn(batch_size, out_seq, hidden_dim)
                for _ in range(num_layers + 1)
            )
            mock_outputs = MagicMock()
            mock_outputs.past_key_values = mock_kv_cache
            mock_outputs.hidden_states = mock_hidden_states
            return mock_outputs

        mock_model = MagicMock(side_effect=mock_model_call)
        wrapper.model = mock_model

        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.tensor([[1, 1, 1]])

        wrapper.generate_latent(
            input_ids=input_ids,
            attention_mask=attention_mask,
            latent_steps=latent_steps,
        )

        # First call should use input_ids
        assert "input_ids" in calls[0]
        assert calls[0].get("inputs_embeds") is None

        # Second call (latent step) should use inputs_embeds
        assert "inputs_embeds" in calls[1]
        assert calls[1]["inputs_embeds"] is not None
        # inputs_embeds should have shape [B, 1, D]
        assert calls[1]["inputs_embeds"].shape == (batch_size, 1, hidden_dim)


class TestLatentRealignment:
    """Tests for latent space realignment methods."""

    def test_build_latent_realign_matrix_requires_embeddings(self):
        """_build_latent_realign_matrix should raise if embeddings not accessible."""
        from lcn.models.model_wrapper import LCNModelWrapper

        wrapper = LCNModelWrapper(
            model_name="test-model",
            device=torch.device("cpu"),
            latent_space_realign=True,
        )

        # Mock model without proper embeddings
        mock_model = MagicMock()
        mock_model.get_input_embeddings.return_value = None

        with pytest.raises(RuntimeError, match="[Ee]mbedding"):
            wrapper._build_latent_realign_matrix(mock_model, torch.device("cpu"))

    def test_build_latent_realign_matrix_returns_matrix_and_norm(self):
        """_build_latent_realign_matrix should return (matrix, target_norm)."""
        from lcn.models.model_wrapper import LCNModelWrapper

        wrapper = LCNModelWrapper(
            model_name="test-model",
            device=torch.device("cpu"),
            latent_space_realign=True,
        )

        vocab_size, hidden_dim = 100, 64

        # Mock model with proper embeddings
        mock_input_embeds = MagicMock()
        mock_input_embeds.weight = torch.randn(vocab_size, hidden_dim)

        mock_output_embeds = MagicMock()
        mock_output_embeds.weight = torch.randn(vocab_size, hidden_dim)

        mock_model = MagicMock()
        mock_model.get_input_embeddings.return_value = mock_input_embeds
        mock_model.get_output_embeddings.return_value = mock_output_embeds

        matrix, target_norm = wrapper._build_latent_realign_matrix(
            mock_model, torch.device("cpu")
        )

        assert isinstance(matrix, torch.Tensor)
        assert matrix.shape == (hidden_dim, hidden_dim)
        assert isinstance(target_norm, torch.Tensor)

    def test_apply_latent_realignment_transforms_hidden_state(self):
        """_apply_latent_realignment should transform hidden state."""
        from lcn.models.model_wrapper import LCNModelWrapper

        wrapper = LCNModelWrapper(
            model_name="test-model",
            device=torch.device("cpu"),
            latent_space_realign=True,
        )

        batch_size, hidden_dim = 2, 64
        vocab_size = 100

        # Set up realignment matrix
        mock_input_embeds = MagicMock()
        mock_input_embeds.weight = torch.randn(vocab_size, hidden_dim)

        mock_output_embeds = MagicMock()
        mock_output_embeds.weight = torch.randn(vocab_size, hidden_dim)

        mock_model = MagicMock()
        mock_model.get_input_embeddings.return_value = mock_input_embeds
        mock_model.get_output_embeddings.return_value = mock_output_embeds

        hidden = torch.randn(batch_size, hidden_dim)

        aligned = wrapper._apply_latent_realignment(hidden, mock_model)

        assert aligned.shape == hidden.shape
        # Result should be different from input (unless identity, which is unlikely)
        # Just check it runs without error

    def test_latent_realignment_skipped_when_disabled(self):
        """When latent_space_realign=False, realignment should be skipped."""
        from lcn.models.model_wrapper import LCNModelWrapper

        wrapper = LCNModelWrapper(
            model_name="test-model",
            device=torch.device("cpu"),
            latent_space_realign=False,  # Disabled
        )

        batch_size, initial_seq_len, hidden_dim = 1, 3, 64
        num_layers, num_heads, head_dim = 1, 2, 32
        latent_steps = 1

        calls = []

        def mock_model_call(**kwargs):
            calls.append(kwargs)
            current_seq = initial_seq_len + len(calls) - 1
            mock_kv_cache = tuple(
                (
                    torch.randn(batch_size, num_heads, current_seq, head_dim),
                    torch.randn(batch_size, num_heads, current_seq, head_dim),
                )
                for _ in range(num_layers)
            )
            out_seq = 1 if len(calls) > 1 else initial_seq_len
            mock_hidden_states = tuple(
                torch.randn(batch_size, out_seq, hidden_dim)
                for _ in range(num_layers + 1)
            )
            mock_outputs = MagicMock()
            mock_outputs.past_key_values = mock_kv_cache
            mock_outputs.hidden_states = mock_hidden_states
            return mock_outputs

        mock_model = MagicMock(side_effect=mock_model_call)
        wrapper.model = mock_model

        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.tensor([[1, 1, 1]])

        # Should work without calling _build_latent_realign_matrix
        wrapper.generate_latent(
            input_ids=input_ids,
            attention_mask=attention_mask,
            latent_steps=latent_steps,
        )

        # Latent step should use hidden state directly
        assert len(calls) == 2


class TestGenerateLatentWithPastKeyValues:
    """Tests for generate_latent with past_key_values."""

    def test_generate_latent_extends_attention_mask_for_past(self):
        """generate_latent should prepend past length to attention_mask."""
        from lcn.models.model_wrapper import LCNModelWrapper

        wrapper = LCNModelWrapper(
            model_name="test-model",
            device=torch.device("cpu"),
        )

        batch_size, input_seq_len, hidden_dim = 1, 3, 64
        num_layers, num_heads, head_dim = 1, 2, 32
        past_seq_len = 5

        # Create past_key_values
        past_kv = tuple(
            (
                torch.randn(batch_size, num_heads, past_seq_len, head_dim),
                torch.randn(batch_size, num_heads, past_seq_len, head_dim),
            )
            for _ in range(num_layers)
        )

        def mock_model_call(**kwargs):
            total_seq = past_seq_len + input_seq_len
            mock_kv_cache = tuple(
                (
                    torch.randn(batch_size, num_heads, total_seq, head_dim),
                    torch.randn(batch_size, num_heads, total_seq, head_dim),
                )
                for _ in range(num_layers)
            )
            mock_hidden_states = tuple(
                torch.randn(batch_size, input_seq_len, hidden_dim)
                for _ in range(num_layers + 1)
            )
            mock_outputs = MagicMock()
            mock_outputs.past_key_values = mock_kv_cache
            mock_outputs.hidden_states = mock_hidden_states
            return mock_outputs

        mock_model = MagicMock(side_effect=mock_model_call)
        wrapper.model = mock_model

        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.tensor([[1, 1, 1]])

        wrapper.generate_latent(
            input_ids=input_ids,
            attention_mask=attention_mask,
            latent_steps=0,
            past_key_values=past_kv,
        )

        # Verify attention_mask was extended
        call_kwargs = mock_model.call_args[1]
        actual_mask = call_kwargs["attention_mask"]
        expected_length = past_seq_len + input_seq_len
        assert actual_mask.shape == (batch_size, expected_length)


class TestGenerateText:
    """Tests for LCNModelWrapper.generate_text method."""

    def test_generate_text_raises_error_when_model_not_loaded(self):
        """generate_text should raise RuntimeError if model is None."""
        from lcn.models.model_wrapper import LCNModelWrapper

        wrapper = LCNModelWrapper(
            model_name="test-model",
            device=torch.device("cpu"),
        )
        # model is None by default

        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.tensor([[1, 1, 1]])

        with pytest.raises(RuntimeError, match="[Mm]odel.*not.*loaded"):
            wrapper.generate_text(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

    def test_generate_text_raises_error_for_non_2d_input(self):
        """generate_text should raise ValueError for non-2D input_ids."""
        from lcn.models.model_wrapper import LCNModelWrapper

        wrapper = LCNModelWrapper(
            model_name="test-model",
            device=torch.device("cpu"),
        )
        wrapper.model = MagicMock()
        wrapper.tokenizer = MagicMock()

        # 1D input instead of 2D
        input_ids = torch.tensor([1, 2, 3])
        attention_mask = torch.tensor([1, 1, 1])

        with pytest.raises(ValueError, match="2D"):
            wrapper.generate_text(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

    def test_generate_text_returns_list_of_strings(self):
        """generate_text should return List[str]."""
        from lcn.models.model_wrapper import LCNModelWrapper

        wrapper = LCNModelWrapper(
            model_name="test-model",
            device=torch.device("cpu"),
        )

        batch_size, seq_len = 2, 5

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.decode = MagicMock(side_effect=["Generated text 1", "Generated text 2"])
        wrapper.tokenizer = mock_tokenizer

        # Mock model.generate output
        mock_outputs = MagicMock()
        # generate returns sequences including prompt + generated
        mock_outputs.sequences = torch.tensor([
            [1, 2, 3, 4, 5, 10, 11, 12],  # prompt + generated
            [1, 2, 3, 4, 5, 20, 21, 22],
        ])

        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value=mock_outputs)
        wrapper.model = mock_model

        input_ids = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        result = wrapper.generate_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        assert isinstance(result, list)
        assert len(result) == batch_size
        assert all(isinstance(s, str) for s in result)

    def test_generate_text_calls_model_generate_with_correct_params(self):
        """generate_text should call model.generate with expected parameters."""
        from lcn.models.model_wrapper import LCNModelWrapper

        wrapper = LCNModelWrapper(
            model_name="test-model",
            device=torch.device("cpu"),
        )

        batch_size, seq_len = 1, 3

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.decode = MagicMock(return_value="Generated")
        wrapper.tokenizer = mock_tokenizer

        mock_outputs = MagicMock()
        mock_outputs.sequences = torch.tensor([[1, 2, 3, 10, 11]])

        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value=mock_outputs)
        wrapper.model = mock_model

        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.tensor([[1, 1, 1]])
        max_new_tokens = 128

        wrapper.generate_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )

        mock_model.generate.assert_called_once()
        call_kwargs = mock_model.generate.call_args[1]

        assert call_kwargs["max_new_tokens"] == max_new_tokens
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["top_p"] == 0.95
        assert call_kwargs["do_sample"] is True
        assert call_kwargs["pad_token_id"] == mock_tokenizer.pad_token_id
        assert call_kwargs["return_dict_in_generate"] is True

    def test_generate_text_decodes_with_skip_special_tokens(self):
        """generate_text should decode with skip_special_tokens=True."""
        from lcn.models.model_wrapper import LCNModelWrapper

        wrapper = LCNModelWrapper(
            model_name="test-model",
            device=torch.device("cpu"),
        )

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.decode = MagicMock(return_value="  Generated text  ")
        wrapper.tokenizer = mock_tokenizer

        mock_outputs = MagicMock()
        mock_outputs.sequences = torch.tensor([[1, 2, 3, 10, 11]])

        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value=mock_outputs)
        wrapper.model = mock_model

        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.tensor([[1, 1, 1]])

        result = wrapper.generate_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        mock_tokenizer.decode.assert_called_once()
        call_args = mock_tokenizer.decode.call_args
        assert call_args[1]["skip_special_tokens"] is True

        # Result should be stripped
        assert result[0] == "Generated text"

    def test_generate_text_strips_prompt_from_output(self):
        """generate_text should strip prompt tokens from generated sequences."""
        from lcn.models.model_wrapper import LCNModelWrapper

        wrapper = LCNModelWrapper(
            model_name="test-model",
            device=torch.device("cpu"),
        )

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.decode = MagicMock(return_value="Generated")
        wrapper.tokenizer = mock_tokenizer

        # Prompt length is 3, generated is 2 tokens
        mock_outputs = MagicMock()
        mock_outputs.sequences = torch.tensor([[1, 2, 3, 10, 11]])  # 3 prompt + 2 generated

        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value=mock_outputs)
        wrapper.model = mock_model

        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.tensor([[1, 1, 1]])

        wrapper.generate_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Verify decode was called with only the generated tokens (not prompt)
        call_args = mock_tokenizer.decode.call_args
        decoded_ids = call_args[0][0]
        # Should be [10, 11] (the generated part), not [1, 2, 3, 10, 11]
        assert decoded_ids.tolist() == [10, 11]

    def test_generate_text_default_max_new_tokens(self):
        """generate_text should default to max_new_tokens=256."""
        from lcn.models.model_wrapper import LCNModelWrapper

        wrapper = LCNModelWrapper(
            model_name="test-model",
            device=torch.device("cpu"),
        )

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.decode = MagicMock(return_value="Generated")
        wrapper.tokenizer = mock_tokenizer

        mock_outputs = MagicMock()
        mock_outputs.sequences = torch.tensor([[1, 2, 3, 10]])

        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value=mock_outputs)
        wrapper.model = mock_model

        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.tensor([[1, 1, 1]])

        wrapper.generate_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # Not passing max_new_tokens - should default to 256
        )

        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["max_new_tokens"] == 256


class TestGenerateTextWithPastKeyValues:
    """Tests for generate_text with past_key_values."""

    def test_generate_text_extends_attention_mask_for_past(self):
        """generate_text should prepend past length to attention_mask."""
        from lcn.models.model_wrapper import LCNModelWrapper

        wrapper = LCNModelWrapper(
            model_name="test-model",
            device=torch.device("cpu"),
        )

        batch_size, input_seq_len = 1, 3
        num_layers, num_heads, head_dim = 1, 2, 32
        past_seq_len = 5

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.decode = MagicMock(return_value="Generated")
        wrapper.tokenizer = mock_tokenizer

        # Create past_key_values
        past_kv = tuple(
            (
                torch.randn(batch_size, num_heads, past_seq_len, head_dim),
                torch.randn(batch_size, num_heads, past_seq_len, head_dim),
            )
            for _ in range(num_layers)
        )

        mock_outputs = MagicMock()
        mock_outputs.sequences = torch.tensor([[1, 2, 3, 10, 11]])

        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value=mock_outputs)
        wrapper.model = mock_model

        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.tensor([[1, 1, 1]])

        wrapper.generate_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_kv,
        )

        call_kwargs = mock_model.generate.call_args[1]
        actual_mask = call_kwargs["attention_mask"]
        expected_length = past_seq_len + input_seq_len
        assert actual_mask.shape == (batch_size, expected_length)

    def test_generate_text_passes_cache_position_when_past_provided(self):
        """generate_text should pass cache_position when past_key_values is provided."""
        from lcn.models.model_wrapper import LCNModelWrapper

        wrapper = LCNModelWrapper(
            model_name="test-model",
            device=torch.device("cpu"),
        )

        batch_size, input_seq_len = 1, 3
        num_layers, num_heads, head_dim = 1, 2, 32
        past_seq_len = 5

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.decode = MagicMock(return_value="Generated")
        wrapper.tokenizer = mock_tokenizer

        past_kv = tuple(
            (
                torch.randn(batch_size, num_heads, past_seq_len, head_dim),
                torch.randn(batch_size, num_heads, past_seq_len, head_dim),
            )
            for _ in range(num_layers)
        )

        mock_outputs = MagicMock()
        mock_outputs.sequences = torch.tensor([[1, 2, 3, 10, 11]])

        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value=mock_outputs)
        wrapper.model = mock_model

        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.tensor([[1, 1, 1]])

        wrapper.generate_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_kv,
        )

        call_kwargs = mock_model.generate.call_args[1]
        cache_position = call_kwargs["cache_position"]

        assert cache_position is not None
        # cache_position should be [past_seq_len, past_seq_len+1, past_seq_len+2]
        expected = torch.arange(past_seq_len, past_seq_len + input_seq_len)
        assert torch.equal(cache_position, expected)

    def test_generate_text_passes_past_key_values_to_generate(self):
        """generate_text should pass past_key_values to model.generate."""
        from lcn.models.model_wrapper import LCNModelWrapper

        wrapper = LCNModelWrapper(
            model_name="test-model",
            device=torch.device("cpu"),
        )

        batch_size, input_seq_len = 1, 3
        num_layers, num_heads, head_dim = 1, 2, 32
        past_seq_len = 5

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.decode = MagicMock(return_value="Generated")
        wrapper.tokenizer = mock_tokenizer

        past_kv = tuple(
            (
                torch.randn(batch_size, num_heads, past_seq_len, head_dim),
                torch.randn(batch_size, num_heads, past_seq_len, head_dim),
            )
            for _ in range(num_layers)
        )

        mock_outputs = MagicMock()
        mock_outputs.sequences = torch.tensor([[1, 2, 3, 10, 11]])

        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value=mock_outputs)
        wrapper.model = mock_model

        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.tensor([[1, 1, 1]])

        wrapper.generate_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_kv,
        )

        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["past_key_values"] is past_kv

    def test_generate_text_no_cache_position_when_no_past(self):
        """generate_text should not pass cache_position when no past_key_values."""
        from lcn.models.model_wrapper import LCNModelWrapper

        wrapper = LCNModelWrapper(
            model_name="test-model",
            device=torch.device("cpu"),
        )

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.decode = MagicMock(return_value="Generated")
        wrapper.tokenizer = mock_tokenizer

        mock_outputs = MagicMock()
        mock_outputs.sequences = torch.tensor([[1, 2, 3, 10]])

        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value=mock_outputs)
        wrapper.model = mock_model

        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.tensor([[1, 1, 1]])

        wrapper.generate_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
        )

        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs.get("cache_position") is None
