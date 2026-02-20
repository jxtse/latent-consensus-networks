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
