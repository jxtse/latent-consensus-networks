import sys
import types

import pytest
import torch

from lcn.models import MockModelWrapper, Qwen35ModelWrapper
from lcn.models.model_wrapper import _load_fast_tokenizer


class TestMockModelWrapper:
    """Tests for the deterministic mock model wrapper."""

    def test_initialize_state_is_deterministic(self):
        model = MockModelWrapper(hidden_dim=16)

        state_a = model.initialize_state("prompt", persona="analyst")
        state_b = model.initialize_state("prompt", persona="analyst")

        assert state_a.shape == (1, 16)
        assert torch.allclose(state_a, state_b)

    def test_build_kv_cache_matches_requested_shape(self):
        model = MockModelWrapper(hidden_dim=16, kv_num_layers=3, kv_seq_len=5)
        state = model.initialize_state("prompt")

        kv_cache = model.build_kv_cache(state)

        assert len(kv_cache) == 3
        assert kv_cache[0][0].shape == (1, 1, 5, 16)

    def test_choose_option_returns_best_scoring_label(self):
        model = MockModelWrapper(hidden_dim=16)
        state = model.initialize_state("prefer candidate c")

        choice = model.choose_option(
            state,
            ["A", "B", "C"],
            metadata={
                "option_hints": {
                    "A": "negative evidence",
                    "B": "mixed evidence",
                    "C": "strong hidden strengths",
                }
            },
        )

        assert choice in {"A", "B", "C"}


class TestQwen35ModelWrapper:
    def test_initialize_state_uses_qwen_chat_template(self, monkeypatch):
        fake_transformers = _build_fake_qwen_transformers(
            version="5.2.1",
            hidden_size=12,
        )
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

        model = Qwen35ModelWrapper(
            model_name="Qwen/Qwen3.5-4B",
            hidden_dim=None,
            torch_dtype="bfloat16",
        )
        state = model.initialize_state(
            "Assess the hidden profile",
            persona="careful analyst",
            metadata={"observation": "Private clue"},
        )

        assert model.hidden_dim == 12
        assert state.shape == (1, 12)
        assert fake_transformers.AutoTokenizer.last_messages == [
            {
                "role": "system",
                "content": "Adopt this persona while reasoning: careful analyst",
            },
            {
                "role": "user",
                "content": "Assess the hidden profile\nObservation: Private clue",
            },
        ]
        assert fake_transformers.AutoModelForImageTextToText.last_from_pretrained == (
            "Qwen/Qwen3.5-4B",
            {"dtype": torch.bfloat16},
        )

    def test_rejects_old_transformers_versions(self, monkeypatch):
        fake_transformers = _build_fake_qwen_transformers(
            version="5.1.9",
            hidden_size=8,
        )
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

        with pytest.raises(ImportError, match="transformers>=5.2.0"):
            Qwen35ModelWrapper()

    def test_load_fast_tokenizer_falls_back_without_auto_tokenizer(self, monkeypatch):
        class BrokenAutoTokenizer:
            @classmethod
            def from_pretrained(cls, model_name: str, use_fast: bool):
                raise ImportError("protobuf missing")

        class FakePreTrainedTokenizerFast:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.chat_template = None

        fake_transformers = types.SimpleNamespace(
            AutoTokenizer=BrokenAutoTokenizer,
            PreTrainedTokenizerFast=FakePreTrainedTokenizerFast,
        )
        fake_hf_hub = types.SimpleNamespace(
            hf_hub_download=lambda model_name, filename: f"/tmp/{filename}"
        )

        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_hub)
        monkeypatch.setattr(
            "lcn.models.model_wrapper.Path.read_text",
            lambda self, encoding="utf-8": (
                '{"bos_token":"<bos>","eos_token":"<eos>","unk_token":"<unk>",'
                '"pad_token":"<pad>","chat_template":"template"}'
            ),
        )

        tokenizer = _load_fast_tokenizer("Qwen/Qwen3.5-4B")

        assert tokenizer.kwargs["tokenizer_file"] == "/tmp/tokenizer.json"
        assert tokenizer.chat_template == "template"


def _build_fake_qwen_transformers(
    version: str,
    hidden_size: int,
) -> types.SimpleNamespace:
    class FakeTokenizer:
        last_messages = None

        def __init__(self):
            self.pad_token_id = None
            self.eos_token = "<eos>"
            self.pad_token = None

        @classmethod
        def from_pretrained(cls, model_name: str, use_fast: bool = True):
            assert use_fast is True
            instance = cls()
            instance.model_name = model_name
            return instance

        def apply_chat_template(
            self,
            messages,
            *,
            tokenize: bool,
            add_generation_prompt: bool,
        ):
            type(self).last_messages = messages
            assert tokenize is False
            assert add_generation_prompt is False
            return "<|im_start|>user\nstub<|im_end|>\n"

        def __call__(self, text, *, return_tensors: str, truncation: bool):
            assert text == "<|im_start|>user\nstub<|im_end|>\n"
            assert return_tensors == "pt"
            assert truncation is True
            return {
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }

    class FakeImageTextModel:
        last_from_pretrained = None

        def __init__(self):
            self.config = types.SimpleNamespace(
                text_config=types.SimpleNamespace(hidden_size=hidden_size)
            )

        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            cls.last_from_pretrained = (model_name, kwargs)
            return cls()

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.is_eval = True
            return self

        def __call__(self, **kwargs):
            assert kwargs["output_hidden_states"] is True
            assert kwargs["use_cache"] is False
            hidden = torch.arange(
                3 * hidden_size,
                dtype=torch.float32,
            ).reshape(1, 3, hidden_size)
            return types.SimpleNamespace(hidden_states=[hidden, hidden + 1.0])

    return types.SimpleNamespace(
        __version__=version,
        AutoTokenizer=FakeTokenizer,
        AutoModelForImageTextToText=FakeImageTextModel,
    )
