"""Model wrappers for mock and Hugging Face backed LCN runs."""

from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from lcn.core.kv_cache import KVCache
from lcn.models.latent_ops import state_to_kv_cache

MIN_QWEN35_TRANSFORMERS_VERSION = "5.2.0"
DEFAULT_QWEN35_4B_MODEL_NAME = "Qwen/Qwen3.5-4B"


class BaseModelWrapper(ABC):
    """Abstract interface consumed by the consensus protocol."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        kv_num_layers: int = 2,
        kv_seq_len: int = 4,
        device: str = "cpu",
    ):
        self.hidden_dim = hidden_dim
        self.kv_num_layers = kv_num_layers
        self.kv_seq_len = kv_seq_len
        self.device = _resolve_runtime_device(device)

    @abstractmethod
    def initialize_state(
        self,
        prompt: str,
        *,
        persona: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        """Encode the agent-specific prompt into an initial latent state."""

    @abstractmethod
    def latent_step(
        self,
        agent_state: torch.Tensor,
        fused_state: torch.Tensor,
        *,
        prompt: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        step_idx: int = 0,
    ) -> torch.Tensor:
        """Run one latent update step."""

    @abstractmethod
    def score_options(
        self,
        agent_state: torch.Tensor,
        options: Sequence[str],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        """Score candidate options for final decision making."""

    def build_kv_cache(self, state: torch.Tensor) -> KVCache:
        """Create a synthetic KV representation that can participate in LCN."""
        return state_to_kv_cache(
            state.to(self.device),
            num_layers=self.kv_num_layers,
            seq_len=self.kv_seq_len,
        )

    def choose_option(
        self,
        agent_state: torch.Tensor,
        options: Sequence[str],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """Return the best-scoring option label."""
        scores = self.score_options(agent_state, options, metadata=metadata)
        best_idx = int(scores.argmax(dim=-1).item())
        return options[best_idx]


def _extract_hidden_dim_from_config(config: Any) -> int:
    """Read the language hidden size from a Hugging Face config."""
    hidden_dim = getattr(config, "hidden_size", None)
    if hidden_dim is not None:
        return int(hidden_dim)

    text_config = getattr(config, "text_config", None)
    if (
        text_config is not None
        and getattr(text_config, "hidden_size", None) is not None
    ):
        return int(text_config.hidden_size)

    raise ValueError("Unable to infer hidden_size from model config")


def _resolve_torch_dtype(
    dtype: Optional[str | torch.dtype],
) -> Optional[str | torch.dtype]:
    """Normalize string dtype aliases accepted by config files."""
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype

    normalized = dtype.lower()
    if normalized == "auto":
        return "auto"

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in dtype_map:
        raise ValueError(f"Unsupported torch dtype: {dtype}")
    return dtype_map[normalized]


def _build_dtype_kwargs(dtype: Optional[str | torch.dtype]) -> dict[str, str | torch.dtype]:
    """Build the model-loading dtype kwargs expected by recent transformers."""
    resolved_dtype = _resolve_torch_dtype(dtype)
    if resolved_dtype is None:
        return {}
    return {"dtype": resolved_dtype}


def _resolve_runtime_device(device: str) -> torch.device:
    """Map requested device strings to an available runtime device."""
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available in this process. "
            "Check CUDA_VISIBLE_DEVICES, NVIDIA runtime access, and driver visibility."
        )
    return requested


def _infer_model_dtype(model: nn.Module) -> torch.dtype:
    """Infer a module parameter dtype, defaulting to float32 when empty."""
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float32


def _parse_version(version: str) -> tuple[int, ...]:
    """Extract numeric version components from package version strings."""
    return tuple(int(component) for component in re.findall(r"\d+", version))


def _ensure_minimum_version(current: str, minimum: str, package_name: str) -> None:
    """Raise a helpful import error when a package version is too old."""
    if _parse_version(current) < _parse_version(minimum):
        raise ImportError(
            f"{package_name}>={minimum} is required, but {current} is installed"
        )


def _load_fast_tokenizer(model_name: str) -> Any:
    """Load a text tokenizer without requiring vision or protobuf extras."""
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception:
        from huggingface_hub import hf_hub_download
        from transformers import PreTrainedTokenizerFast

        tokenizer_config_path = hf_hub_download(model_name, "tokenizer_config.json")
        tokenizer_json_path = hf_hub_download(model_name, "tokenizer.json")

        tokenizer_config = json.loads(
            Path(tokenizer_config_path).read_text(encoding="utf-8")
        )
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_json_path,
            bos_token=tokenizer_config.get("bos_token"),
            eos_token=tokenizer_config.get("eos_token"),
            unk_token=tokenizer_config.get("unk_token"),
            pad_token=tokenizer_config.get("pad_token"),
        )
        tokenizer.chat_template = tokenizer_config.get("chat_template")
        return tokenizer


class MockModelWrapper(BaseModelWrapper):
    """Deterministic latent model used for tests and dry runs."""

    def __init__(
        self,
        hidden_dim: int = 32,
        *,
        kv_num_layers: int = 2,
        kv_seq_len: int = 4,
        device: str = "cpu",
    ):
        super().__init__(
            hidden_dim,
            kv_num_layers=kv_num_layers,
            kv_seq_len=kv_seq_len,
            device=device,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def initialize_state(
        self,
        prompt: str,
        *,
        persona: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        chunks = [prompt]
        if persona:
            chunks.append(persona)
        if metadata and metadata.get("observation"):
            chunks.append(str(metadata["observation"]))
        return self._embed_text("\n".join(chunks)).to(self.device)

    def latent_step(
        self,
        agent_state: torch.Tensor,
        fused_state: torch.Tensor,
        *,
        prompt: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        step_idx: int = 0,
    ) -> torch.Tensor:
        prompt_state = self._embed_text(
            prompt or "",
            batch_size=agent_state.shape[0],
        ).to(self.device)
        updated = (
            agent_state.to(self.device)
            + 0.6 * fused_state.to(self.device)
            + 0.2 * prompt_state
        )
        return self.norm(updated)

    def score_options(
        self,
        agent_state: torch.Tensor,
        options: Sequence[str],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        if not options:
            raise ValueError("options must not be empty")

        state = agent_state.to(self.device)
        option_hints = metadata.get("option_hints", {}) if metadata else {}
        option_embeddings = torch.cat(
            [
                self._embed_text(
                    f"{option} {option_hints.get(option, '')}".strip()
                ).to(self.device)
                for option in options
            ],
            dim=0,
        )
        return torch.matmul(state, option_embeddings.transpose(0, 1))

    def _embed_text(self, text: str, batch_size: int = 1) -> torch.Tensor:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = torch.tensor(list(digest), dtype=torch.float32)
        repeat_count = (self.hidden_dim + raw.numel() - 1) // raw.numel()
        expanded = raw.repeat(repeat_count)[: self.hidden_dim]
        centered = (expanded - expanded.mean()) / (expanded.std(unbiased=False) + 1e-6)
        return centered.unsqueeze(0).repeat(batch_size, 1)


class HuggingFaceCausalLMWrapper(BaseModelWrapper):
    """Lightweight Hugging Face integration for real-model LCN runs."""

    def __init__(
        self,
        model_name: str,
        *,
        hidden_dim: Optional[int] = None,
        kv_num_layers: int = 2,
        kv_seq_len: int = 4,
        device: str = "cpu",
        torch_dtype: Optional[str | torch.dtype] = None,
    ):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required to use HuggingFaceCausalLMWrapper"
            ) from exc

        runtime_device = _resolve_runtime_device(device)
        model_kwargs = _build_dtype_kwargs(torch_dtype)
        model_kwargs["low_cpu_mem_usage"] = True

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.to(runtime_device)
        self.model.eval()
        aux_dtype = _infer_model_dtype(self.model)
        if (
            getattr(self.tokenizer, "pad_token_id", None) is None
            and getattr(self.tokenizer, "eos_token", None)
        ):
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_hidden_dim = _extract_hidden_dim_from_config(self.model.config)
        super().__init__(
            hidden_dim or model_hidden_dim,
            kv_num_layers=kv_num_layers,
            kv_seq_len=kv_seq_len,
            device=str(runtime_device),
        )
        self.model_name = model_name
        if self.hidden_dim != model_hidden_dim:
            self.state_projector = nn.Linear(
                model_hidden_dim,
                self.hidden_dim,
                bias=False,
            ).to(device=self.device, dtype=aux_dtype)
        else:
            self.state_projector = nn.Identity()

        self.update_gate = nn.Linear(self.hidden_dim * 2, self.hidden_dim).to(
            device=self.device,
            dtype=aux_dtype,
        )
        self.norm = nn.LayerNorm(self.hidden_dim).to(device=self.device, dtype=aux_dtype)

    @torch.inference_mode()
    def initialize_state(
        self,
        prompt: str,
        *,
        persona: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        full_prompt = prompt
        if persona:
            full_prompt = f"Persona: {persona}\n{full_prompt}"
        if metadata and metadata.get("observation"):
            full_prompt = f"{full_prompt}\nObservation: {metadata['observation']}"

        encoded = self.tokenizer(full_prompt, return_tensors="pt", truncation=True)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        outputs = self.model(**encoded, output_hidden_states=True, use_cache=False)
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        return self.state_projector(last_hidden)

    @torch.inference_mode()
    def latent_step(
        self,
        agent_state: torch.Tensor,
        fused_state: torch.Tensor,
        *,
        prompt: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        step_idx: int = 0,
    ) -> torch.Tensor:
        combined = torch.cat(
            [agent_state.to(self.device), fused_state.to(self.device)],
            dim=-1,
        )
        gated = torch.tanh(self.update_gate(combined))
        return self.norm(agent_state.to(self.device) + gated)

    @torch.inference_mode()
    def score_options(
        self,
        agent_state: torch.Tensor,
        options: Sequence[str],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        if not options:
            raise ValueError("options must not be empty")

        option_hints = metadata.get("option_hints", {}) if metadata else {}
        option_states = []
        for option in options:
            hint = option_hints.get(option, "")
            option_states.append(self.initialize_state(f"{option} {hint}".strip()))

        option_matrix = torch.cat(option_states, dim=0)
        return torch.matmul(agent_state.to(self.device), option_matrix.transpose(0, 1))


class Qwen35ModelWrapper(BaseModelWrapper):
    """Qwen 3.5 integration using a text-only tokenizer/model path."""

    def __init__(
        self,
        model_name: str = DEFAULT_QWEN35_4B_MODEL_NAME,
        *,
        hidden_dim: Optional[int] = None,
        kv_num_layers: int = 2,
        kv_seq_len: int = 4,
        device: str = "cpu",
        torch_dtype: Optional[str | torch.dtype] = "auto",
    ):
        try:
            import transformers
            from transformers import AutoModelForImageTextToText
        except ImportError as exc:
            raise ImportError(
                "transformers is required to use Qwen35ModelWrapper"
            ) from exc

        _ensure_minimum_version(
            getattr(transformers, "__version__", "0.0.0"),
            MIN_QWEN35_TRANSFORMERS_VERSION,
            "transformers",
        )

        runtime_device = _resolve_runtime_device(device)
        model_kwargs = _build_dtype_kwargs(torch_dtype)
        model_kwargs["low_cpu_mem_usage"] = True

        self.tokenizer = _load_fast_tokenizer(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            **model_kwargs,
        )
        self.model.to(runtime_device)
        self.model.eval()
        aux_dtype = _infer_model_dtype(self.model)

        if (
            getattr(self.tokenizer, "pad_token_id", None) is None
            and getattr(self.tokenizer, "eos_token", None)
        ):
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_hidden_dim = _extract_hidden_dim_from_config(self.model.config)
        super().__init__(
            hidden_dim or model_hidden_dim,
            kv_num_layers=kv_num_layers,
            kv_seq_len=kv_seq_len,
            device=str(runtime_device),
        )
        self.model_name = model_name
        if self.hidden_dim != model_hidden_dim:
            self.state_projector = nn.Linear(
                model_hidden_dim,
                self.hidden_dim,
                bias=False,
            ).to(device=self.device, dtype=aux_dtype)
        else:
            self.state_projector = nn.Identity()

        self.update_gate = nn.Linear(self.hidden_dim * 2, self.hidden_dim).to(
            device=self.device,
            dtype=aux_dtype,
        )
        self.norm = nn.LayerNorm(self.hidden_dim).to(device=self.device, dtype=aux_dtype)

    @torch.inference_mode()
    def initialize_state(
        self,
        prompt: str,
        *,
        persona: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        messages = self._build_messages(prompt, persona=persona, metadata=metadata)
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        encoded = self.tokenizer(
            encoded,
            return_tensors="pt",
            truncation=True,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        outputs = self.model(**encoded, output_hidden_states=True, use_cache=False)
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        return self.state_projector(last_hidden)

    @torch.inference_mode()
    def latent_step(
        self,
        agent_state: torch.Tensor,
        fused_state: torch.Tensor,
        *,
        prompt: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        step_idx: int = 0,
    ) -> torch.Tensor:
        combined = torch.cat([agent_state.to(self.device), fused_state.to(self.device)], dim=-1)
        gated = torch.tanh(self.update_gate(combined))
        return self.norm(agent_state.to(self.device) + gated)

    @torch.inference_mode()
    def score_options(
        self,
        agent_state: torch.Tensor,
        options: Sequence[str],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        if not options:
            raise ValueError("options must not be empty")

        option_hints = metadata.get("option_hints", {}) if metadata else {}
        option_states = []
        for option in options:
            hint = option_hints.get(option, "")
            option_states.append(self.initialize_state(f"{option} {hint}".strip()))

        option_matrix = torch.cat(option_states, dim=0)
        return torch.matmul(agent_state.to(self.device), option_matrix.transpose(0, 1))

    def _build_messages(
        self,
        prompt: str,
        *,
        persona: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if persona:
            messages.append(
                {
                    "role": "system",
                    "content": f"Adopt this persona while reasoning: {persona}",
                }
            )

        user_chunks = [prompt]
        if metadata and metadata.get("observation"):
            user_chunks.append(f"Observation: {metadata['observation']}")
        messages.append({"role": "user", "content": "\n".join(user_chunks)})
        return messages
