"""Configuration schema and YAML loading helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    backend: str = "mock"
    model_name: str = ""
    hidden_dim: int | None = 32
    device: str = "cpu"
    kv_num_layers: int = 2
    kv_seq_len: int = 4
    torch_dtype: str | None = None
    per_agent_gpu: bool = False


@dataclass
class ProtocolConfig:
    num_rounds: int = 3
    latent_steps: int = 2
    num_groups: int = 2
    agents_per_group: int = 2
    temperature: float = 1.0


@dataclass
class ExperimentConfig:
    environment: str = "hidden_profile"
    seed: int = 0
    model: ModelConfig = field(default_factory=ModelConfig)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment config from YAML."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    model_raw = raw.get("model", {})
    protocol_raw = raw.get("protocol", {})
    return ExperimentConfig(
        environment=raw.get("environment", "hidden_profile"),
        seed=int(raw.get("seed", 0)),
        model=ModelConfig(**model_raw),
        protocol=ProtocolConfig(**protocol_raw),
    )


__all__ = [
    "ExperimentConfig",
    "ModelConfig",
    "ProtocolConfig",
    "load_config",
]
