"""Generic experiment runner for LCN environments."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import torch

from lcn.configs import load_config
from lcn.core.attention import CrossLevelAttention
from lcn.core.consensus import ConsensusProtocol
from lcn.core.kv_cache import HierarchicalKVCache
from lcn.environments import (
    AschConformityEnvironment,
    HiddenProfileEnvironment,
    WisdomOfCrowdsEnvironment,
)
from lcn.models import (
    HuggingFaceCausalLMWrapper,
    MockModelWrapper,
    Qwen35ModelWrapper,
)


def build_environment(name: str, hidden_dim: int, num_groups: int):
    if name == "hidden_profile":
        return HiddenProfileEnvironment(hidden_dim=hidden_dim, num_groups=num_groups)
    if name == "asch_conformity":
        return AschConformityEnvironment(hidden_dim=hidden_dim)
    if name == "wisdom_of_crowds":
        return WisdomOfCrowdsEnvironment(hidden_dim=hidden_dim, num_groups=num_groups)
    raise ValueError(f"Unsupported environment: {name}")


def _instantiate_model(config, *, device_override: str | None = None):
    device = device_override or config.device
    if config.backend == "mock":
        return MockModelWrapper(
            hidden_dim=config.hidden_dim or 32,
            kv_num_layers=config.kv_num_layers,
            kv_seq_len=config.kv_seq_len,
            device=device,
        )
    if config.backend == "huggingface":
        if not config.model_name:
            raise ValueError("model.model_name must be set for huggingface backend")
        return HuggingFaceCausalLMWrapper(
            model_name=config.model_name,
            hidden_dim=config.hidden_dim,
            kv_num_layers=config.kv_num_layers,
            kv_seq_len=config.kv_seq_len,
            device=device,
            torch_dtype=config.torch_dtype,
        )
    if config.backend == "qwen3.5":
        return Qwen35ModelWrapper(
            model_name=config.model_name or "Qwen/Qwen3.5-4B",
            hidden_dim=config.hidden_dim,
            kv_num_layers=config.kv_num_layers,
            kv_seq_len=config.kv_seq_len,
            device=device,
            torch_dtype=config.torch_dtype,
        )
    raise ValueError(f"Unsupported backend: {config.backend}")


def build_model(config):
    return _instantiate_model(config)


def build_model_wrappers(config, agents=None):
    if not config.per_agent_gpu:
        return build_model(config)

    if agents is None:
        raise ValueError("agents must be provided when model.per_agent_gpu is enabled")
    if not str(config.device).startswith("cuda"):
        raise ValueError("model.per_agent_gpu requires a CUDA device config")

    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        raise RuntimeError("model.per_agent_gpu requires at least one visible CUDA device")
    if len(agents) > gpu_count:
        raise ValueError(
            f"model.per_agent_gpu requires one visible GPU per agent, got {len(agents)} agents and {gpu_count} GPUs"
        )

    wrappers = {}
    for device_idx, agent in enumerate(agents):
        wrappers[agent.agent_id] = _instantiate_model(
            config,
            device_override=f"cuda:{device_idx}",
        )
    return wrappers


def infer_hidden_dim(model_or_wrappers) -> int:
    if isinstance(model_or_wrappers, Mapping):
        first_wrapper = next(iter(model_or_wrappers.values()), None)
        if first_wrapper is None:
            raise ValueError("model wrapper mapping must not be empty")
        return first_wrapper.hidden_dim
    return model_or_wrappers.hidden_dim


def describe_runtime(model_or_wrappers) -> str:
    if isinstance(model_or_wrappers, Mapping):
        devices = []
        runtime_dtype = None
        for agent_id, wrapper in sorted(model_or_wrappers.items()):
            devices.append(f"{agent_id}:{wrapper.device}")
            model_module = getattr(wrapper, "model", None)
            if runtime_dtype is None and model_module is not None:
                try:
                    runtime_dtype = next(model_module.parameters()).dtype
                except StopIteration:
                    runtime_dtype = None
        return (
            f"runtime_devices={devices} "
            f"runtime_dtype={runtime_dtype} "
            f"cuda_available={torch.cuda.is_available()} "
            f"cuda_visible_devices={torch.cuda.device_count()}"
        )

    model = model_or_wrappers
    model_module = getattr(model, "model", None)
    runtime_dtype = None
    if model_module is not None:
        try:
            runtime_dtype = next(model_module.parameters()).dtype
        except StopIteration:
            runtime_dtype = None

    return (
        f"runtime_device={model.device} "
        f"runtime_dtype={runtime_dtype} "
        f"cuda_available={torch.cuda.is_available()} "
        f"cuda_visible_devices={torch.cuda.device_count()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an LCN experiment")
    parser.add_argument(
        "--config",
        default=str(Path("lcn/configs/default.yaml")),
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    environment = build_environment(
        config.environment,
        hidden_dim=config.model.hidden_dim or 32,
        num_groups=config.protocol.num_groups,
    )
    agents, task = environment.setup_episode(seed=config.seed)
    model = build_model_wrappers(config.model, agents=agents)
    hidden_dim = infer_hidden_dim(model)
    for agent in agents:
        agent.hidden_dim = hidden_dim
    print(describe_runtime(model))

    kv_cache = HierarchicalKVCache(
        num_groups=config.protocol.num_groups,
        agents_per_group=config.protocol.agents_per_group,
    )
    attention = CrossLevelAttention(
        hidden_dim=hidden_dim,
        temperature=config.protocol.temperature,
    )
    protocol = ConsensusProtocol(
        kv_cache=kv_cache,
        attention=attention,
        num_rounds=config.protocol.num_rounds,
        latent_steps=config.protocol.latent_steps,
        model_wrapper=model,
    )
    protocol.register_agents(agents)

    result = protocol.run(task)
    metrics = environment.evaluate_episode(result, task)

    print(f"environment={config.environment}")
    print(f"decision={result['decision']}")
    print(f"agent_decisions={result['agent_decisions']}")
    print(f"metrics={metrics}")


if __name__ == "__main__":
    main()
