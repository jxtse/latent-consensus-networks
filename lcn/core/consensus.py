"""Consensus formation protocol for LCN."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping as MappingABC
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch

from lcn.core.agent import LCNAgent
from lcn.core.attention import CrossLevelAttention
from lcn.core.kv_cache import HierarchicalKVCache
from lcn.models.latent_ops import cache_to_level_tensor, stack_kv_cache_summaries
from lcn.models.model_wrapper import BaseModelWrapper

ModelWrapperRef = BaseModelWrapper | Mapping[int, BaseModelWrapper]


class ConsensusProtocol:
    """
    Consensus formation protocol for Latent Consensus Networks.

    Orchestrates the multi-round consensus formation process where
    agents iteratively update their states based on cross-level attention.
    """

    def __init__(
        self,
        kv_cache: HierarchicalKVCache,
        attention: CrossLevelAttention,
        num_rounds: int = 3,
        latent_steps: int = 5,
        model_wrapper: Optional[ModelWrapperRef] = None,
    ):
        self.kv_cache = kv_cache
        self.attention = attention
        self.num_rounds = num_rounds
        self.latent_steps = latent_steps
        self.model_wrapper = model_wrapper

        self.agents: List[LCNAgent] = []
        self._agent_map: Dict[int, LCNAgent] = {}

    def register_agents(self, agents: List[LCNAgent]) -> None:
        """Register agents with the protocol."""
        self.agents = agents
        self._agent_map = {agent.agent_id: agent for agent in agents}

    def get_agent(self, agent_id: int) -> Optional[LCNAgent]:
        """Get an agent by ID."""
        return self._agent_map.get(agent_id)

    def get_agents_by_group(self, group_id: int) -> List[LCNAgent]:
        """Get all agents in a specific group."""
        return [agent for agent in self.agents if agent.group_id == group_id]

    @property
    def group_ids(self) -> List[int]:
        """Get all unique group IDs."""
        return list(set(agent.group_id for agent in self.agents))

    def initialize_agent_states(
        self,
        task: Mapping[str, Any],
        *,
        model_wrapper: Optional[ModelWrapperRef] = None,
        force: bool = False,
    ) -> None:
        """Initialize missing agent states and caches from the model wrapper."""
        model_ref = self._resolve_model_wrapper(model_wrapper)
        task_prompt = str(task.get("prompt", ""))

        for agent in self.agents:
            if agent.state is not None and not force:
                continue

            model = self._resolve_agent_model_wrapper(agent, model_ref)
            observation = agent.metadata.get("observation", "")
            prompt = f"{task_prompt}\n{observation}".strip()
            state = model.initialize_state(
                prompt,
                persona=agent.persona,
                metadata=agent.metadata,
            )
            agent.set_state(state)
            self.kv_cache.update_local(
                agent_id=agent.agent_id,
                group_id=agent.group_id,
                kv_cache=model.build_kv_cache(state),
            )

        self._refresh_hierarchy()

    def run(
        self,
        task: Mapping[str, Any],
        *,
        model_wrapper: Optional[ModelWrapperRef] = None,
    ) -> Dict[str, Any]:
        """Execute multi-round latent consensus and return group-level outputs."""
        if not self.agents:
            raise ValueError("No agents registered")

        model_ref = self._resolve_model_wrapper(model_wrapper)
        self.initialize_agent_states(task, model_wrapper=model_ref)

        history: List[Dict[str, Any]] = []
        for round_idx in range(self.num_rounds):
            round_trace = []
            for agent in self.agents:
                agent_model = self._resolve_agent_model_wrapper(agent, model_ref)
                fused_state, attn_weights = self._fuse_agent_context(agent)
                updated_state = self._run_latent_steps(
                    agent=agent,
                    fused_state=fused_state,
                    task=task,
                    model_wrapper=agent_model,
                )
                agent.set_state(updated_state)
                self.kv_cache.update_local(
                    agent_id=agent.agent_id,
                    group_id=agent.group_id,
                    kv_cache=agent_model.build_kv_cache(updated_state),
                )
                round_trace.append(
                    {
                        "agent_id": agent.agent_id,
                        "group_id": agent.group_id,
                        "attention_weights": attn_weights.detach().cpu(),
                    }
                )

            self._refresh_hierarchy()
            history.append({"round": round_idx, "updates": round_trace})

        agent_decisions = self._collect_agent_decisions(task, model_ref)
        final_decision = self._majority_vote(agent_decisions)
        return {
            "decision": final_decision,
            "agent_decisions": agent_decisions,
            "history": history,
        }

    def _fuse_agent_context(self, agent: LCNAgent) -> Tuple[torch.Tensor, torch.Tensor]:
        query_state = agent.get_state()
        if query_state is None:
            raise ValueError(f"Agent {agent.agent_id} does not have an initialized state")

        self.attention.to(device=query_state.device, dtype=query_state.dtype)
        local_caches, group_cache, global_cache = self.kv_cache.get_all_levels(agent.agent_id)
        local_repr = stack_kv_cache_summaries(local_caches)
        group_repr = cache_to_level_tensor(group_cache)
        global_repr = cache_to_level_tensor(global_cache)
        if local_repr is not None:
            local_repr = local_repr.to(device=query_state.device, dtype=query_state.dtype)
        if group_repr is not None:
            group_repr = group_repr.to(device=query_state.device, dtype=query_state.dtype)
        if global_repr is not None:
            global_repr = global_repr.to(device=query_state.device, dtype=query_state.dtype)
        return self.attention(query_state, local_repr, group_repr, global_repr)

    def _run_latent_steps(
        self,
        *,
        agent: LCNAgent,
        fused_state: torch.Tensor,
        task: Mapping[str, Any],
        model_wrapper: BaseModelWrapper,
    ) -> torch.Tensor:
        prompt = str(task.get("prompt", ""))
        state = agent.get_state()
        for step_idx in range(self.latent_steps):
            state = model_wrapper.latent_step(
                state,
                fused_state,
                prompt=prompt,
                metadata=agent.metadata,
                step_idx=step_idx,
            )
        return state

    def _collect_agent_decisions(
        self,
        task: Mapping[str, Any],
        model_wrapper: ModelWrapperRef,
    ) -> Dict[int, str]:
        options = task.get("options")
        if not options:
            raise ValueError("task must define 'options'")

        agent_decisions = {}
        for agent in self.agents:
            agent_model = self._resolve_agent_model_wrapper(agent, model_wrapper)
            agent_decisions[agent.agent_id] = agent_model.choose_option(
                agent.get_state(),
                options,
                metadata=task,
            )
        return agent_decisions

    def _majority_vote(self, agent_decisions: Mapping[int, str]) -> str:
        counts = Counter(agent_decisions.values())
        return counts.most_common(1)[0][0]

    def _refresh_hierarchy(self) -> None:
        for group_id in self.group_ids:
            self.kv_cache.aggregate_group(group_id)
        self.kv_cache.aggregate_global()

    def _resolve_model_wrapper(
        self,
        model_wrapper: Optional[ModelWrapperRef] = None,
    ) -> ModelWrapperRef:
        resolved = model_wrapper or self.model_wrapper
        if resolved is None:
            raise ValueError("A model_wrapper must be provided to run consensus")
        return resolved

    def _resolve_agent_model_wrapper(
        self,
        agent: LCNAgent,
        model_wrapper: Optional[ModelWrapperRef] = None,
    ) -> BaseModelWrapper:
        resolved = self._resolve_model_wrapper(model_wrapper)
        if isinstance(resolved, MappingABC):
            try:
                return resolved[agent.agent_id]
            except KeyError as exc:
                raise KeyError(
                    f"No model wrapper registered for agent_id={agent.agent_id}"
                ) from exc
        return resolved
