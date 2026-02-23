# LCN Phase 3 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Validate LCN works end-to-end with real model (Qwen3-4B) on Hidden Profile task

**Architecture:** Add model loading to LCNModelWrapper, fix prompt integration, create experiment script

**Tech Stack:** Python 3.10+, PyTorch, Transformers, Qwen3-4B, pytest

---

## Task 20: Add `load()` method to LCNModelWrapper

**Files:**
- Modify: `lcn/models/model_wrapper.py`
- Modify: `tests/test_model_wrapper.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_model_wrapper.py

class TestLCNModelWrapperLoad:
    """Tests for LCNModelWrapper.load() method."""

    def test_load_method_exists(self):
        """load() method should exist."""
        from lcn.models.model_wrapper import LCNModelWrapper
        import torch

        wrapper = LCNModelWrapper("mock-model", torch.device("cpu"))
        assert hasattr(wrapper, "load")
        assert callable(wrapper.load)

    def test_is_loaded_property_false_before_load(self):
        """is_loaded should be False before load() is called."""
        from lcn.models.model_wrapper import LCNModelWrapper
        import torch

        wrapper = LCNModelWrapper("mock-model", torch.device("cpu"))
        assert wrapper.is_loaded is False

    def test_unload_method_exists(self):
        """unload() method should exist."""
        from lcn.models.model_wrapper import LCNModelWrapper
        import torch

        wrapper = LCNModelWrapper("mock-model", torch.device("cpu"))
        assert hasattr(wrapper, "unload")
        assert callable(wrapper.unload)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_model_wrapper.py::TestLCNModelWrapperLoad -v`
Expected: FAIL with "AttributeError: 'LCNModelWrapper' object has no attribute 'load'"

**Step 3: Write minimal implementation**

Add to `lcn/models/model_wrapper.py` after `__init__`:

```python
@property
def is_loaded(self) -> bool:
    """Check if model and tokenizer are loaded."""
    return self.model is not None and self.tokenizer is not None

def load(self) -> None:
    """
    Load model and tokenizer from HuggingFace.

    Uses device_map="auto" for multi-GPU support and bfloat16 for efficiency.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    self.model = AutoModelForCausalLM.from_pretrained(
        self.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    self.model.eval()

    # Ensure pad token exists
    if self.tokenizer.pad_token_id is None:
        self.tokenizer.pad_token = self.tokenizer.eos_token

def unload(self) -> None:
    """Unload model and tokenizer to free memory."""
    self.model = None
    self.tokenizer = None
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_model_wrapper.py::TestLCNModelWrapperLoad -v`
Expected: PASS

**Step 5: Commit**

```bash
git add lcn/models/model_wrapper.py tests/test_model_wrapper.py
git commit -m "feat(models): add load() and unload() methods to LCNModelWrapper"
```

---

## Task 21: Fix agent prompt integration in ConsensusProtocol

**Files:**
- Modify: `lcn/core/consensus.py`
- Modify: `tests/test_consensus.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_consensus.py

class TestRunWithCustomPrompts:
    """Tests for ConsensusProtocol.run() with custom agent prompts."""

    def test_run_accepts_agent_prompts_parameter(self):
        """run() should accept optional agent_prompts dict."""
        from lcn.core.consensus import ConsensusProtocol
        from lcn.core.kv_cache import HierarchicalKVCache
        from lcn.core.attention import CrossLevelAttention
        from lcn.core.agent import LCNAgent
        from unittest.mock import MagicMock
        import torch

        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=2)
        attention = CrossLevelAttention(hidden_dim=64)
        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=1,
            latent_steps=1,
        )

        agents = [
            LCNAgent(agent_id=0, group_id=0, hidden_dim=64),
            LCNAgent(agent_id=1, group_id=0, hidden_dim=64),
        ]
        protocol.register_agents(agents)

        # Create mock model
        mock_model = MagicMock()
        mock_model.prepare_input.return_value = (
            torch.zeros(1, 10),
            torch.ones(1, 10),
        )
        mock_kv = tuple(
            (torch.randn(1, 4, 3, 16), torch.randn(1, 4, 3, 16))
            for _ in range(2)
        )
        mock_model.generate_latent.return_value = (mock_kv, torch.randn(1, 64))
        mock_model.generate_text.return_value = ["Decision A"]

        # Custom prompts per agent
        agent_prompts = {
            0: "You are Agent 0. Your task is X.",
            1: "You are Agent 1. Your task is Y.",
        }

        # Should not raise
        result = protocol.run("task", mock_model, agent_prompts=agent_prompts)
        assert result is not None

    def test_run_uses_custom_prompts_when_provided(self):
        """run() should use agent_prompts instead of _build_agent_prompt when provided."""
        from lcn.core.consensus import ConsensusProtocol
        from lcn.core.kv_cache import HierarchicalKVCache
        from lcn.core.attention import CrossLevelAttention
        from lcn.core.agent import LCNAgent
        from unittest.mock import MagicMock, call
        import torch

        kv_cache = HierarchicalKVCache(num_groups=1, agents_per_group=2)
        attention = CrossLevelAttention(hidden_dim=64)
        protocol = ConsensusProtocol(
            kv_cache=kv_cache,
            attention=attention,
            num_rounds=1,
            latent_steps=1,
        )

        agents = [
            LCNAgent(agent_id=0, group_id=0, hidden_dim=64),
            LCNAgent(agent_id=1, group_id=0, hidden_dim=64),
        ]
        protocol.register_agents(agents)

        mock_model = MagicMock()
        mock_model.prepare_input.return_value = (
            torch.zeros(1, 10),
            torch.ones(1, 10),
        )
        mock_kv = tuple(
            (torch.randn(1, 4, 3, 16), torch.randn(1, 4, 3, 16))
            for _ in range(2)
        )
        mock_model.generate_latent.return_value = (mock_kv, torch.randn(1, 64))
        mock_model.generate_text.return_value = ["Decision A"]

        agent_prompts = {
            0: "Custom prompt for agent 0",
            1: "Custom prompt for agent 1",
        }

        protocol.run("task", mock_model, agent_prompts=agent_prompts)

        # Check that prepare_input was called with custom prompts
        calls = mock_model.prepare_input.call_args_list
        # First two calls are from _initialize_agents
        assert any(
            "Custom prompt for agent 0" in str(c) for c in calls
        )
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_consensus.py::TestRunWithCustomPrompts -v`
Expected: FAIL with "TypeError: run() got an unexpected keyword argument 'agent_prompts'"

**Step 3: Write minimal implementation**

Modify `lcn/core/consensus.py`:

1. Update `_initialize_agents` signature:
```python
def _initialize_agents(
    self,
    task: str,
    model: "LCNModelWrapper",
    agent_prompts: Optional[Dict[int, str]] = None,
) -> None:
    """
    Initialize all agents with task, generate initial latent states.

    Args:
        task: The task/question for agents to consider
        model: The LCNModelWrapper to use for generation
        agent_prompts: Optional dict mapping agent_id to custom prompt string.
                      If provided, uses these instead of _build_agent_prompt.
    """
    for agent in self.agents:
        # Use custom prompt if provided, otherwise build default
        if agent_prompts and agent.agent_id in agent_prompts:
            messages = [{"role": "user", "content": agent_prompts[agent.agent_id]}]
        else:
            messages = self._build_agent_prompt(agent, task)

        # Rest of the method unchanged...
        input_ids, attention_mask = model.prepare_input(messages)
        kv_cache, hidden_state = model.generate_latent(
            input_ids=input_ids,
            attention_mask=attention_mask,
            latent_steps=self.latent_steps,
        )
        agent.set_state(hidden_state)
        self.kv_cache.update_local(agent.agent_id, agent.group_id, kv_cache)
```

2. Update `run` signature:
```python
def run(
    self,
    task: str,
    model: "LCNModelWrapper",
    agent_prompts: Optional[Dict[int, str]] = None,
) -> ConsensusResult:
    """
    Execute full consensus formation loop.

    Args:
        task: The task/question for agents to consider
        model: The LCNModelWrapper to use for generation
        agent_prompts: Optional dict mapping agent_id to custom prompt string.
                      If provided, uses these instead of _build_agent_prompt.
    """
    # Step 1: Initialize agents
    self._initialize_agents(task, model, agent_prompts)
    # ... rest unchanged
```

3. Also update `_make_decision` to use custom prompts:
```python
def _make_decision(
    self,
    task: str,
    model: "LCNModelWrapper",
    attention_history: List[Dict],
    agent_prompts: Optional[Dict[int, str]] = None,
) -> ConsensusResult:
    # ... in the loop:
    if agent_prompts and agent.agent_id in agent_prompts:
        messages = [{"role": "user", "content": agent_prompts[agent.agent_id]}]
    else:
        messages = self._build_agent_prompt(agent, task)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_consensus.py::TestRunWithCustomPrompts -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All 207+ tests PASS

**Step 6: Commit**

```bash
git add lcn/core/consensus.py tests/test_consensus.py
git commit -m "feat(consensus): add agent_prompts parameter to run() for custom prompts"
```

---

## Task 22: Create experiment script

**Files:**
- Create: `experiments/run_hidden_profile.py`

**Step 1: Create the experiment script**

```python
#!/usr/bin/env python
"""
Run Hidden Profile experiment with LCN.

This script validates that LCN works end-to-end with a real model.
It runs a single Hidden Profile scenario and prints the results.

Usage:
    python experiments/run_hidden_profile.py
    python experiments/run_hidden_profile.py --model Qwen/Qwen3-14B
    python experiments/run_hidden_profile.py --num-rounds 5 --latent-steps 10
"""

import argparse
import torch

from lcn.core.agent import LCNAgent
from lcn.core.kv_cache import HierarchicalKVCache
from lcn.core.attention import CrossLevelAttention
from lcn.core.consensus import ConsensusProtocol
from lcn.models.model_wrapper import LCNModelWrapper
from lcn.environments.hidden_profile import HiddenProfileEnvironment


def parse_args():
    parser = argparse.ArgumentParser(description="Run Hidden Profile experiment")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B",
        help="HuggingFace model name (default: Qwen/Qwen3-4B)",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=6,
        help="Number of agents (default: 6)",
    )
    parser.add_argument(
        "--num-groups",
        type=int,
        default=2,
        help="Number of groups (default: 2)",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=3,
        help="Number of consensus rounds (default: 3)",
    )
    parser.add_argument(
        "--latent-steps",
        type=int,
        default=5,
        help="Number of latent steps per round (default: 5)",
    )
    parser.add_argument(
        "--scenario-idx",
        type=int,
        default=0,
        help="Scenario index to run (default: 0)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("LCN Hidden Profile Experiment")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Agents: {args.num_agents}, Groups: {args.num_groups}")
    print(f"Rounds: {args.num_rounds}, Latent Steps: {args.latent_steps}")
    print("=" * 60)

    # 1. Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # 2. Load model
    print(f"\nLoading model: {args.model}...")
    model = LCNModelWrapper(args.model, device)
    model.load()
    hidden_dim = model.model.config.hidden_size
    print(f"Model loaded. Hidden dim: {hidden_dim}")

    # 3. Create environment and agents
    print("\nCreating environment and agents...")
    env = HiddenProfileEnvironment(
        num_agents=args.num_agents,
        num_groups=args.num_groups,
        hidden_dim=hidden_dim,
    )
    agents = env.create_agents()
    scenario = env.scenarios[args.scenario_idx]
    print(f"Scenario: {args.scenario_idx}")
    print(f"Correct answer: {scenario.correct_answer}")

    # 4. Build per-agent prompts
    print("\nBuilding agent prompts...")
    agent_prompts = {
        agent.agent_id: env.get_agent_prompt(agent, scenario)
        for agent in agents
    }

    # 5. Setup consensus protocol
    print("\nSetting up consensus protocol...")
    agents_per_group = args.num_agents // args.num_groups
    kv_cache = HierarchicalKVCache(
        num_groups=args.num_groups,
        agents_per_group=agents_per_group,
    )
    attention = CrossLevelAttention(hidden_dim=hidden_dim)
    attention = attention.to(device)

    protocol = ConsensusProtocol(
        kv_cache=kv_cache,
        attention=attention,
        num_rounds=args.num_rounds,
        latent_steps=args.latent_steps,
    )
    protocol.register_agents(agents)

    # 6. Run consensus
    print("\nRunning consensus...")
    print("-" * 40)
    task = f"Choose the best candidate from: {', '.join(scenario.shared_info.keys())}"
    result = protocol.run(task, model, agent_prompts=agent_prompts)

    # 7. Evaluate
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    metrics = env.evaluate(result, scenario)

    print(f"\nGroup Decision: {result.decision}")
    print(f"Correct Answer: {scenario.correct_answer}")
    print(f"Accuracy: {metrics.accuracy:.2f}")

    print("\nAgent Decisions:")
    for agent_id, decision in result.agent_decisions.items():
        correct = "✓" if decision == scenario.correct_answer else "✗"
        print(f"  Agent {agent_id}: {decision} {correct}")

    print(f"\nDecision Distribution: {metrics.decision_distribution}")
    print(f"Convergence Round: {result.convergence_round}")

    print("\n" + "=" * 60)
    print("Experiment complete!")
    print("=" * 60)

    # Cleanup
    model.unload()


if __name__ == "__main__":
    main()
```

**Step 2: Verify script is syntactically correct**

Run: `python -m py_compile experiments/run_hidden_profile.py`
Expected: No output (success)

**Step 3: Commit**

```bash
git add experiments/run_hidden_profile.py
git commit -m "feat(experiments): add run_hidden_profile.py script"
```

---

## Task 23: Run validation and fix issues

**Files:**
- May need to modify various files based on errors encountered

**Step 1: Run the experiment**

```bash
cd /path/to/agent-social-simulacra
python experiments/run_hidden_profile.py --model Qwen/Qwen3-4B
```

**Step 2: Debug any errors**

Common issues to watch for:
- CUDA out of memory: Try `--num-agents 4`
- Model loading issues: Check HuggingFace credentials
- Dimension mismatches: Verify hidden_dim propagation
- Tokenizer issues: Check pad_token handling

**Step 3: Verify results**

Expected output:
```
================================================================
LCN Hidden Profile Experiment
================================================================
Model: Qwen/Qwen3-4B
Agents: 6, Groups: 2
Rounds: 3, Latent Steps: 5
================================================================

Device: cuda
Loading model: Qwen/Qwen3-4B...
Model loaded. Hidden dim: 2048

Creating environment and agents...
Scenario: 0
Correct answer: C

Building agent prompts...
Setting up consensus protocol...
Running consensus...
----------------------------------------

================================================================
RESULTS
================================================================

Group Decision: [some decision]
Correct Answer: C
Accuracy: [0.0 or 1.0]

Agent Decisions:
  Agent 0: [decision] [✓ or ✗]
  Agent 1: [decision] [✓ or ✗]
  ...

Decision Distribution: {'A': X, 'B': Y, 'C': Z}
Convergence Round: 3

================================================================
Experiment complete!
================================================================
```

**Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix: address issues found during validation run"
```

---

## Success Criteria

- [ ] `LCNModelWrapper.load()` successfully loads Qwen3-4B
- [ ] `ConsensusProtocol.run()` accepts and uses `agent_prompts`
- [ ] `experiments/run_hidden_profile.py` runs without errors
- [ ] `ConsensusResult` contains valid decision and agent_decisions
- [ ] `HiddenProfileMetrics` computes accuracy correctly
- [ ] All 210+ tests pass

---

*Plan created: 2026-02-23*
