# LCN Phase 3 Design: Validation with Real Model

**Date**: 2026-02-23
**Phase**: 3 of 4
**Goal**: Validate LCN works end-to-end with real model (Qwen3-4B)

---

## 1. Overview

Phase 3 focuses on **minimal validation first** - getting a single Hidden Profile scenario running with a real model before building additional infrastructure.

### Approach

**Minimal Validation First**:
1. Add model loading to `LCNModelWrapper`
2. Fix integration gaps between components
3. Create simple experiment script
4. Run single scenario, verify results

This approach gets fast feedback on whether the system works before investing in more environments or infrastructure.

### Configuration

- **Model**: Qwen/Qwen3-4B
- **Agents**: 6 agents, 2 groups
- **Scenarios**: 1 (quick validation)
- **GPU**: Multiple GPUs with `device_map="auto"`

---

## 2. Model Loading

Add `load()` method to `LCNModelWrapper`:

```python
def load(self) -> None:
    """Load model and tokenizer from HuggingFace."""
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    self.model = AutoModelForCausalLM.from_pretrained(
        self.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Use multiple GPUs automatically
    )
    self.model.eval()

    # Ensure pad token exists
    if self.tokenizer.pad_token_id is None:
        self.tokenizer.pad_token = self.tokenizer.eos_token
```

**Key decisions:**
- Use `device_map="auto"` to leverage multiple GPUs
- Use `bfloat16` for memory efficiency
- Reuse pattern from `LatentMAS/models.py`

---

## 3. Integration Fixes

### Issue 1: Agent Prompt Integration

`ConsensusProtocol._build_agent_prompt()` builds generic prompts, but `HiddenProfileEnvironment.get_agent_prompt()` builds scenario-specific prompts with hidden info.

**Solution:** Modify `ConsensusProtocol.run()` to accept optional `agent_prompts: Dict[int, str]`. When provided, use per-agent prompts instead of `_build_agent_prompt()`.

### Issue 2: Hidden Dimension Mismatch

Agents created with `hidden_dim=64` (default) but model has `hidden_dim=2048`.

**Solution:** Pass model's `hidden_size` to environment after model loads:
```python
env = HiddenProfileEnvironment(
    num_agents=6,
    num_groups=2,
    hidden_dim=model.model.config.hidden_size
)
```

### Issue 3: CrossLevelAttention Dimension

**Solution:** Get hidden_dim dynamically from model:
```python
hidden_dim = model.model.config.hidden_size
attention = CrossLevelAttention(hidden_dim=hidden_dim)
```

---

## 4. Experiment Script

Create `experiments/run_hidden_profile.py`:

```python
"""Run Hidden Profile experiment with LCN."""

def main():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "Qwen/Qwen3-4B"

    # 2. Load model
    model = LCNModelWrapper(model_name, device)
    model.load()
    hidden_dim = model.model.config.hidden_size

    # 3. Create environment and agents
    env = HiddenProfileEnvironment(num_agents=6, num_groups=2, hidden_dim=hidden_dim)
    agents = env.create_agents()
    scenario = env.scenarios[0]

    # 4. Build per-agent prompts
    agent_prompts = {
        agent.agent_id: env.get_agent_prompt(agent, scenario)
        for agent in agents
    }

    # 5. Setup consensus protocol
    kv_cache = HierarchicalKVCache(num_groups=2, agents_per_group=3)
    attention = CrossLevelAttention(hidden_dim=hidden_dim)
    protocol = ConsensusProtocol(
        kv_cache=kv_cache,
        attention=attention,
        num_rounds=3,
        latent_steps=5,
    )
    protocol.register_agents(agents)

    # 6. Run consensus
    result = protocol.run(task=scenario.correct_answer, model=model, agent_prompts=agent_prompts)

    # 7. Evaluate
    metrics = env.evaluate(result, scenario)

    # 8. Print results
    print(f"Decision: {result.decision}")
    print(f"Correct answer: {scenario.correct_answer}")
    print(f"Accuracy: {metrics.accuracy}")
    print(f"Agent decisions: {result.agent_decisions}")
```

---

## 5. Task Breakdown

### Task 20: Add `load()` method to LCNModelWrapper
- Add `load()` method with `device_map="auto"` for multi-GPU
- Add `is_loaded` property
- Update existing methods to check model is loaded
- Test with mock (don't require actual model in unit tests)

### Task 21: Fix agent prompt integration
- Modify `ConsensusProtocol.run()` to accept optional `agent_prompts: Dict[int, str]`
- When provided, use per-agent prompts instead of `_build_agent_prompt()`
- Update `_initialize_agents()` to use custom prompts

### Task 22: Create experiment script
- Create `experiments/run_hidden_profile.py`
- Pass model's `hidden_size` to environment and attention
- Build per-agent prompts from environment
- Run single scenario and print results

### Task 23: Run validation and fix issues
- Run script with Qwen3-4B
- Debug runtime issues
- Verify `ConsensusResult` and `HiddenProfileMetrics` are correct

---

## 6. Success Criteria

- [ ] `LCNModelWrapper.load()` successfully loads Qwen3-4B
- [ ] `ConsensusProtocol.run()` completes without errors
- [ ] `ConsensusResult` contains valid decision and agent_decisions
- [ ] `HiddenProfileMetrics` computes accuracy correctly
- [ ] Script runs end-to-end on single scenario

---

## 7. Next Steps (After Validation)

Once validation passes:
1. Run multiple scenarios for statistical significance
2. Implement Asch Conformity environment
3. Implement Wisdom of Crowds environment
4. Run ablation studies
5. Generate paper-ready results

---

*Design created: 2026-02-23*
