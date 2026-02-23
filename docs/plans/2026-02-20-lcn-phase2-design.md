# LCN Phase 2 Design: Model Integration & Hidden Profile Task

**Date**: 2026-02-20
**Phase**: 2 of 4
**Status**: ✅ COMPLETE (2026-02-23)
**Goal**: Integrate LCN with language model and implement first experiment environment

---

## 1. Overview

Phase 2 builds on the foundation framework from Phase 1 to create a working end-to-end system. We will:

1. Create a standalone `LCNModelWrapper` for latent space operations
2. Implement `ConsensusProtocol.run()` for the full consensus loop
3. Build the Hidden Profile Task experiment environment

### Approach

**Bottom-Up Integration**: Model Wrapper → ConsensusProtocol.run() → Hidden Profile Environment

This ensures each layer is tested before building on top of it.

---

## 2. LCNModelWrapper

### Purpose

Standalone model wrapper that handles:
- Loading and managing the language model
- Generating latent representations (KV-Cache)
- Converting latent states to text output

### Design

```python
class LCNModelWrapper:
    """
    Model wrapper for LCN latent space operations.

    Standalone implementation that provides:
    - Model loading and tokenization
    - Latent step generation (KV-Cache production)
    - Text generation from latent states
    - Latent space realignment (optional)
    """

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        latent_space_realign: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.latent_space_realign = latent_space_realign

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate_latent(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        latent_steps: int,
        past_key_values: Optional[KVCache] = None,
    ) -> Tuple[KVCache, torch.Tensor]:
        """
        Generate latent representations.

        Returns:
            kv_cache: Updated KV-Cache after latent steps
            hidden_state: Final hidden state [B, D]
        """

    def generate_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Optional[KVCache] = None,
        max_new_tokens: int = 256,
    ) -> List[str]:
        """
        Generate text from current state.
        """

    def prepare_input(
        self,
        messages: List[Dict],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare chat messages for model input.
        """
```

### Key Methods

| Method | Input | Output | Purpose |
|--------|-------|--------|---------|
| `generate_latent` | input_ids, latent_steps | KVCache, hidden_state | Produce latent representations |
| `generate_text` | input_ids, past_kv | List[str] | Generate text output |
| `prepare_input` | messages | input_ids, attention_mask | Tokenize chat messages |

---

## 3. ConsensusProtocol.run()

### Purpose

Orchestrate the multi-round consensus formation process where agents iteratively update their states based on cross-level attention.

### Design

```python
def run(
    self,
    task: str,
    model: LCNModelWrapper,
) -> ConsensusResult:
    """
    Execute consensus formation.

    Args:
        task: The task/question to solve
        model: Model wrapper for latent operations

    Returns:
        ConsensusResult with final decision and metrics
    """
    # Initialize agents with task
    for agent in self.agents:
        messages = self._build_agent_prompt(agent, task)
        input_ids, mask = model.prepare_input(messages)
        kv_cache, hidden = model.generate_latent(input_ids, mask, self.latent_steps)

        agent.set_state(hidden)
        self.kv_cache.update_local(agent.agent_id, agent.group_id, kv_cache)

    # Consensus rounds
    for round_idx in range(self.num_rounds):
        # Aggregate group and global caches
        for group_id in self.group_ids:
            self.kv_cache.aggregate_group(group_id)
        self.kv_cache.aggregate_global()

        # Each agent updates based on cross-level attention
        for agent in self.agents:
            local, group, global_ = self.kv_cache.get_all_levels(agent.agent_id)

            # Convert KV-Caches to representations
            local_repr = self._kv_to_repr(local)
            group_repr = self._kv_to_repr(group)
            global_repr = self._kv_to_repr(global_)

            # Cross-level attention fusion
            fused, weights = self.attention(
                agent.get_state(),
                local_repr,
                group_repr,
                global_repr,
            )

            # Generate new latent state
            new_kv, new_hidden = model.generate_latent(
                inputs_embeds=fused.unsqueeze(1),
                latent_steps=self.latent_steps,
            )

            agent.set_state(new_hidden)
            self.kv_cache.update_local(agent.agent_id, agent.group_id, new_kv)

    # Final decision
    return self._make_decision(task, model)
```

### ConsensusResult

```python
@dataclass
class ConsensusResult:
    """Result of consensus formation."""
    decision: str                    # Final group decision
    agent_decisions: Dict[int, str]  # Per-agent decisions
    attention_history: List[Dict]    # Attention weights over rounds
    convergence_round: Optional[int] # Round when consensus reached
```

---

## 4. Hidden Profile Task Environment

### Purpose

Implement the classic group decision-making paradigm where each agent sees only partial information, and the correct answer requires integrating information from all agents.

### Scenario Design

```
Scenario: Choose best candidate (A, B, C)

Shared information (all agents know):
- Candidate A: 2 positive traits
- Candidate B: 2 positive traits
- Candidate C: 2 positive traits

Hidden information (unique to each agent):
- Agent 0: A has 1 negative trait
- Agent 1: A has 1 negative trait
- Agent 2: B has 1 negative trait
- Agent 3: C has 3 additional positive traits (critical!)

Correct answer: C (only discoverable by integrating all information)
Common error: A or B (due to shared information bias)
```

### Design

```python
class HiddenProfileEnvironment:
    """
    Hidden Profile Task experiment environment.

    Tests whether LCN can integrate distributed information
    to make optimal group decisions.
    """

    def __init__(
        self,
        num_agents: int = 6,
        num_groups: int = 2,
    ):
        self.num_agents = num_agents
        self.num_groups = num_groups
        self.scenarios = self._load_scenarios()

    def create_agents(self) -> List[LCNAgent]:
        """Create agents with hidden profile personas."""

    def get_agent_prompt(
        self,
        agent: LCNAgent,
        scenario: HiddenProfileScenario,
    ) -> str:
        """
        Get agent's prompt with their unique information.

        Each agent sees:
        - Shared information
        - Their unique hidden information
        - Task instruction
        """

    def evaluate(
        self,
        result: ConsensusResult,
        scenario: HiddenProfileScenario,
    ) -> HiddenProfileMetrics:
        """
        Evaluate consensus result.

        Metrics:
        - accuracy: Did group choose correct answer?
        - information_integration: How much hidden info influenced decision?
        - individual_accuracy: Per-agent accuracy
        """


@dataclass
class HiddenProfileScenario:
    """A single hidden profile scenario."""
    shared_info: Dict[str, List[str]]   # candidate -> shared traits
    hidden_info: Dict[int, Dict[str, List[str]]]  # agent_id -> candidate -> hidden traits
    correct_answer: str


@dataclass
class HiddenProfileMetrics:
    """Evaluation metrics for hidden profile task."""
    accuracy: float
    information_integration: float
    individual_accuracy: Dict[int, float]
    decision_distribution: Dict[str, int]
```

---

## 5. File Structure

```
lcn/
├── models/
│   ├── __init__.py
│   └── model_wrapper.py      # LCNModelWrapper
│
├── core/
│   ├── consensus.py          # Add run() method
│   └── results.py            # ConsensusResult dataclass
│
└── environments/
    ├── __init__.py
    ├── base.py               # BaseEnvironment abstract class
    └── hidden_profile.py     # HiddenProfileEnvironment

tests/
├── test_model_wrapper.py
├── test_consensus_run.py
└── test_hidden_profile.py
```

---

## 6. Task Breakdown

### Task 8: LCNModelWrapper - Basic Structure
- Create `lcn/models/model_wrapper.py`
- Implement `__init__`, `prepare_input`
- Test with mock inputs

### Task 9: LCNModelWrapper - Latent Generation
- Implement `generate_latent`
- Handle KV-Cache management
- Test latent step production

### Task 10: LCNModelWrapper - Text Generation
- Implement `generate_text`
- Handle past_key_values
- Test text output

### Task 11: ConsensusResult Dataclass
- Create `lcn/core/results.py`
- Define ConsensusResult, AttentionRecord
- Test serialization

### Task 12: ConsensusProtocol.run() - Initialization
- Implement agent initialization phase
- Build prompts, generate initial states
- Test with 2 agents

### Task 13: ConsensusProtocol.run() - Consensus Loop
- Implement multi-round consensus
- Integrate cross-level attention
- Test convergence behavior

### Task 14: ConsensusProtocol.run() - Decision Making
- Implement `_make_decision`
- Aggregate agent outputs
- Test decision extraction

### Task 15: BaseEnvironment
- Create `lcn/environments/base.py`
- Define abstract interface
- Document expected methods

### Task 16: HiddenProfileEnvironment - Scenarios
- Create `lcn/environments/hidden_profile.py`
- Implement scenario loading/generation
- Test scenario validity

### Task 17: HiddenProfileEnvironment - Agent Prompts
- Implement `get_agent_prompt`
- Handle information distribution
- Test prompt correctness

### Task 18: HiddenProfileEnvironment - Evaluation
- Implement `evaluate`
- Compute metrics
- Test with mock results

### Task 19: Integration Test
- End-to-end test with small model
- Run single scenario
- Verify all components work together

---

## 7. Dependencies

### Required
- torch>=2.0.0
- transformers>=4.35.0
- pytest>=7.0.0

### Model
- Qwen/Qwen3-4B (for development/testing)
- Qwen/Qwen3-14B (for experiments)

---

## 8. Success Criteria

- [x] LCNModelWrapper can generate latent representations
- [x] ConsensusProtocol.run() executes full consensus loop
- [x] Hidden Profile environment generates valid scenarios
- [x] Integration test passes with mock model (Qwen3-4B ready)
- [x] All unit tests pass (207 tests)

---

## 9. Implementation Summary

**Completed**: 2026-02-23

### Files Created
```
lcn/
├── models/
│   ├── __init__.py
│   └── model_wrapper.py      # LCNModelWrapper (Tasks 8-10)
├── core/
│   └── results.py            # ConsensusResult (Task 11)
└── environments/
    ├── __init__.py
    ├── base.py               # BaseEnvironment (Task 15)
    └── hidden_profile.py     # HiddenProfileEnvironment (Tasks 16-18)

tests/
├── test_model_wrapper.py     # 43 tests
├── test_consensus.py         # 51 tests (extended)
├── test_environments.py      # 14 tests
├── test_hidden_profile.py    # 55 tests
└── test_integration.py       # 26 tests
```

### Test Coverage
- **207 total tests passing**
- All components fully tested with TDD approach
- End-to-end integration tests with mock model

### Key Commits
- `feat(models): add LCNModelWrapper with basic init and tokenization`
- `feat(models): add generate_latent method with latent space realignment`
- `feat(models): add generate_text method for text generation`
- `feat(core): add ConsensusResult dataclass for consensus outcomes`
- `feat(consensus): add agent initialization phase to ConsensusProtocol`
- `feat(consensus): implement run() method for consensus formation loop`
- `feat(consensus): add decision making with majority vote aggregation`
- `feat(environments): add BaseEnvironment abstract base class`
- `feat(environments): add HiddenProfileEnvironment for hidden profile task`
- `feat(environments): implement get_agent_prompt for HiddenProfileEnvironment`
- `feat(environments): add HiddenProfileMetrics and evaluate method`
- `test(integration): add end-to-end integration tests with mock model`

---

*Design created: 2026-02-20*
*Implementation completed: 2026-02-23*
