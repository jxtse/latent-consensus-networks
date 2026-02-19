# Latent Consensus Networks (LCN) - Design Document

**Date**: 2026-02-19
**Target**: NeurIPS 2026 (Deadline: ~May 2026)
**Timeline**: 12 weeks (Feb - May 2026)

---

## 1. Executive Summary

### Project Name
**Latent Consensus Networks (LCN)**

### One-Line Description
A hierarchical, attention-driven latent space consensus mechanism that enables multiple LLM agents to form layered group consensus through shared KV-Cache, achieving collective intelligence beyond individual capabilities.

### Core Contribution
1. **Framework**: First hierarchical latent-space framework for multi-agent consensus (Local/Group/Global KV-Cache)
2. **Mechanism**: Cross-Level Attention module for adaptive information fusion across hierarchy levels
3. **Empirical Findings**: Optimal sharing regime discovery, classic social psychology phenomena reproduction

---

## 2. Research Context

### Target Venue
- **Conference**: NeurIPS 2026
- **Track**: AI/ML (Main Conference)
- **Deadline**: ~May 15, 2026

### Resource Constraints
- **Timeline**: 2-3 months
- **Compute**: 1-2 GPUs
- **Agent Scale**: 10-50 agents

### Research Positioning
| Dimension | Choice |
|-----------|--------|
| Core Contribution | Latent Consensus Protocol (Technical Innovation) |
| Application | Social Dynamics Simulation |
| Architecture | Hierarchical + Cross-Level Attention |
| Experiments | Group Decision Tasks + Classic Social Science Experiments |

---

## 3. Technical Architecture

### 3.1 System Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                    Latent Consensus Networks (LCN)                  │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Agent₁    Agent₂    Agent₃    Agent₄    Agent₅    Agent₆        │
│     │         │         │         │         │         │            │
│     └────┬────┘         └────┬────┘         └────┬────┘            │
│          ▼                   ▼                   ▼                  │
│    ┌──────────┐        ┌──────────┐        ┌──────────┐            │
│    │ Group A  │        │ Group B  │        │ Group C  │            │
│    │ KV-Cache │        │ KV-Cache │        │ KV-Cache │            │
│    └────┬─────┘        └────┬─────┘        └────┬─────┘            │
│         └──────────────┬────┴───────────────────┘                  │
│                        ▼                                            │
│               ┌─────────────────┐                                   │
│               │  Global KV-Cache │                                  │
│               │  (Social Norm)   │                                  │
│               └─────────────────┘                                   │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │              Cross-Level Attention Module                    │  │
│   │   Agent_i's decision = Attend([Local, Group, Global])        │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### 3.2 Comparison with LatentMAS

| Aspect | LatentMAS (Original) | LCN (Ours) |
|--------|---------------------|------------|
| KV-Cache Structure | Single shared | Hierarchical (Local/Group/Global) |
| Information Flow | Linear pipeline | Network topology + hierarchical aggregation |
| Agent Interaction | Fixed order | Dynamic Attention selection |
| Application | Reasoning tasks | Reasoning + Social dynamics |

### 3.3 Cross-Level Attention Mechanism

```python
class CrossLevelAttention:
    """
    Agent_i uses its current state as Query,
    attends to three levels of KV-Cache,
    produces weighted fusion of Latent States.
    """

    def forward(self, agent_state, local_kv, group_kv, global_kv):
        # agent_state: [B, D] - Agent_i's current hidden state
        # local_kv:  KV-Cache from neighbor agents
        # group_kv:  Aggregated KV-Cache from group
        # global_kv: Aggregated KV-Cache from all groups

        # Step 1: Project all levels to unified space
        local_repr  = self.project_local(local_kv)   # [B, L1, D]
        group_repr  = self.project_group(group_kv)   # [B, L2, D]
        global_repr = self.project_global(global_kv) # [B, L3, D]

        # Step 2: Concatenate all levels
        all_levels = concat([local_repr, group_repr, global_repr])  # [B, L_total, D]

        # Step 3: Agent state as Query, compute Attention
        query = self.query_proj(agent_state)  # [B, 1, D]
        attn_weights = softmax(query @ all_levels.T / sqrt(D))  # [B, 1, L_total]

        # Step 4: Weighted fusion
        fused_state = attn_weights @ all_levels  # [B, 1, D]

        return fused_state, attn_weights
```

### 3.4 Hierarchical KV-Cache Aggregation

```
Local Level:
  Agent_i's neighbor set N(i)'s KV-Cache
  local_kv = Stack([kv_j for j in N(i)])  # No aggregation, keep each neighbor

Group Level:
  Aggregate all agents' KV-Cache within Group_g
  group_kv = MeanPool([kv_j for j in Group_g])

Global Level:
  Aggregate all Groups' KV-Cache
  global_kv = WeightedMean([group_kv_g for g in Groups], weights=group_sizes)
```

### 3.5 Key Hyperparameters

| Parameter | Meaning | Suggested Range |
|-----------|---------|-----------------|
| `num_groups` | Number of groups | 2-5 |
| `agents_per_group` | Agents per group | 3-10 |
| `latent_steps` | Latent reasoning steps per round | 5-20 |
| `num_rounds` | Consensus formation rounds | 3-10 |
| `temperature` | Attention softmax temperature | 0.1-1.0 |

---

## 4. Experiment Design

### 4.1 Experiment Overview

| Type | Purpose | Count |
|------|---------|-------|
| **Main Experiments** | Prove LCN effectiveness | 2-3 tasks |
| **Ablation Studies** | Analyze component contributions | 4-5 variants |
| **Analysis Experiments** | Provide insights and interpretability | 2-3 analyses |

### 4.2 Main Experiment 1: Hidden Profile Task

**Description**: Classic group decision-making paradigm where each agent sees only partial information.

**Setup**:
```
Scenario: Choose best candidate (A, B, C)

Information distribution:
  - Shared info: All agents know A has 2 pros, B has 2 pros, C has 2 pros
  - Hidden info:
    - Agent₁ alone knows: A has 1 con
    - Agent₂ alone knows: A has 1 con
    - Agent₃ alone knows: B has 1 con
    - Agent₄ alone knows: C has 3 pros (critical info!)

Correct answer: C (only discoverable by integrating all information)
Common error: Choose A or B (due to shared information bias)
```

**Metrics**:
- Accuracy: Proportion of correct group decisions
- Information integration: Degree to which hidden info influences decisions

**Baselines**:
- Single Agent (partial information only)
- Text-MAS (text communication)
- LatentMAS (original design)
- **LCN (ours)**

### 4.3 Main Experiment 2: Asch Conformity Experiment

**Description**: Classic social psychology experiment testing conformity under group pressure.

**Setup**:
```
Scenario: Line length judgment (clearly has correct answer)

Configuration:
  - 1 Target Agent (observed subject)
  - 5 Confederate Agents (preset to give wrong answers)

Variables:
  - Confederate consistency: All wrong vs one correct
  - Sharing degree: α_group intensity

Measurements:
  - Target Agent conformity rate
  - Relationship between conformity and sharing degree
```

**Research Questions**:
- Do LCN agents exhibit human-like conformity behavior?
- Can Cross-Level Attention help agents resist incorrect group pressure?
- How do different levels (Local/Group/Global) affect conformity?

**Human Data Comparison**:
- Original Asch experiment: ~33% conformity rate
- We can tune parameters and observe LLM agent conformity curves

### 4.4 Main Experiment 3: Wisdom of Crowds

**Description**: Classic collective intelligence paradigm testing whether group estimates outperform individuals.

**Setup**:
```
Task types:
  1. Numerical estimation: Fermi problems ("How many piano tuners in Chicago?")
  2. Probability prediction: Event likelihood ("Will X happen next year?")
  3. Ranking tasks: Order options by criteria

Evaluation:
  - Individual error vs group error
  - Group error vs best individual error
  - Group performance under different sharing degrees
```

**Key Finding Target**:
- Prove existence of "optimal sharing regime": too little fails to integrate info, too much loses diversity
- This is LCN's core insight

### 4.5 Ablation Studies

| Variant | Modification | Purpose |
|---------|--------------|---------|
| **LCN-NoLocal** | Remove Local Attention | Verify peer influence contribution |
| **LCN-NoGroup** | Remove Group Level | Verify hierarchical structure necessity |
| **LCN-NoGlobal** | Remove Global Level | Verify social norm contribution |
| **LCN-Uniform** | Fixed α=1/3 | Verify adaptive Attention value |
| **LCN-FullShare** | All agents fully share | Verify hierarchical isolation necessity |

### 4.6 Analysis Experiments

**Analysis 1: Attention Weight Visualization**
- Question: How do agents allocate attention in different contexts?
- Method: Visualize α_local, α_group, α_global changes over time/tasks
- Expected findings: Context-dependent attention patterns

**Analysis 2: Consensus Convergence Dynamics**
- Question: How does consensus form?
- Method: Track KV-Cache similarity across rounds
- Metrics: Intra-group consistency, inter-group diversity

**Analysis 3: Diversity-Accuracy Trade-off**
- Question: How does sharing degree affect group performance?
- Method: Sweep temperature parameter, plot diversity-accuracy curve
- Expected finding: Inverted U-curve (moderate sharing is optimal)

### 4.7 Resource Estimation

| Experiment | Agents | Samples | Est. GPU Time |
|------------|--------|---------|---------------|
| Hidden Profile | 6-10 | 500 | ~8h |
| Asch Experiment | 6 | 300 | ~4h |
| Wisdom of Crowds | 10-20 | 500 | ~12h |
| Ablation Studies | 6-10 | 300×5 | ~20h |
| Analysis Experiments | 6-10 | 200 | ~6h |
| **Total** | - | - | **~50h** |

On 1-2 GPUs, core experiments completable in **1-2 weeks**.

---

## 5. Paper Structure

### 5.1 Title Candidates

**Primary**:
> **Latent Consensus Networks: Hierarchical Multi-Agent Collaboration via Cross-Level Attention in Latent Space**

**Alternatives**:
- "Beyond Token-Space Collaboration: Learning Hierarchical Consensus in Latent Space"
- "Cross-Level Latent Attention for Emergent Collective Intelligence in Multi-Agent Systems"

### 5.2 Paper Outline

```
1. Introduction                                    (~1.5 pages)
   - Motivation: Why hierarchical latent consensus?
   - Gap: Limitations of existing methods
   - Contribution: Three-point contribution statement

2. Related Work                                    (~1 page)
   - Multi-Agent LLM Systems
   - Latent Space Communication (LatentMAS, Coconut, etc.)
   - Collective Intelligence & Social Simulation

3. Method: Latent Consensus Networks               (~2.5 pages)
   3.1 Problem Formulation
   3.2 Hierarchical KV-Cache Architecture
   3.3 Cross-Level Attention Mechanism
   3.4 Consensus Formation Protocol

4. Experiments                                     (~3 pages)
   4.1 Experimental Setup
   4.2 Main Results: Hidden Profile Task
   4.3 Main Results: Asch Conformity Experiment
   4.4 Main Results: Wisdom of Crowds
   4.5 Ablation Studies

5. Analysis                                        (~1.5 pages)
   5.1 Attention Weight Dynamics
   5.2 Consensus Convergence Analysis
   5.3 Diversity-Accuracy Trade-off

6. Discussion & Conclusion                         (~0.5 pages)
   - Limitations
   - Broader Impact
   - Future Work

References                                         (~1 page)
Appendix                                           (supplementary)
```

**Total: ~9 pages + references + appendix** (NeurIPS format compliant)

### 5.3 Contribution Statement

We make the following contributions:

1. **Framework**: We propose Latent Consensus Networks (LCN), the first hierarchical latent-space framework for multi-agent consensus formation, featuring Local, Group, and Global KV-cache layers.

2. **Mechanism**: We design a Cross-Level Attention module that allows agents to adaptively balance peer influence, group consensus, and social norms without explicit text communication.

3. **Empirical Findings**: Through experiments on collective decision-making and classic social psychology paradigms, we demonstrate that (a) LCN achieves superior collective intelligence, (b) hierarchical structure is essential, and (c) there exists an optimal sharing regime that balances diversity and consensus.

### 5.4 Related Work Positioning

| Area | Representative Work | Our Distinction |
|------|---------------------|-----------------|
| **Multi-Agent LLM** | AutoGen, MetaGPT, AgentVerse | We use Latent not Text communication |
| **Latent Space MAS** | LatentMAS, Coconut | We introduce **hierarchical structure** and **Attention** |
| **Collective Intelligence** | Wisdom of Crowds, Superforecasting | We provide **computational framework** not just theory |
| **Social Simulation** | Generative Agents, CAMEL | We focus on **consensus mechanism** not role-playing |

---

## 6. Implementation Plan

### 6.1 Code Architecture

```
agent-social-simulacra/
├── LatentMAS/                    # Original LatentMAS (unchanged, as baseline)
│
├── lcn/                          # Our Latent Consensus Networks
│   ├── __init__.py
│   │
│   ├── core/                     # Core modules
│   │   ├── agent.py              # LCNAgent: hierarchy-aware agent
│   │   ├── kv_cache.py           # HierarchicalKVCache: 3-level KV management
│   │   ├── attention.py          # CrossLevelAttention: cross-level attention
│   │   └── consensus.py          # ConsensusProtocol: consensus formation flow
│   │
│   ├── models/                   # Model wrappers
│   │   ├── model_wrapper.py      # Extended LatentMAS ModelWrapper
│   │   └── latent_ops.py         # Latent space operations (realignment etc.)
│   │
│   ├── environments/             # Experiment environments
│   │   ├── base.py               # BaseEnvironment abstract class
│   │   ├── hidden_profile.py     # Hidden Profile Task
│   │   ├── asch_conformity.py    # Asch Conformity Experiment
│   │   └── wisdom_crowds.py      # Wisdom of Crowds
│   │
│   ├── configs/                  # Configuration files
│   │   ├── default.yaml
│   │   ├── hidden_profile.yaml
│   │   ├── asch.yaml
│   │   └── wisdom.yaml
│   │
│   └── utils/                    # Utilities
│       ├── metrics.py            # Evaluation metrics
│       ├── visualization.py      # Visualization
│       └── logging.py            # Logging
│
├── experiments/                  # Experiment scripts
│   ├── run_hidden_profile.py
│   ├── run_asch.py
│   ├── run_wisdom.py
│   ├── run_ablation.py
│   └── run_analysis.py
│
├── scripts/                      # Convenience scripts
│   ├── train.sh
│   ├── eval.sh
│   └── visualize.sh
│
├── notebooks/                    # Analysis notebooks
│   ├── attention_analysis.ipynb
│   ├── convergence_analysis.ipynb
│   └── diversity_accuracy.ipynb
│
├── docs/                         # Documentation
│   └── plans/
│       └── 2026-02-19-lcn-design.md
│
└── tests/                        # Unit tests
    ├── test_kv_cache.py
    ├── test_attention.py
    └── test_consensus.py
```

### 6.2 Core Class Designs

**HierarchicalKVCache**
```python
class HierarchicalKVCache:
    """Three-level KV-Cache management"""

    def __init__(self, num_groups, agents_per_group):
        self.local_caches = {}      # agent_id -> KVCache
        self.group_caches = {}      # group_id -> KVCache
        self.global_cache = None    # KVCache

    def update_local(self, agent_id, kv_cache):
        """Update single agent's Local Cache"""

    def aggregate_group(self, group_id):
        """Aggregate agents' Cache to Group Level"""

    def aggregate_global(self):
        """Aggregate all Groups' Cache to Global Level"""

    def get_all_levels(self, agent_id):
        """Get three levels of Cache accessible to agent"""
        return local, group, global
```

**ConsensusProtocol**
```python
class ConsensusProtocol:
    """Consensus formation protocol"""

    def __init__(self, model, kv_cache, attention, num_rounds):
        self.model = model
        self.kv_cache = kv_cache
        self.attention = attention
        self.num_rounds = num_rounds

    def run(self, agents, task):
        """Execute consensus formation flow"""
        for round in range(self.num_rounds):
            for agent in agents:
                # 1. Get three-level KV-Cache
                local, group, global_ = self.kv_cache.get_all_levels(agent.id)

                # 2. Cross-Level Attention fusion
                fused, weights = self.attention(agent.state, local, group, global_)

                # 3. Latent reasoning step
                new_state = self.model.latent_step(agent.state, fused)

                # 4. Update agent state and Local Cache
                agent.state = new_state
                self.kv_cache.update_local(agent.id, new_state)

            # 5. Aggregate Group and Global
            for group_id in self.kv_cache.group_ids:
                self.kv_cache.aggregate_group(group_id)
            self.kv_cache.aggregate_global()

        # 6. Final decision
        return self.make_decision(agents, task)
```

### 6.3 Development Milestones

**Phase 1: Foundation Framework (Week 1-2)**
| Task | Output | Est. Time |
|------|--------|-----------|
| Setup project structure | Code skeleton | 2 days |
| Implement HierarchicalKVCache | Core data structure | 3 days |
| Implement CrossLevelAttention | Core algorithm | 3 days |
| Implement ConsensusProtocol | Main flow | 2 days |
| Unit tests | Verify correctness | 2 days |

**Phase 2: Experiment Environments (Week 3-4)**
| Task | Output | Est. Time |
|------|--------|-----------|
| Hidden Profile environment | First experiment | 3 days |
| Asch Conformity environment | Second experiment | 2 days |
| Wisdom of Crowds environment | Third experiment | 2 days |
| Baseline integration | Comparison methods | 3 days |
| Pilot experiment validation | Verify idea feasibility | 4 days |

**Phase 3: Full Experiments (Week 5-8)**
| Task | Output | Est. Time |
|------|--------|-----------|
| Run main experiments | Main results | 1 week |
| Ablation studies | Component analysis | 1 week |
| Analysis experiments | Visualizations and insights | 1 week |
| Results compilation | Tables and figures | 3 days |

**Phase 4: Paper Writing (Week 9-12)**
| Task | Output | Est. Time |
|------|--------|-----------|
| Method section | Technical description | 4 days |
| Experiments section | Results presentation | 4 days |
| Introduction & Related Work | Narrative | 3 days |
| Analysis & Discussion | Insight summary | 3 days |
| Polish and review | Final version | 1 week |

### 6.4 Timeline Overview (Aligned with NeurIPS 2026)

```
2026
Feb  ├─ Week 1-2: Phase 1 Foundation Framework
     │
Mar  ├─ Week 3-4: Phase 2 Experiment Environments
     ├─ Week 5-6: Phase 3a Main Experiments
     │
Apr  ├─ Week 7-8: Phase 3b Ablation + Analysis
     ├─ Week 9-10: Phase 4a Paper Writing
     │
May  ├─ Week 11-12: Phase 4b Polish
     ├─ Week 13: Internal Review & Revisions
     └─ May 15 (estimated): NeurIPS Deadline ──→ Submit!
```

---

## 7. Risk Management

| Risk | Probability | Impact | Mitigation Strategy |
|------|-------------|--------|---------------------|
| KV-Cache merging ineffective | Medium | High | Prepare multiple aggregation methods (mean/attention/concat) |
| Insignificant experimental results | Medium | High | Run pilot experiments early, adjust direction if needed |
| Insufficient compute resources | Low | Medium | Validate with small model (7B) first, then scale |
| Time shortage | Medium | High | Prioritize core experiments, simplify analysis if needed |
| Reviewer questions novelty | Medium | High | Emphasize hierarchical structure and Attention innovation |

---

## 8. Success Criteria

### Minimum Viable Paper
- [ ] LCN outperforms LatentMAS on at least 2/3 main tasks
- [ ] Ablation studies show all components contribute
- [ ] At least one interesting insight (e.g., optimal sharing regime)

### Strong Paper
- [ ] LCN significantly outperforms all baselines on all tasks
- [ ] Reproduce human-like social phenomena (conformity curves match)
- [ ] Discover novel insight about collective intelligence
- [ ] Clean visualizations that tell compelling story

### Stretch Goals
- [ ] Theoretical analysis of consensus convergence
- [ ] Scale to 100+ agents
- [ ] Additional application domains

---

## Appendix A: Related Work References

### Latent Space MAS
- LatentMAS (Zou et al., 2025): arXiv:2511.20639
- Coconut (continuous thought): Related latent reasoning work
- KNN-LatentMAS, Hybrid-LatentMAS: Community extensions

### Multi-Agent LLM Systems
- AutoGen (Microsoft)
- MetaGPT
- AgentVerse
- CAMEL

### Collective Intelligence
- Wisdom of Crowds (Surowiecki, 2004)
- Superforecasting (Tetlock, 2015)
- Hidden Profile Task (Stasser & Titus, 1985)

### Social Psychology
- Asch Conformity Experiments (Asch, 1951)
- Social Influence Theory
- Groupthink (Janis, 1972)

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **KV-Cache** | Key-Value cache storing attention states in transformer models |
| **Latent Space** | The hidden representation space of neural networks |
| **Cross-Level Attention** | Attention mechanism operating across hierarchy levels |
| **Consensus** | Agreement or shared state among multiple agents |
| **Hidden Profile** | Information distribution where optimal solution requires sharing unique info |
| **Conformity** | Tendency to align with group opinion |

---

*Document created: 2026-02-19*
*Last updated: 2026-02-19*
