# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements **Latent Consensus Networks (LCN)** - a framework for simulating social consensus formation through shared latent space (KV-Cache). It extends the LatentMAS multi-agent reasoning framework to model social phenomena like opinion dynamics, norm emergence, and collective intelligence.

**Research Goal:** NeurIPS 2026 submission on "Implicit Social Consensus via Shared Latent Space"

## Architecture

### Two-Layer Structure

1. **LatentMAS (submodule)**: The original multi-agent reasoning framework for math/logic problems
   - Located in `LatentMAS/`
   - Provides KV-Cache manipulation and latent space collaboration primitives
   - Entry point: `LatentMAS/run.py`

2. **LCN (new framework)**: Social simulation layer built on top of LatentMAS
   - Located in `lcn/`
   - Implements hierarchical KV-Cache (Local/Group/Global) with Cross-Level Attention

### LCN Core Components (`lcn/core/`)

- **HierarchicalKVCache** (`kv_cache.py`): Three-level cache management (Local → Group → Global) with mean-pooling aggregation
- **CrossLevelAttention** (`attention.py`): Attention mechanism for fusing information across hierarchy levels
- **LCNAgent** (`agent.py`): Agent with persona, group membership, and hidden state
- **ConsensusProtocol** (`consensus.py`): Orchestrates multi-round consensus formation

### Key Design Decisions

- KV-Cache merging uses **mean pooling** (configurable in future)
- Agents are organized into **groups** for hierarchical social structure
- Cross-level attention uses **temperature-controlled softmax** for tunable influence sharpness

## Commands

### Install Dependencies

```bash
pip install -e ".[dev]"
```

### Run Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_kv_cache.py -v

# With coverage
pytest tests/ --cov=lcn --cov-report=term-missing
```

### Run LatentMAS Experiments

```bash
cd LatentMAS

# Baseline (single model)
python run.py --method baseline --model_name Qwen/Qwen3-14B --task gsm8k --max_samples -1 --max_new_tokens 2048

# TextMAS (text-based multi-agent)
python run.py --method text_mas --model_name Qwen/Qwen3-14B --task gsm8k --prompt sequential --max_samples -1 --max_new_tokens 2048

# LatentMAS (latent-space multi-agent)
python run.py --method latent_mas --model_name Qwen/Qwen3-14B --task gsm8k --prompt sequential --max_samples -1 --max_new_tokens 2048

# With latent space realignment
python run.py --method latent_mas --model_name Qwen/Qwen3-14B --task gsm8k --prompt sequential --latent_space_realign --max_new_tokens 2048
```

### Key LatentMAS Arguments

- `--latent_steps`: Number of latent reasoning steps (0-80)
- `--prompt`: MAS architecture (`sequential` or `hierarchical`)
- `--task`: Dataset (`gsm8k`, `aime2024`, `aime2025`, `gpqa`, `arc_easy`, `arc_challenge`, `mbppplus`, `humanevalplus`, `medqa`)
- `--use_vllm`: Enable vLLM backend for faster inference

## Implementation Plan

See `docs/plans/2026-02-20-lcn-implementation-plan.md` for the detailed TDD implementation plan. The plan follows strict test-driven development: write failing test → implement → verify → commit.

## Tech Stack

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.35+
- Qwen models (Qwen3-4B, Qwen3-14B)
- pytest for testing
