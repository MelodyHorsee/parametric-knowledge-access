# Parametric Knowledge Access

This repository contains scripts and results for the paper [Improving Parametric Knowledge Access in Reasoning Language Models](https://arxiv.org/abs/2602.22193).

## Overview

| Directory | Description |
|-----------|-------------|
| `scripts/` | Model evaluation on various datasets and RL training scripts |
| `results/` | Evaluation logs and results |

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

**tinker and tinker-cookbook**: Install from [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)

### Environment Variables

OpenAI and Tinker API keys are required:

```bash
export OPENAI_API_KEY="your-openai-key"
export TINKER_API_KEY="your-tinker-key"
```

---

# Evaluation

## Scripts Overview

| Script | Description |
|--------|-------------|
| `run_eval.py` | Evaluate on TriviaQA, Natural Questions, HotpotQA, SimpleQA, StrategyQA |
| `run_eval_math.py` | Evaluate on MATH and AIME |
| `Ex-Recall.py` | Refine extracted answers using GPT-5-mini (supports single-pass and batch submit/collect workflows, and pass@N multi-sample files) |
| `extract_thinking_tokens.py` | Extract and analyze thinking token counts from evaluation results |
| `sigtest.py` | McNemar's test for statistical significance between two result trace files |
| `reasoningsft_train.py` | SFT / Rejection-FT training on GPT-OSS-20B |
| `/rl` | RL training on GPT-OSS-20B, including RL-NoThinking variant |

## Datasets

- **TriviaQA**: Auto-downloaded from HuggingFace (`mandarjoshi/trivia_qa`)
- **Natural Questions**: Requires `NQ/NQ-open.dev.jsonl` ([download](https://github.com/efficientqa/nq-open))
- **HotpotQA**: Auto-downloaded from HuggingFace (`hotpotqa/hotpot_qa`)
- **SimpleQA**: Auto-downloaded from OpenAI blob storage
- **StrategyQA**: Requires `StrategyQA/strategyqa_train.json` ([download](https://github.com/eladsegal/strategyqa))
- **MATH**: Auto-downloaded from HuggingFace (`EleutherAI/hendrycks_math`)
- **AIME**: Auto-downloaded from HuggingFace. `AIME` pools three sources (`di-zhang-fdu/AIME_1983_2024`, `math-ai/aime25`, `MathArena/aime_2026`); `AIME25` and `AIME26` isolate just the 2025 or 2026 set.

## Models

1. **GPT-OSS-20B**: OpenAI's GPT-OSS-20B model (uses Tinker API)
2. **Qwen3-8B**: Alibaba's Qwen3-8B model (uses Tinker API)
3. **Nemotron3**: NVIDIA's Nemotron-3-Nano-30B-A3B model (uses Tinker API)

GPT-OSS-20B is the only model with multiple trained checkpoints in `results/` — see **Checkpoint Naming** below for what each suffix means.

## Checkpoint Naming

Result filenames encode which checkpoint was evaluated via a suffix after the model name (e.g. `triviaqa_gptoss20b_{suffix}_no_cues_with_thinking.json`). No suffix means the base model.

| Suffix | Meaning |
|--------|---------|
| *(none)* | Base model, no fine-tuning |
| `inference{N}` | RL checkpoint at training step N |
| `rlnothinking` | RL checkpoint trained with no thinking tokens (RL-NoThinking in paper) |
| `sftnothinking` | Direct SFT checkpoint — trained on a boilerplate `"Need answer: {answer}."` template rather than real reasoning (SFT-NoThinking in paper) |
| `rejectionft` | Rejection fine-tuning checkpoint — SFT trained on real reasoning traces from correct (recalled) RL rollouts (Rejection-FT in paper) |

The same four checkpoint types (`rlnothinking`, `sftnothinking`, `rejectionft`, plus the base model) are evaluated across all seven datasets; `inference{N}` checkpoints and the `qwen3-8b`/`nemotron3` checkpoint variants (e.g. `inference2420`, `nemotron1560`) vary by dataset.

## Eval Parameters

- `--model`: Model to evaluate
  - Choices: `gpt-oss-20b`, `qwen3-8b`, `nemotron3`

- `--dataset`: Dataset to use
  - `run_eval.py` choices: `triviaqa`, `nq`, `hotpotqa`, `simpleqa`, `strategyqa`, `all` (`all` runs all five)
  - `run_eval_math.py` choices: `MATH`, `AIME`

- `--thinking`: Enable thinking mode
  - Choices: `yes`, `no`
  - Default: `yes`

- `--temperature`: Sampling temperature
  - Default: `0.6`

- `--top_p`: Top-p sampling parameter
  - Default: `0.95`

- `--output_dir`: Output directory for results
  - Default: `results`

- `--checkpoint_uri`: Tinker URI for model checkpoint
  - Optional, uses base model if not provided

- `--inference_name`: Name for model checkpoint
  - Required if `--checkpoint_uri` is provided

- `--limit`: Limit the number of examples evaluated
  - Optional

- `--seed`: Shuffle the dataset with this seed before applying `--limit`
  - Optional

- `--num_samples` (`run_eval.py` only): Number of samples per question, for pass@N metrics
  - Default: `1`

## Usage

### Knowledge Access Dataset Evaluation Example

```bash
python run_eval.py \
    --model "gpt-oss-20b" \
    --dataset "triviaqa" \
    --thinking "yes" \
    --temperature 1 \
    --top_p 1 \
    --output_dir "results/TriviaQA/gpt-oss-20b"
```

### MATH / AIME Evaluation Example

```bash
python run_eval_math.py \
    --model "gpt-oss-20b" \
    --dataset "MATH" \
    --thinking "yes" \
    --temperature 1 \
    --top_p 1 \
    --output_dir "results/MATH/gpt-oss-20b"
```

### Loading from Trained Model Checkpoint

To evaluate a checkpoint instead of the base model:

```bash
# Knowledge access dataset with checkpoint
python run_eval.py \
    --model "gpt-oss-20b" \
    --dataset "triviaqa" \
    --thinking "yes" \
    --temperature 1 \
    --top_p 1 \
    --output_dir "results/TriviaQA/gpt-oss-20b" \
    --checkpoint_uri "your_checkpoint_uri" \
    --inference_name "inference1240"

# MATH/AIME benchmark with checkpoint
python run_eval_math.py \
    --model gpt-oss-20b \
    --dataset "MATH" \
    --thinking "yes" \
    --temperature 1 \
    --top_p 1 \
    --checkpoint_uri "your_checkpoint_uri" \
    --inference_name "inference1240"
```

| Argument | Description |
|----------|-------------|
| `--checkpoint_uri` | Tinker URI to the checkpoint weights |
| `--inference_name` | Optional name for the model checkpoint |

## Output Files

Results are saved with the naming convention:

- `{dataset}_{model}_{thinking}.json` (`run_eval_math.py`, and `run_eval.py` results predating the `--num_samples` option)
- `{dataset}_{model}_{thinking}_pass{N}.json` (`run_eval.py` with `--num_samples`)
- With checkpoint: `{dataset}_{model}_{inference_name}_{thinking}[_pass{N}].json`

Examples:
- `triviaqa_gptoss20b_with_thinking.json`
- `triviaqa_gptoss20b_rejectionft_with_thinking.json`
- `nq_nemotron3_nemotron1560_with_thinking_pass1.json`

**Note:** Most result files currently in this repository were generated before the `--cues` option was removed, so their filenames still carry the legacy `no_cues_` segment (e.g. `triviaqa_gptoss20b_no_cues_with_thinking.json`). New runs with the current scripts no longer include that segment, since there is no `--cues` flag to toggle.

**Note:** Result files larger than ~50MB are stored via [Git LFS](https://git-lfs.github.com) (tracked via `.gitattributes` for `results/**/*.json`) — run `git lfs install` before cloning to fetch them properly.

Each output file contains:
- `results`: List of individual predictions with metrics
- `aggregate_metrics`: Overall performance statistics
- `errors`: Any errors encountered during evaluation

**Knowledge-access metrics:**
- `exact_match_rate`: Exact match accuracy
- `recall_rate`: Recall accuracy

**MATH/AIME metrics:**
- `accuracy`: Proportion of correct answers

---

# Thinking Token Analysis

`extract_thinking_tokens.py` extracts thinking token counts from evaluation results and compares token usage between correctly (recalled) and incorrectly answered (not recalled) questions.

## Usage

```bash
# Single file (auto-detects model type from filename)
python scripts/extract_thinking_tokens.py results/TriviaQA/gpt-oss-20b/triviaqa_gptoss20b_no_cues_with_thinking.json

# Entire directory (recursively processes all result files)
python scripts/extract_thinking_tokens.py results/

# Save aggregated stats to JSON
python scripts/extract_thinking_tokens.py results/ --output thinking_stats.json

# Skip tokenizer (use character counts instead)
python scripts/extract_thinking_tokens.py results/ --no-tokenizer
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| `input` | Path to a results JSON file or directory |
| `--model-type` | Model type (`gpt-oss-20b`). Auto-detected from filename if not specified |
| `--no-tokenizer` | Skip tokenizer, only compute character counts |
| `--output` | Output file path for aggregated stats JSON |

## Output

For each file, reports:
- Total samples and average thinking tokens
- Average thinking tokens for correct vs incorrect answers

The script automatically pairs result files with their corresponding `ex_` refined recall files (from `Ex-Recall.py`) when available.

---

# SFT / Rejection-FT Training

Script for supervised fine-tuning GPT-OSS-20B on TriviaQA, supporting two modes.

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/reasoningsft_train.py` | SFT and Rejection-FT training script |

## Modes

### 1. Direct SFT (without `trajectory_path`)

Trains on TriviaQA with synthetic minimal thinking traces. The model learns to produce `"Need answer: {answer}."` as its reasoning, followed by the answer. Checkpoints from this mode are labeled `sftnothinking` in `results/`.

```bash
python scripts/reasoningsft_train.py \
    batch_size=32 \
    learning_rate=1e-5 \
    num_epochs=8 \
    seed=6 \
    wandb_project=triviaqa-sft
```

### 2. Rejection-FT (with `trajectory_path`)

Trains on correct trajectories from an RL-trained model. The model learns to reproduce full reasoning traces from examples where the RL model answered correctly (recall=True). Checkpoints from this mode are labeled `rejectionft` in `results/`.

```bash
python scripts/reasoningsft_train.py \
    trajectory_path=path/to/trajectories.json \
    batch_size=32 \
    learning_rate=1e-5 \
    num_epochs=8 \
    seed=6 \
    wandb_project=triviaqa-rejectionft
```

The trajectory file should be a JSON with a `results` array, where each item has `question`, `raw_prediction`, and `recall` fields. Only examples with `recall=True` are used.

Arguments are passed as `key=value` (no `config.` prefix, no `--`), since this script uses `chz.nested_entrypoint`.

## Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model_name` | Base model to train | `openai/gpt-oss-20b` |
| `lora_rank` | LoRA rank | `32` |
| `batch_size` | Batch size | `32` |
| `learning_rate` | Learning rate | `1e-5` |
| `num_epochs` | Number of epochs | `8` |
| `max_length` | Max sequence length | `32768` |
| `seed` | Random seed | `6` |
| `max_steps` | Cap training steps, cutting off mid-epoch if needed | `None` |
| `save_every` | Checkpoint frequency | `100` |
| `trace_every` | HTML example-trace frequency (mirrors RL training's rollout traces) | `100` |
| `num_examples_to_log` | Examples logged per HTML trace | `4` |
| `trajectory_path` | Path to correct trajectories JSON (enables Rejection-FT) | `None` |
| `val_path` | Path to validation set for NLL and generation eval | `None` |
| `eval_every` | Validation NLL eval frequency | `100` |
| `gen_eval_every` | Generation-based EM/Recall eval frequency | `500` |
| `nll_threshold` | Only train on examples with NLL below threshold | `None` |
| `wandb_project` | W&B project name | `triviaqa-sft` |
| `log_path` | Log directory | Auto-generated (`triviaqa_sft/{run_name}`) |

Training automatically filters out any training examples whose question also appears in `val_path`, to prevent train/val contamination.

## Training Format

The assistant message content is a list of typed parts for the `gpt_oss_low_reasoning` renderer:
- `{"type": "thinking", "thinking": ...}` — goes to the analysis channel (reasoning trace)
- `{"type": "text", "text": ...}` — goes to the final channel (visible output)

In Direct SFT mode, the thinking trace is synthetic: `"Need answer: {answer}."`. In Rejection-FT mode, the thinking trace is extracted from the RL model's correct trajectory.

# RL Training

Scripts for training GPT-OSS-20B with reinforcement learning on TriviaQA, including a no-thinking variant.

Our trained model used for evaluation in the paper is available at [melodyhorse/gpt-oss-20b-triviaqa-rl](https://huggingface.co/melodyhorse/gpt-oss-20b-triviaqa-rl).

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/rl/train.py` | Main RL training script |
| `scripts/rl/trivia_env.py` | TriviaQA environment for RL, including the no-thinking renderer mode |
| `scripts/rl/trivia_grading.py` | Grading utilities (EM, Recall, thinking extraction) |

## Usage

Standard RL training:

```bash
python scripts/rl/train.py \
    model_name=openai/gpt-oss-20b \
    lora_rank=32 \
    learning_rate=2e-5 \
    group_size=8 \
    groups_per_batch=32 \
    wandb_project=triviaqa-rl
```

### No-Thinking RL

To train with the thinking channel forced empty (producing the `rlnothinking` checkpoints in `results/`), pass `renderer_name=gpt_oss_no_reasoning`. Optionally add a `prefill` (e.g. `"The answer is"`) to force the answer's visible start:

```bash
python scripts/rl/train.py \
    model_name=openai/gpt-oss-20b \
    renderer_name=gpt_oss_no_reasoning \
    prefill="The answer is" \
    lora_rank=32 \
    learning_rate=2e-5 \
    group_size=8 \
    groups_per_batch=32 \
    wandb_project=triviaqa-rl-nothinking
```

## Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model_name` | Base model to train | `openai/gpt-oss-20b` |
| `lora_rank` | LoRA rank | `32` |
| `renderer_name` | Renderer to use; pass `gpt_oss_no_reasoning` for no-thinking RL | `gpt_oss_low_reasoning` |
| `prefill` | Optional forced text appended right after the thinking block | `None` |
| `learning_rate` | Learning rate | `2e-5` |
| `group_size` | Samples per group | `8` |
| `groups_per_batch` | Groups per batch | `32` |
| `max_tokens` | Max tokens per response | `2048` |
| `kl_penalty_coef` | KL penalty coefficient (builds a reference-model KL term when > 0) | `0.01` |
| `eval_every` | Evaluation frequency | `20` |
| `save_every` | Checkpoint frequency | `20` |
| `ttl_seconds` | TTL for periodic (non-final) checkpoints; the final checkpoint is always kept indefinitely | `604800` (7 days) |
| `max_steps` | Cap on total training steps | `1200` |
| `seed` | Random seed | `0` |
| `wandb_project` | W&B project name | `triviaqa-rl-nemotron3` |
| `log_path` | Log directory | Auto-generated (`logs/{run_name}`) |

Arguments are passed as `key=value` (no `--`), since this script uses `chz.entrypoint`.

---

# Results

The `results/` directory contains evaluation outputs organized by dataset and model. Large files (over ~50MB) are stored via Git LFS — see **Output Files** above.

## Directory Structure

```
results/
├── TriviaQA/
│   ├── gpt-oss-20b/
│   ├── nemotron3/
│   └── qwen3-8b/
├── NQ/
│   ├── gpt-oss-20b/
│   ├── nemotron3/
│   └── qwen3-8b/
├── HotpotQA/
│   ├── gpt-oss-20b/
│   ├── nemotron3/
│   └── qwen3-8b/
├── SimpleQA/
│   ├── gpt-oss-20b/
│   ├── nemotron3/
│   └── qwen3-8b/
├── StrategyQA/
│   └── gpt-oss-20b/
├── MATH/
│   ├── gpt-oss-20b/
│   ├── nemotron3/
│   └── qwen3-8b/
└── AIME/
    └── gpt-oss-20b/
```
