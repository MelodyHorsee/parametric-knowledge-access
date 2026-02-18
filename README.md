# Parametric Knowledge Access

This repository contains scripts and results for the paper [Paper Title](link-to-paper).

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
| `run_eval_math.py` | Evaluate on MATH |
| `Ex-Recall.py` | Refine model answer using GPT-5-mini |
| `extract_thinking_tokens.py` | Extract and analyze thinking token counts from evaluation results |
| `bootstrap_ci.py` | Bootstrap confidence intervals on two results trace files |
| `reasoningsft_train.py` | SFT / Reasoning-SFT training on GPT-OSS-20B |
| `/rl` | RL training on GPT-OSS-20B |

## Datasets

- **TriviaQA**: Auto-downloaded from HuggingFace (`mandarjoshi/trivia_qa`)
- **Natural Questions**: Requires `NQ/NQ-open.dev.jsonl` ([download](https://github.com/efficientqa/nq-open))
- **HotpotQA**: Auto-downloaded from HuggingFace (`hotpotqa/hotpot_qa`)
- **SimpleQA**: Auto-downloaded from OpenAI blob storage
- **StrategyQA**: Requires `StrategyQA/strategyqa_train.json` ([download](https://github.com/eladsegal/strategyqa))
- **MATH**: Auto-downloaded from HuggingFace (`EleutherAI/hendrycks_math`)

## Models

1. **GPT-5.2**: OpenAI's GPT-5.2 model (uses OpenAI API)
   - 2 variants: no-cues and with-cues (thinking cannot be disabled)

2. **GPT-OSS-20B**: OpenAI's GPT-OSS-20B model (uses Tinker API)
   - 3 variants: no-cues-with-thinking, with-cues-with-thinking, no-cues-no-thinking

3. **Olmo-3-7B-Think**: AllenAI's Olmo-3-7B-Think model (uses vLLM)
   - 3 variants: no-cues-with-thinking, with-cues-with-thinking, no-cues-no-thinking

4. **R1-Distill-Qwen-1.5B**: DeepSeek's R1-Distill-Qwen-1.5B model (uses vLLM)
   - 3 variants: no-cues-with-thinking, with-cues-with-thinking, no-cues-no-thinking

## Eval Parameters

- `--model`: Model to evaluate
  - Choices: `gpt-5.2`, `gpt-oss-20b`, `olmo-3-7b-think`, `r1-distill-qwen-1.5b`

- `--dataset`: Dataset to use (run_eval.py only)
  - Choices: `triviaqa`, `nq`, `hotpotqa`, `simpleqa`, `strategyqa`, `all`

- `--cues`: Use thinking cues ("Think step-by-step")
  - Choices: `yes`, `no`
  - Default: `no`

- `--thinking`: Enable thinking mode
  - Choices: `yes`, `no`
  - Default: `yes`
  - Note: Not applicable for `gpt-5.2` (always enabled)

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

## Usage

### Knowledge Access Dataset Evaluation Example

```bash
python run_eval.py \
    --model "gpt-oss-20b" \
    --dataset "triviaqa" \
    --cues "no" \
    --thinking "yes" \
    --temperature 1 \
    --top_p 1 \
    --output_dir "results/TriviaQA/gpt-oss-20b"
```

### MATH Evaluation Example

```bash
python run_eval_math.py \
    --model "gpt-oss-20b" \
    --cues "no" \
    --thinking "yes" \
    --temperature 1 \
    --top_p 1 \
    --output_dir "results/MATH/gpt-oss-20b"
```

### Loading from Trained Model Checkpoint (GPT-OSS-20B only)

To evaluate a checkpoint instead of the base model:

```bash
# Knowledge access dataset with checkpoint
python run_eval.py \
    --model "gpt-oss-20b" \
    --dataset "triviaqa" \
    --cues "no" \
    --thinking "yes" \
    --temperature 1 \
    --top_p 1 \
    --output_dir "results/TriviaQA/gpt-oss-20b" \
    --checkpoint_uri "your_checkpoint_uri" \
    --inference_name "inference1240"

# MATH benchmark with checkpoint
python run_eval_math.py \
    --model gpt-oss-20b \
    --cues "no" \
    --thinking "yes" \
    --temperature 1 \
    --top_p 1 \
    --checkpoint_uri "your_checkpoint_uri" \
    --inference_name "inference1240" \
```

| Argument | Description |
|----------|-------------|
| `--checkpoint_uri` | Tinker URI to the checkpoint weights |
| `--inference_name` | Optional name for the model checkpoint |

## Output Files

Results are saved with the naming convention:

- GPT-5.2: `{dataset}_{model}_{cues}.json`
- Other models: `{dataset}_{model}_{cues}_{thinking}.json`
- With checkpoint: `{dataset}_{model}_{inference_name}_{cues}_{thinking}.json`

Examples:
- `triviaqa_gpt52_no_cues.json`
- `triviaqa_gptoss20b_no_cues_with_thinking.json`
- `triviaqa_gptoss20b_checkpoint1240_no_cues_with_thinking.json`
  
**Note:** Two TriviaQA result files for Olmo-3-7B-Think (`triviaqa_olmo3_no_cues_with_thinking.json` and `triviaqa_olmo3_with_cues_with_thinking.json`) are too large and thus are not included in this repository.

Each output file contains:
- `results`: List of individual predictions with metrics
- `aggregate_metrics`: Overall performance statistics
- `errors`: Any errors encountered during evaluation

**Knowledge-access metrics:**
- `exact_match_rate`: Exact match accuracy
- `recall_rate`: Recall accuracy

**MATH metrics:**
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
| `--model-type` | Model type (`gpt-oss-20b`, `olmo`, `r1-qwen`). Auto-detected from filename if not specified |
| `--no-tokenizer` | Skip tokenizer, only compute character counts |
| `--output` | Output file path for aggregated stats JSON |

## Output

For each file, reports:
- Total samples and average thinking tokens
- Average thinking tokens for correct vs incorrect answers

The script automatically pairs result files with their corresponding `ex_` refined recall files (from `Ex-Recall.py`) when available.

---

# SFT / Reasoning-SFT Training

Script for supervised fine-tuning GPT-OSS-20B on TriviaQA, supporting two modes.

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/reasoningsft_train.py` | SFT and Reasoning-SFT training script |

## Modes

### 1. Direct SFT (without `trajectory_path`)

Trains on TriviaQA with synthetic minimal thinking traces. The model learns to produce `"Need answer: {answer}."` as its reasoning, followed by the answer.

```bash
python scripts/reasoningsft_train.py \
    config.batch_size=32 \
    config.learning_rate=1e-5 \
    config.num_epochs=8 \
    config.seed=6 \
    config.wandb_project=triviaqa-sft
```

### 2. Reasoning-SFT (with `trajectory_path`)

Trains on correct trajectories from an RL-trained model. The model learns to reproduce full reasoning traces from examples where the RL model answered correctly (recall=True).

```bash
python scripts/reasoningsft_train.py \
    config.trajectory_path=path/to/trajectories.json \
    config.batch_size=32 \
    config.learning_rate=1e-5 \
    config.num_epochs=8 \
    config.seed=6 \
    config.wandb_project=triviaqa-reasoning-sft
```

The trajectory file should be a JSON with a `results` array, where each item has `question`, `raw_prediction`, and `recall` fields. Only examples with `recall=True` are used.

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
| `save_every` | Checkpoint frequency | `100` |
| `trajectory_path` | Path to correct trajectories JSON (enables Reasoning-SFT) | `None` |
| `val_path` | Path to validation set for NLL and generation eval | `None` |
| `eval_every` | Validation NLL eval frequency | `100` |
| `gen_eval_every` | Generation-based EM/Recall eval frequency | `500` |
| `nll_threshold` | Only train on examples with NLL below threshold | `None` |
| `wandb_project` | W&B project name | `triviaqa-sft` |
| `log_path` | Log directory | Auto-generated |

## Training Format

The assistant message uses two fields for the `gpt_oss_low_reasoning` renderer:
- **thinking**: Goes to the analysis channel (reasoning trace)
- **content**: Goes to the final channel (visible output)

In Direct SFT mode, the thinking trace is synthetic: `"Need answer: {answer}."`. In Reasoning-SFT mode, the thinking trace is extracted from the RL model's correct trajectory.

# RL Training

Scripts for training GPT-OSS-20B with reinforcement learning on TriviaQA.

Our trained model used for evaluation in the paper is available at [melodyhorse/gpt-oss-20b-triviaqa-rl](https://huggingface.co/melodyhorse/gpt-oss-20b-triviaqa-rl).

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/rl/train.py` | Main RL training script |
| `scripts/rl/trivia_env.py` | TriviaQA environment for RL |
| `scripts/rl/trivia_grading.py` | Grading utilities (EM, Recall) |

## Usage

```bash
python scripts/rl/train.py \
    --model_name openai/gpt-oss-20b \
    --lora_rank 32 \
    --learning_rate 2e-5 \
    --group_size 8 \
    --groups_per_batch 32 \
    --wandb_project triviaqa-rl
```

## Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_name` | Base model to train | `openai/gpt-oss-20b` |
| `--lora_rank` | LoRA rank | `32` |
| `--learning_rate` | Learning rate | `2e-5` |
| `--group_size` | Samples per group | `8` |
| `--groups_per_batch` | Groups per batch | `32` |
| `--max_tokens` | Max tokens per response | `1028` |
| `--kl_penalty_coef` | KL penalty coefficient | `0.01` |
| `--eval_every` | Evaluation frequency | `20` |
| `--save_every` | Checkpoint frequency | `20` |
| `--seed` | Random seed | `0` |
| `--wandb_project` | W&B project name | `triviaqa-rl` |
| `--log_path` | Log directory | `/tmp/tinker-examples/...` |

---

# Results

The `results/` directory contains evaluation outputs organized by dataset and model.

## Directory Structure

```
results/
├── TriviaQA/
│   ├── gpt-5.2/
│   ├── gpt-oss-20b/
│   ├── olmo-3-7b-think/
│   └── r1-distill-qwen-1.5b/
├── NQ/
│   ├── gpt-5.2/
│   ├── gpt-oss-20b/
│   ├── olmo-3-7b-think/
│   └── r1-distill-qwen-1.5b/
├── MATH/
│   ├── gpt-5.2/
│   ├── gpt-oss-20b/
│   ├── olmo-3-7b-think/
│   └── r1-distill-qwen-1.5b/
├── HotpotQA/
│   └── gpt-oss-20b/
├── SimpleQA/
│   └── gpt-oss-20b/
└── StrategyQA/
    └── gpt-oss-20b/
```