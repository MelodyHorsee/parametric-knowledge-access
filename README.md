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
| `run_eval.py` | Evaluate on TriviaQA, Natural Questions, HotpotQA, SimpleQA |
| `run_eval_math.py` | Evaluate on MATH |
| `Ex-Recall.py` | Refine model answer using GPT-5-mini |
| `bootstrap_ci.py` | Bootstrap confidence intervals on two results trace files |
| `sft_train.py` | SFT training on GPT-OSS-20B |
| `/rl` | RL training on GPT-OSS-20B |

## Datasets

- **TriviaQA**: Auto-downloaded from HuggingFace (`mandarjoshi/trivia_qa`)
- **Natural Questions**: Requires `NQ/NQ-open.dev.jsonl` ([download](https://github.com/efficientqa/nq-open))
- **HotpotQA**: Auto-downloaded from HuggingFace (`hotpotqa/hotpot_qa`)
- **SimpleQA**: Auto-downloaded from OpenAI blob storage
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
  - Choices: `triviaqa`, `nq`, `hotpotqa`, `simpleqa`

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

### Using a Shell Script

```bash
#!/bin/bash
python run_eval.py \
    --model "gpt-oss-20b" \
    --dataset "triviaqa" \
    --cues "no" \
    --thinking "yes" \
    --temperature 1 \
    --top_p 1 \
    --output_dir "results/TriviaQA/gpt-oss-20b"
```

## Output Files

Results are saved with the naming convention:

**Knowledge-access (run_eval.py):**
- GPT-5.2: `{dataset}_{model}_{cues}.json`
- Other models: `{dataset}_{model}_{cues}_{thinking}.json`
- With checkpoint: `{dataset}_{model}_{inference_name}_{cues}_{thinking}.json`

**MATH (run_eval_math.py):**
- GPT-5.2: `math_{model}_{cues}.json`
- Other models: `math_{model}_{cues}_{thinking}.json`
- With checkpoint: `math_{model}_{inference_name}_{cues}_{thinking}.json`

Examples:
- `triviaqa_gpt52_no_cues.json`
- `triviaqa_gptoss20b_no_cues_with_thinking.json`
- `triviaqa_gptoss20b_checkpoint1240_no_cues_with_thinking.json`
- `math_gptoss20b_no_cues_with_thinking.json`
- `math_gptoss20b_checkpoint1240_no_cues_with_thinking.json`
  
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

## Evaluation Metrics

- **Exact Match (EM)**: Prediction exactly matches ground truth after normalization
- **Recall**: Ground truth appears in prediction after normalization
- **Token Statistics**: Token length per response

---

# RL Training

Scripts for training GPT-OSS-20B with reinforcement learning on TriviaQA.

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

# SFT Training

Scripts for supervised fine-tuning GPT-OSS-20B on TriviaQA.

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/sft/sft_train.py` | Main SFT training script |

## Usage

```bash
python scripts/sft/sft_train.py \
    --batch_size 128 \
    --learning_rate 1e-4 \
    --num_epochs 1 \
    --seed 0 \
    --lora_rank 32 \
    --save_every 100 \
    --wandb_project triviaqa-sft
```

## Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_name` | Base model to train | `openai/gpt-oss-20b` |
| `--lora_rank` | LoRA rank | `32` |
| `--batch_size` | Batch size | `512` |
| `--learning_rate` | Learning rate | `2e-5` |
| `--num_epochs` | Number of epochs | `5` |
| `--max_length` | Max sequence length | `32768` |
| `--seed` | Random seed | `0` |
| `--save_every` | Checkpoint frequency | `100` |
| `--wandb_project` | W&B project name | `triviaqa-sft` |
| `--log_path` | Log directory | `tmp/triviaqa_sft/...` |

## Training Format

Build your own SFT dataset by modifying the `build_messages()` function in `sft_train.py`:

```python
def build_messages(question: str, answer: str) -> list[dict]:
    thinking_trace = f"Need answer: {answer}."
    formatted_answer = f"The answer is <answer>{answer}</answer>."

    return [
        {"role": "system", "content": "You will be given a question. Give your final answer in <answer></answer> tags."},
        {"role": "user", "content": question},
        {"role": "assistant", "thinking": thinking_trace, "content": formatted_answer},
    ]
```

The assistant message uses two fields for the `gpt_oss_low_reasoning` renderer:
- **thinking**: Goes to the analysis channel (reasoning trace)
- **content**: Goes to the final channel (visible output)

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
└── SimpleQA/
    └── gpt-oss-20b/
```