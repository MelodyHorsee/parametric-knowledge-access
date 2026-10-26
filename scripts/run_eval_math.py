#!/usr/bin/env python3
import re
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import tinker
from tinker.types import SamplingParams as TinkerSamplingParams
from tinker_cookbook import renderers, tokenizer_utils
from tinker_cookbook.renderers import Message, Role
from tinker_cookbook.renderers.gpt_oss import GptOssRenderer
from datasets import load_dataset as hf_load_dataset

# MATH dataset categories
MATH_CONFIGS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus"
]

# from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces and LaTeX spacing commands
    string = string.replace("\\!", "")
    string = string.replace("\\,", "")
    string = string.replace("\\;", "")
    string = string.replace("\\:", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace(r"\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string


class GPTOSSEvaluator:
    def __init__(self, temperature, top_p, checkpoint_uri=None, inference_name=None):
        """
        Args:
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            checkpoint_uri: Optional Tinker URI to load from checkpoint (e.g., "tinker://.../:train:0/weights/001240")
            inference_name: Optional name for inference client when loading from checkpoint
        """
        self.tokenizer = tokenizer_utils.get_tokenizer('openai/gpt-oss-20b')
        self.service_client = tinker.ServiceClient()

        # Load from checkpoint if provided, otherwise use base model
        if checkpoint_uri and inference_name:
            training_client = self.service_client.create_training_client_from_state(checkpoint_uri)
            self.sampling_client = training_client.save_weights_and_get_sampling_client(inference_name)
        else:
            self.sampling_client = self.service_client.create_sampling_client(base_model='openai/gpt-oss-20b')

        self.temperature = temperature
        self.top_p = top_p
        self.GptOssRenderer = GptOssRenderer
        self.SamplingParams = TinkerSamplingParams
        self.renderers = renderers

    def build_prompt(self, question):
        system_content = "You will be given a question. Give your final answer in \\boxed{}."
        return [
            {'role': 'system', 'content': system_content},
            {'role': 'user', 'content': question}
        ]

    def generate(self, question, use_thinking):
        messages = self.build_prompt(question)

        # GptOssRenderer's default generation suffix is just "<|start|>assistant" —
        # it does NOT auto-open a thinking channel (that's the model's own choice).
        # So no-thinking mode is achieved via prefill, not a custom renderer
        # subclass: same pattern as eval-recall.py / trivia_env.py.
        prefill = None
        if use_thinking:
            renderer = self.renderers.get_renderer('gpt_oss_low_reasoning', self.tokenizer)
        else:
            renderer = self.GptOssRenderer(self.tokenizer, use_system_prompt=True, reasoning_effort="low")
            prefill = "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>"

        stop_sequences = renderer.get_stop_sequences()
        prompt = renderer.build_generation_prompt(messages, prefill=prefill)
        prompt_len = prompt.length if hasattr(prompt, 'length') else len(prompt)
        max_tokens = min(32000, 32768 - prompt_len)
        sampling_params = self.SamplingParams(max_tokens=max_tokens, temperature=self.temperature, top_p=self.top_p, stop=stop_sequences)
        output = self.sampling_client.sample(prompt, sampling_params=sampling_params, num_samples=1).result()
        seq = output.sequences[0]

        # Full raw output including analysis + final channels, same as eval-recall.py's GPTOSSEvaluator —
        # decoding the raw token sequence directly instead of going through parse_response's
        # structured ContentPart split, which silently drops thinking content when flattened.
        text = self.tokenizer.decode(seq.tokens)

        # Extract thinking tokens: content between <|channel|>analysis<|message|> and <|end|>
        thinking_start = text.find("<|channel|>analysis<|message|>")
        thinking_end = text.find("<|end|>", thinking_start) if thinking_start != -1 else -1

        if thinking_start == -1:
            # Model chose not to open a thinking channel at all — the response
            # is already complete; there's nothing to force-close.
            return text, len(seq.tokens), 0

        if thinking_end != -1:
            thinking_text = text[thinking_start + len("<|channel|>analysis<|message|>"):thinking_end]
            thinking_token_count = len(self.tokenizer.encode(thinking_text, add_special_tokens=False))
            return text, len(seq.tokens), thinking_token_count

        # Thinking was opened but never closed — genuinely truncated mid-thinking.
        # Force-close it and let the model continue straight to its boxed answer.
        thinking_token_count = len(seq.tokens)
        force_suffix = "<|end|><|start|>assistant<|channel|>final<|message|>"
        force_suffix_tokens = self.tokenizer.encode(force_suffix, add_special_tokens=False)
        forced_prompt = tinker.ModelInput.from_ints(
            list(prompt.to_ints()) + list(seq.tokens) + force_suffix_tokens
        )
        forced_params = self.SamplingParams(
            max_tokens=2048, temperature=self.temperature, top_p=self.top_p, stop=stop_sequences
        )
        forced_output = self.sampling_client.sample(forced_prompt, sampling_params=forced_params, num_samples=1).result()
        forced_text = self.tokenizer.decode(forced_output.sequences[0].tokens)
        total_token_count = len(seq.tokens) + len(forced_output.sequences[0].tokens)
        return text + force_suffix + forced_text, total_token_count, thinking_token_count


class Qwen3Evaluator:
    def __init__(self, temperature, top_p, checkpoint_uri=None, inference_name=None):
        self.tokenizer = tokenizer_utils.get_tokenizer('Qwen/Qwen3-8B')
        self.service_client = tinker.ServiceClient()

        if checkpoint_uri and inference_name:
            training_client = self.service_client.create_training_client_from_state(checkpoint_uri)
            self.sampling_client = training_client.save_weights_and_get_sampling_client(inference_name)
        else:
            self.sampling_client = self.service_client.create_sampling_client(base_model='Qwen/Qwen3-8B')

        self.temperature = temperature
        self.top_p = top_p
        self.SamplingParams = TinkerSamplingParams

    def build_prompt(self, question):
        system_content = "You will be given a question. Give your final answer in \\boxed{}."
        return [
            {'role': 'system', 'content': system_content},
            {'role': 'user', 'content': question}
        ]

    def generate(self, question, use_thinking):
        messages = self.build_prompt(question)

        renderer_name = 'qwen3' if use_thinking else 'qwen3_disable_thinking'
        renderer = renderers.get_renderer(renderer_name, self.tokenizer)

        stop_sequences = renderer.get_stop_sequences()
        sampling_params = self.SamplingParams(max_tokens=8092, temperature=self.temperature, top_p=self.top_p, stop=stop_sequences)

        prompt = renderer.build_generation_prompt(messages)
        output = self.sampling_client.sample(prompt, sampling_params=sampling_params, num_samples=1).result()
        seq = output.sequences[0]
        raw_full = self.tokenizer.decode(seq.tokens)

        if not use_thinking:
            # The <think> opening tag is part of the prompt (generation prefix) for
            # the disable-thinking renderer, never part of the sampled completion —
            # so there's nothing to detect or retry here.
            return raw_full, len(seq.tokens), 0

        # Unlike the disable-thinking renderer, <think> is NOT prefilled in thinking
        # mode (verified: the 'qwen3' renderer's prompt just ends at the assistant
        # turn marker) — the model decides whether to open it at all. So a missing
        # </think> is only a truncation if <think> was actually opened.
        think_start = raw_full.find("<think>")
        think_end = raw_full.find("</think>", think_start) if think_start != -1 else -1

        if think_start == -1:
            # Model chose not to open thinking at all — nothing to force-close.
            return raw_full, len(seq.tokens), 0

        if think_end != -1:
            thinking_text = raw_full[think_start + len("<think>"):think_end]
            thinking_token_count = len(self.tokenizer.encode(thinking_text, add_special_tokens=False))
            return raw_full, len(seq.tokens), thinking_token_count

        # Thinking was opened but never closed — genuinely truncated mid-thinking.
        # Force-close it and let the model continue straight to its \boxed{} answer.
        thinking_token_count = len(seq.tokens)
        force_suffix = "\n</think>\n"
        force_suffix_tokens = self.tokenizer.encode(force_suffix, add_special_tokens=False)
        forced_prompt = tinker.ModelInput.from_ints(
            list(prompt.to_ints()) + list(seq.tokens) + force_suffix_tokens
        )
        forced_params = self.SamplingParams(
            max_tokens=2048, temperature=self.temperature, top_p=self.top_p, stop=stop_sequences
        )
        forced_output = self.sampling_client.sample(forced_prompt, sampling_params=forced_params, num_samples=1).result()
        forced_text = self.tokenizer.decode(forced_output.sequences[0].tokens)
        total_token_count = len(seq.tokens) + len(forced_output.sequences[0].tokens)
        return raw_full + force_suffix + forced_text, total_token_count, thinking_token_count


class Nemotron3Evaluator:
    def __init__(self, temperature, top_p, checkpoint_uri=None, inference_name=None):
        self.model_name = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
        self.tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
        self.service_client = tinker.ServiceClient()

        if checkpoint_uri and inference_name:
            training_client = self.service_client.create_training_client_from_state(checkpoint_uri)
            self.sampling_client = training_client.save_weights_and_get_sampling_client(inference_name)
        else:
            self.sampling_client = self.service_client.create_sampling_client(base_model=self.model_name)

        self.temperature = temperature
        self.top_p = top_p
        self.SamplingParams = TinkerSamplingParams

    def build_prompt(self, question):
        system_content = "You will be given a question. Give your final answer in \\boxed{}."
        return [
            {'role': 'system', 'content': system_content},
            {'role': 'user', 'content': question}
        ]

    def generate(self, question, use_thinking):
        messages = self.build_prompt(question)

        # Nemotron3Renderer(Qwen3_5Renderer): <think>\n...\n</think>\n formatting.
        # Disable variant prepends <think></think> to skip thinking.
        renderer_name = 'nemotron3' if use_thinking else 'nemotron3_disable_thinking'
        renderer = renderers.get_renderer(renderer_name, self.tokenizer)

        stop_sequences = renderer.get_stop_sequences()
        sampling_params = self.SamplingParams(max_tokens=8092, temperature=self.temperature, top_p=self.top_p, stop=stop_sequences)

        prompt = renderer.build_generation_prompt(messages)
        output = self.sampling_client.sample(prompt, sampling_params=sampling_params, num_samples=1).result()
        seq = output.sequences[0]
        raw_full = self.tokenizer.decode(seq.tokens)

        if not use_thinking:
            return raw_full, len(seq.tokens), 0

        think_end = raw_full.find("</think>")
        if think_end != -1:
            thinking_text = raw_full[:think_end]
            thinking_token_count = len(self.tokenizer.encode(thinking_text, add_special_tokens=False))
            return raw_full, len(seq.tokens), thinking_token_count

        # Max tokens hit mid-thinking — force close thinking and let the model
        # continue straight to its \boxed{} answer.
        thinking_token_count = len(seq.tokens)
        force_suffix = "\n</think>\n"
        force_suffix_tokens = self.tokenizer.encode(force_suffix, add_special_tokens=False)
        forced_prompt = tinker.ModelInput.from_ints(
            list(prompt.to_ints()) + list(seq.tokens) + force_suffix_tokens
        )
        forced_params = self.SamplingParams(
            max_tokens=2048, temperature=self.temperature, top_p=self.top_p, stop=stop_sequences
        )
        forced_output = self.sampling_client.sample(forced_prompt, sampling_params=forced_params, num_samples=1).result()
        forced_text = self.tokenizer.decode(forced_output.sequences[0].tokens)
        total_token_count = len(seq.tokens) + len(forced_output.sequences[0].tokens)
        return raw_full + force_suffix + forced_text, total_token_count, thinking_token_count


def prepare_MATH_from_hf(split="test"):
    """Prepare MATH dataset from HuggingFace"""
    print(f"Loading MATH dataset from HuggingFace (split: {split})...")

    all_problems = []

    for cfg in MATH_CONFIGS:
        print(f"  Loading {cfg}...")
        ds = hf_load_dataset("EleutherAI/hendrycks_math", cfg, split=split)
        for item in ds:
            all_problems.append({
                "problem": item["problem"],
                "type": cfg,
                "solution": item["solution"],
            })
        print(f"    {len(ds)} problems")

    print(f"\nTotal dataset size: {len(all_problems)}")
    return all_problems

def prepare_AIME_from_hf():
    """Prepare AIME dataset from all three HuggingFace sources"""
    all_problems = []

    # AIME 1983-2024
    print("Loading AIME 1983-2024...")
    ds = hf_load_dataset("di-zhang-fdu/AIME_1983_2024", split="train")
    for item in ds:
        all_problems.append({
            "problem": item["Question"],
            "solution": item["Answer"],
            "type": item["Year"],
        })
    print(f"    {len(ds)} problems")

    # AIME 2025
    print("Loading AIME 2025...")
    ds = hf_load_dataset("math-ai/aime25", split="test")
    for item in ds:
        all_problems.append({
            "problem": item["problem"],
            "solution": str(item["answer"]),
            "type": "2025",
        })
    print(f"    {len(ds)} problems")

    # AIME 2026
    print("Loading AIME 2026...")
    ds = hf_load_dataset("MathArena/aime_2026", split="train")
    for item in ds:
        all_problems.append({
            "problem": item["problem"],
            "solution": str(item["answer"]),
            "type": "2026",
        })
    print(f"    {len(ds)} problems")

    print(f"\nTotal dataset size: {len(all_problems)}")
    return all_problems

def prepare_AIME25_from_hf():
    """Prepare AIME 2025 dataset from HuggingFace"""
    print("Loading AIME 2025...")
    all_problems = []
    ds = hf_load_dataset("math-ai/aime25", split="test")
    for item in ds:
        all_problems.append({
            "problem": item["problem"],
            "solution": str(item["answer"]),
            "type": "2025",
        })
    print(f"    {len(ds)} problems")
    print(f"\nTotal dataset size: {len(all_problems)}")
    return all_problems

def prepare_AIME26_from_hf():
    """Prepare AIME 2026 dataset from HuggingFace"""
    print("Loading AIME 2026...")
    all_problems = []
    ds = hf_load_dataset("MathArena/aime_2026", split="train")
    for item in ds:
        all_problems.append({
            "problem": item["problem"],
            "solution": str(item["answer"]),
            "type": "2026",
        })
    print(f"    {len(ds)} problems")
    print(f"\nTotal dataset size: {len(all_problems)}")
    return all_problems

def run_evaluation(model_name, dataset, sample, use_thinking, temperature, top_p, output_dir, checkpoint_uri=None, inference_name=None):
    # Build output filename
    thinking_str = "with_thinking" if use_thinking else "no_thinking"

    model_short = {
        "gpt-oss-20b": "gptoss20b",
        "qwen3-8b": "qwen38b",
        "nemotron3": "nemotron3",
    }[model_name]

    # Add inference_name to model_short if checkpoint is being used
    if inference_name:
        model_short = f"{model_short}_{inference_name}"

    dataset_lower = dataset.lower()
    output_file = Path(output_dir) / f"{dataset_lower}_{model_short}_{thinking_str}.json"

    print(f"\nRunning evaluation: {model_name} on {dataset}")
    print(f"  Dataset size: {len(sample)} problems")
    print(f"  Thinking: {use_thinking}")
    print(f"  Temperature: {temperature}, Top-p: {top_p}")
    print(f"  Output: {output_file}")

    # Initialize evaluator
    if model_name == "gpt-oss-20b":
        evaluator = GPTOSSEvaluator(temperature, top_p, checkpoint_uri, inference_name)
    elif model_name == "qwen3-8b":
        evaluator = Qwen3Evaluator(temperature, top_p, checkpoint_uri, inference_name)
    elif model_name == "nemotron3":
        evaluator = Nemotron3Evaluator(temperature, top_p, checkpoint_uri, inference_name)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Resume from existing file if present
    results = []
    errors = []
    token_lengths = []
    correct_token_lengths = []
    incorrect_token_lengths = []
    thinking_token_lengths = []
    success = 0

    if output_file.exists():
        with open(output_file) as f:
            existing = json.load(f)
        results = existing.get("results", [])
        errors = existing.get("errors", [])
        for r in results:
            tc = r.get("num_tokens", 0)
            token_lengths.append(tc)
            thinking_token_lengths.append(r.get("num_thinking_tokens", 0))
            if r["correct"]:
                success += 1
                correct_token_lengths.append(tc)
            else:
                incorrect_token_lengths.append(tc)
        print(f"  Resuming from {len(results)} completed problems, {len(errors)} errors.")

    total_problems = len(sample)

    from collections import defaultdict
    completed_counts = defaultdict(int)
    for r in results:
        completed_counts[r["problem"]] += 1
    occurrence = defaultdict(int)
    remaining = []
    for item in sample:
        p = item["problem"]
        if occurrence[p] < completed_counts[p]:
            occurrence[p] += 1
        else:
            remaining.append(item)
    sample = remaining
    print(f"  Remaining: {len(sample)} problems.")

    for idx, item in tqdm(enumerate(sample), total=len(sample)):
        try:
            text, token_count, thinking_token_count = evaluator.generate(item["problem"], use_thinking)

            try:
                # Extract prediction
                pred_boxed = last_boxed_only_string(text)
                pred = remove_boxed(pred_boxed) if pred_boxed else None
            except Exception as e:
                pred = None
                errors.append({
                    "problem": item["problem"],
                    "error": f"pred_extraction: {str(e)}"
                })

            try:
                # Extract ground truth
                gt_boxed = last_boxed_only_string(item["solution"])
                if gt_boxed:
                    gt = remove_boxed(gt_boxed)
                else:
                    gt = item["solution"].strip()
            except Exception as e:
                gt = item["solution"].strip()
                errors.append({
                    "problem": item["problem"],
                    "error": f"gt_extraction: {str(e)}"
                })

            correct = is_equiv(pred, gt)

            token_lengths.append(token_count)
            thinking_token_lengths.append(thinking_token_count)
            if correct:
                correct_token_lengths.append(token_count)
                success += 1
            else:
                incorrect_token_lengths.append(token_count)

            results.append({
                "problem": item["problem"],
                "solution": item["solution"],
                "type": item.get("type"),
                "raw_prediction": text,
                "extracted_prediction": pred,
                "extracted_gt": gt,
                "num_tokens": token_count,
                "num_thinking_tokens": thinking_token_count,
                "correct": correct,
            })

        except Exception as e:
            errors.append({
                "problem": item["problem"],
                "error": str(e),
            })

        # Save after each question
        aggregate_metrics = {
            "total_problems": total_problems,
            "correct": success,
            "accuracy": success / total_problems,
            "extraction_errors": len(errors),
            "avg_tokens": sum(token_lengths) / len(token_lengths) if token_lengths else 0,
            "avg_tokens_correct": sum(correct_token_lengths) / len(correct_token_lengths) if correct_token_lengths else 0,
            "avg_tokens_incorrect": sum(incorrect_token_lengths) / len(incorrect_token_lengths) if incorrect_token_lengths else 0,
            "avg_thinking_tokens": sum(thinking_token_lengths) / len(thinking_token_lengths) if thinking_token_lengths else 0,
            "num_correct": len(correct_token_lengths),
            "num_incorrect": len(incorrect_token_lengths),
        }

        with open(output_file, "w") as f:
            json.dump(
                {
                    "results": results,
                    "aggregate_metrics": aggregate_metrics,
                    "errors": errors,
                },
                f,
                indent=2,
            )

    print(f"Finished all problems. Results saved to {output_file}")
    print(f"Final Accuracy: {success}/{total_problems} = {success/total_problems:.2%}")


def main():
    parser = argparse.ArgumentParser(description="Run MATH evaluation on various models and various datasets")
    parser.add_argument("--model", type=str, required=True,
                        choices=["gpt-oss-20b", "qwen3-8b", "nemotron3"],
                        help="Model to evaluate")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["MATH", "AIME", "AIME25", "AIME26"],
                        help="Dataset to evaluate")

    # Model options
    parser.add_argument("--thinking", type=str, default="yes",
                        choices=["yes", "no"],
                        help="Enable thinking mode")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p sampling parameter")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--checkpoint_uri", type=str, default=None,
                        help="Tinker URI to load from checkpoint (for gpt-oss-20b only)")
    parser.add_argument("--inference_name", type=str, default=None,
                        help="Inference client name when loading from checkpoint (for gpt-oss-20b only)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of examples to evaluate")
    parser.add_argument("--seed", type=int, default=None,
                        help="Shuffle dataset with this seed before applying --limit")

    args = parser.parse_args()

    use_thinking = args.thinking == "yes"

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load dataset
    if args.dataset == "MATH":
        sample = prepare_MATH_from_hf(split="test")
    elif args.dataset == "AIME25":
        sample = prepare_AIME25_from_hf()
    elif args.dataset == "AIME26":
        sample = prepare_AIME26_from_hf()
    else:
        sample = prepare_AIME_from_hf()

    if args.seed is not None:
        import random
        random.Random(args.seed).shuffle(sample)
    if args.limit is not None:
        sample = sample[:args.limit]

    # Run evaluation
    run_evaluation(
        model_name=args.model,
        dataset=args.dataset,
        sample=sample,
        use_thinking=use_thinking,
        temperature=args.temperature,
        top_p=args.top_p,
        output_dir=args.output_dir,
        checkpoint_uri=args.checkpoint_uri,
        inference_name=args.inference_name,
    )


if __name__ == "__main__":
    main()
