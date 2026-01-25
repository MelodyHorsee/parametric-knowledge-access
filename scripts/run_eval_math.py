#!/usr/bin/env python3
import re
import os
import json
import argparse
import tempfile
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
import tinker
from tinker.types import SamplingParams as TinkerSamplingParams
from tinker_cookbook import renderers, tokenizer_utils
from tinker_cookbook.renderers import GptOssRenderer, Message, Role
from vllm import LLM, SamplingParams as vllmSamplingParams
from transformers import AutoTokenizer
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

    # remove inverse spaces
    string = string.replace("\\!", "")

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


class GPT52Evaluator:
    def __init__(self):
        self.client = OpenAI()
        self.model_name = "gpt-5.2"

    def build_prompt(self, question, use_cues):
        instruction = "You will be given a question."
        if use_cues:
            instruction += " Think step-by-step and give your final answer in \\boxed{}."
        else:
            instruction += " Give your final answer in \\boxed{}."

        return [
            {"role": "user", "content": instruction + " " + question}
        ]

    def generate(self, question, use_cues, use_thinking):
        messages = self.build_prompt(question, use_cues)
        response = self.client.responses.create(
            model=self.model_name,
            reasoning={"effort": "medium"},
            input=messages
        )
        text = response.output_text
        token_count = response.usage.total_tokens
        return text, token_count


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

    def build_prompt(self, question, use_cues):
        system_content = "You will be given a question."
        if use_cues:
            system_content += " Think step-by-step and give your final answer in \\boxed{}."
        else:
            system_content += " Give your final answer in \\boxed{}."

        return [
            {'role': 'system', 'content': system_content},
            {'role': 'user', 'content': question}
        ]

    def generate(self, question, use_cues, use_thinking):
        messages = self.build_prompt(question, use_cues)

        if use_thinking:
            renderer = self.renderers.get_renderer('gpt_oss_low_reasoning', self.tokenizer)
        else:
            # Custom renderer that disables thinking
            MessageClass = Message
            tinker_module = tinker
            tokenizer = self.tokenizer

            class CustomGptOssRenderer(self.GptOssRenderer):
                def build_generation_prompt(
                    self, messages: list[Message], role: Role = "assistant", prefill: str | None = None
                ) -> tinker.ModelInput:
                    tokens: list[int] = []
                    tokens.extend(self._bos_tokens)
                    if self.use_system_prompt:
                        tokens.extend(
                            self.tokenizer.encode(self._build_system_prompt(), add_special_tokens=False)
                        )
                    for message in messages:
                        ob_part, action_part, action_tail = self._render_message(message)
                        tokens.extend(ob_part)
                        tokens.extend(action_part)
                    new_partial_message = MessageClass(role=role, content="")
                    ob_part, _action_part, _action_tail = self._render_message(new_partial_message)
                    tokens.extend(ob_part)
                    tokens.extend(self.tokenizer.encode("<|channel|>analysis<|message|><|end|><|start|>assistant", add_special_tokens=False))
                    tokens.extend(self.tokenizer.encode(prefill or "", add_special_tokens=False))
                    return tinker_module.ModelInput.from_ints(tokens)

            renderer = CustomGptOssRenderer(tokenizer, use_system_prompt=True, reasoning_effort="low")

        stop_sequences = renderer.get_stop_sequences()
        sampling_params = self.SamplingParams(max_tokens=4096, temperature=self.temperature, top_p=self.top_p, stop=stop_sequences)

        prompt = renderer.build_generation_prompt(messages)
        output = self.sampling_client.sample(prompt, sampling_params=sampling_params, num_samples=1).result()
        sampled_message, _ = renderer.parse_response(output.sequences[0].tokens)

        text = sampled_message["content"]
        token_count = len(output.sequences[0].tokens)
        return text, token_count


class OlmoEvaluator:
    def __init__(self, temperature, top_p, batch_size=16):
        self.model_name = "allenai/Olmo-3-7B-Think"
        self.temperature = temperature
        self.top_p = top_p
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.temp_dir = tempfile.mkdtemp()
        self.custom_tokenizer_path = None
        self.SamplingParams = vllmSamplingParams
        self.llm = None

    def build_prompt(self, question, use_cues, use_thinking):
        system_content = "You will be given a question."
        if use_cues:
            system_content += " Think step-by-step and give your final answer in \\boxed{}."
        else:
            system_content += " Give your final answer in \\boxed{}."

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": question}
        ]

        if not use_thinking:
            # Prefill with closed think tags for no-thinking mode
            if self.custom_tokenizer_path is None:
                self.tokenizer.chat_template = """<｜begin▁of▁sentence｜>{% for message in messages %}<｜{{ message.role | capitalize }}｜>{{ message.content }}{% endfor %}<｜Assistant｜><think>\n\n</think>\n"""
                self.custom_tokenizer_path = f"{self.temp_dir}/custom_olmo_tokenizer"
                self.tokenizer.save_pretrained(self.custom_tokenizer_path)

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def initialize_llm(self, use_thinking):
        if self.llm is None:
            if not use_thinking and self.custom_tokenizer_path:
                self.llm = LLM(
                    model=self.model_name,
                    tokenizer=self.custom_tokenizer_path,
                    dtype="bfloat16",
                    gpu_memory_utilization=0.5,
                )
            else:
                self.llm = LLM(
                    model=self.model_name,
                    dtype="bfloat16",
                    gpu_memory_utilization=0.5,
                )

    def generate_batch(self, questions, use_cues, use_thinking):
        prompts = [self.build_prompt(q, use_cues, use_thinking) for q in questions]
        self.initialize_llm(use_thinking)

        if use_thinking:
            thinking_params = self.SamplingParams(
                max_tokens=7000,
                temperature=self.temperature,
                top_p=self.top_p,
                n=1,
            )
            outputs = self.llm.generate(prompts, thinking_params)

            results = []
            for i, out in enumerate(outputs):
                text = out.outputs[0].text
                thinking_tokens = len(out.outputs[0].token_ids)

                if "</think>" in text.lower():
                    final_text = text
                    token_count = thinking_tokens
                else:
                    forced_prompt = prompts[i] + text + "\n</think>\n"
                    answer_params = self.SamplingParams(
                        max_tokens=1028,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        n=1,
                    )
                    answer_out = self.llm.generate([forced_prompt], answer_params)[0]
                    final_text = text + "\n</think>\n" + answer_out.outputs[0].text
                    token_count = thinking_tokens + len(answer_out.outputs[0].token_ids)

                results.append((final_text, token_count))
            return results
        else:
            sampling_params = self.SamplingParams(
                max_tokens=4096,
                temperature=self.temperature,
                top_p=self.top_p,
                n=1,
            )
            outputs = self.llm.generate(prompts, sampling_params)
            return [(out.outputs[0].text, len(out.outputs[0].token_ids)) for out in outputs]


class R1QwenEvaluator:
    def __init__(self, temperature, top_p, batch_size=16):
        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        self.temperature = temperature
        self.top_p = top_p
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.temp_dir = tempfile.mkdtemp()
        self.custom_tokenizer_path = None
        self.SamplingParams = vllmSamplingParams
        self.llm = None

    def build_prompt(self, question, use_cues, use_thinking):
        if use_cues:
            system_prompt = "Think step-by-step and give your final answer in \\boxed{}."
        else:
            system_prompt = "Give your final answer in \\boxed{}."

        messages = [
            {"role": "user", "content": question + "\n" + system_prompt}
        ]

        if not use_thinking:
            # Prefill with closed think tags for no-thinking mode
            if self.custom_tokenizer_path is None:
                self.tokenizer.chat_template = """{% for message in messages %}{{ message.content }}{% endfor %}<think>\n\n</think>\n"""
                self.custom_tokenizer_path = f"{self.temp_dir}/custom_r1_tokenizer"
                self.tokenizer.save_pretrained(self.custom_tokenizer_path)

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def initialize_llm(self, use_thinking):
        if self.llm is None:
            if not use_thinking and self.custom_tokenizer_path:
                self.llm = LLM(
                    model=self.model_name,
                    tokenizer=self.custom_tokenizer_path,
                    dtype="bfloat16",
                    gpu_memory_utilization=0.3,
                )
            else:
                self.llm = LLM(
                    model=self.model_name,
                    dtype="bfloat16",
                    gpu_memory_utilization=0.3,
                )

    def generate_batch(self, questions, use_cues, use_thinking):
        prompts = [self.build_prompt(q, use_cues, use_thinking) for q in questions]
        self.initialize_llm(use_thinking)

        sampling_params = self.SamplingParams(
            max_tokens=4096,
            temperature=self.temperature,
            top_p=self.top_p,
            n=1,
        )

        outputs = self.llm.generate(prompts, sampling_params)
        return [(out.outputs[0].text, len(out.outputs[0].token_ids)) for out in outputs]


def prepare_dataset_from_hf(split="test"):
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


def run_evaluation(model_name, sample, use_cues, use_thinking, temperature, top_p, output_dir, checkpoint_uri=None, inference_name=None):
    # Build output filename
    cues_str = "with_cues" if use_cues else "no_cues"
    thinking_str = "with_thinking" if use_thinking else "no_thinking"

    model_short = {
        "gpt-5.2": "gpt52",
        "gpt-oss-20b": "gptoss20b",
        "olmo-3-7b-think": "olmo3",
        "r1-distill-qwen-1.5b": "r1qwen"
    }[model_name]

    # Add inference_name to model_short if checkpoint is being used
    if inference_name:
        model_short = f"{model_short}_{inference_name}"

    # GPT-5.2 doesn't have thinking control
    if model_name == "gpt-5.2" and not use_thinking:
        print(f"Warning: GPT-5.2 doesn't support no-thinking mode. Skipping.")
        return

    if model_name == "gpt-5.2":
        output_file = Path(output_dir) / f"math_{model_short}_{cues_str}.json"
    else:
        output_file = Path(output_dir) / f"math_{model_short}_{cues_str}_{thinking_str}.json"

    print(f"\nRunning evaluation: {model_name} on MATH")
    print(f"  Dataset size: {len(sample)} problems")
    print(f"  Cues: {use_cues}, Thinking: {use_thinking}")
    if model_name == "gpt-5.2":
        print(f"  Temperature and Top-p settings not applicable to GPT-5.2")
    else:
        print(f"  Temperature: {temperature}, Top-p: {top_p}")
    print(f"  Output: {output_file}")

    # Initialize evaluator
    if model_name == "gpt-5.2":
        evaluator = GPT52Evaluator()
        batch_mode = False
    elif model_name == "gpt-oss-20b":
        evaluator = GPTOSSEvaluator(temperature, top_p, checkpoint_uri, inference_name)
        batch_mode = False
    elif model_name == "olmo-3-7b-think":
        evaluator = OlmoEvaluator(temperature, top_p)
        batch_mode = True
    elif model_name == "r1-distill-qwen-1.5b":
        evaluator = R1QwenEvaluator(temperature, top_p)
        batch_mode = True
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Run evaluation
    results = []
    errors = []
    token_lengths = []
    correct_token_lengths = []
    incorrect_token_lengths = []
    success = 0

    if batch_mode:
        batch_size = evaluator.batch_size
        for i in tqdm(range(0, len(sample), batch_size)):
            batch_items = sample[i:i + batch_size]
            batch_questions = [item["problem"] for item in batch_items]

            try:
                batch_results = evaluator.generate_batch(batch_questions, use_cues, use_thinking)

                for item, (text, token_count) in zip(batch_items, batch_results):
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
                        gt = remove_boxed(gt_boxed) if gt_boxed else None
                    except Exception as e:
                        gt = None
                        errors.append({
                            "problem": item["problem"],
                            "error": f"gt_extraction: {str(e)}"
                        })

                    correct = is_equiv(pred, gt)

                    token_lengths.append(token_count)
                    if correct:
                        correct_token_lengths.append(token_count)
                        success += 1
                    else:
                        incorrect_token_lengths.append(token_count)

                    results.append({
                        "problem": item["problem"],
                        "solution": item["solution"],
                        "type": item["type"],
                        "raw_prediction": text,
                        "extracted_prediction": pred,
                        "extracted_gt": gt,
                        "num_tokens": token_count,
                        "correct": correct,
                    })

            except Exception as e:
                for item in batch_items:
                    errors.append({
                        "problem": item["problem"],
                        "error": str(e),
                    })

            # Save after each batch
            aggregate_metrics = {
                "total_problems": len(sample),
                "correct": success,
                "accuracy": success / len(sample),
                "extraction_errors": len(errors),
                "avg_tokens": sum(token_lengths) / len(token_lengths) if token_lengths else 0,
                "avg_tokens_correct": sum(correct_token_lengths) / len(correct_token_lengths) if correct_token_lengths else 0,
                "avg_tokens_incorrect": sum(incorrect_token_lengths) / len(incorrect_token_lengths) if incorrect_token_lengths else 0,
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
    else:
        for idx, item in tqdm(enumerate(sample), total=len(sample)):
            try:
                text, token_count = evaluator.generate(item["problem"], use_cues, use_thinking)

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
                    gt = remove_boxed(gt_boxed) if gt_boxed else None
                except Exception as e:
                    gt = None
                    errors.append({
                        "problem": item["problem"],
                        "error": f"gt_extraction: {str(e)}"
                    })

                correct = is_equiv(pred, gt)

                token_lengths.append(token_count)
                if correct:
                    correct_token_lengths.append(token_count)
                    success += 1
                else:
                    incorrect_token_lengths.append(token_count)

                results.append({
                    "problem": item["problem"],
                    "solution": item["solution"],
                    "type": item["type"],
                    "raw_prediction": text,
                    "extracted_prediction": pred,
                    "extracted_gt": gt,
                    "num_tokens": token_count,
                    "correct": correct,
                })

            except Exception as e:
                errors.append({
                    "problem": item["problem"],
                    "error": str(e),
                })

            # Save after each question
            aggregate_metrics = {
                "total_problems": len(sample),
                "correct": success,
                "accuracy": success / len(sample),
                "extraction_errors": len(errors),
                "avg_tokens": sum(token_lengths) / len(token_lengths) if token_lengths else 0,
                "avg_tokens_correct": sum(correct_token_lengths) / len(correct_token_lengths) if correct_token_lengths else 0,
                "avg_tokens_incorrect": sum(incorrect_token_lengths) / len(incorrect_token_lengths) if incorrect_token_lengths else 0,
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
    print(f"Final Accuracy: {success}/{len(sample)} = {success/len(sample):.2%}")


def main():
    parser = argparse.ArgumentParser(description="Run MATH evaluation on various models")
    parser.add_argument("--model", type=str, required=True,
                        choices=["gpt-5.2", "gpt-oss-20b", "olmo-3-7b-think", "r1-distill-qwen-1.5b"],
                        help="Model to evaluate")

    # Model options
    parser.add_argument("--cues", type=str, default="no",
                        choices=["yes", "no"],
                        help="Use thinking cues (think step-by-step)")
    parser.add_argument("--thinking", type=str, default="yes",
                        choices=["yes", "no"],
                        help="Enable thinking mode (not applicable for gpt-5.2)")
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

    args = parser.parse_args()

    # Convert string args to boolean
    use_cues = args.cues == "yes"
    use_thinking = args.thinking == "yes"

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load dataset from HuggingFace (always use test split)
    sample = prepare_dataset_from_hf(split="test")

    # Run evaluation
    run_evaluation(
        model_name=args.model,
        sample=sample,
        use_cues=use_cues,
        use_thinking=use_thinking,
        temperature=args.temperature,
        top_p=args.top_p,
        output_dir=args.output_dir,
        checkpoint_uri=args.checkpoint_uri,
        inference_name=args.inference_name
    )


if __name__ == "__main__":
    main()
