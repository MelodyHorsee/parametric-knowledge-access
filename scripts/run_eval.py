#!/usr/bin/env python3
import re
import os
import string
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
import pandas

# From https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py

def normalize_answer(s):
    def normalize_unicode_dashes(text):
        return re.sub(r"[‐-‒–—―−]", " ", text)

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(normalize_unicode_dashes(replace_underscore(s)))))).strip()


def extract_answer(text):
    # Find all <answer>...</answer> matches
    matches = re.finditer(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    for match in matches:
        content = match.group(1).strip()
        # Return the first non-empty match
        if content:
            return content

    # If no answer tags found, extract what comes after </think> tag
    think_match = re.split(r"</think>", text, flags=re.IGNORECASE)
    if len(think_match) > 1:
        return think_match[-1].strip()

    return text.strip()


def exact_match_score(prediction, ground_truths):
    pred = normalize_answer(prediction)
    for gt in ground_truths:
        if pred == normalize_answer(gt):
            return True
    return False


def recall_score(prediction, ground_truths):
    pred = normalize_answer(prediction)
    for gt in ground_truths:
        gt_normalized = normalize_answer(gt)
        # Escape special regex characters in the ground truth
        gt_escaped = re.escape(gt_normalized)
        # Replace escaped spaces with \s+ to match any whitespace
        gt_pattern = gt_escaped.replace(r'\ ', r'\s+')
        # Use word boundaries to ensure whole word matching
        pattern = r'\b' + gt_pattern + r'\b'
        if re.search(pattern, pred):
            return True
    return False


class GPT52Evaluator:
    def __init__(self):
        self.client = OpenAI()
        self.model_name = "gpt-5.2"

    def build_prompt(self, question, use_cues):
        instruction = "You will be given a question."
        if use_cues:
            instruction += " Think step-by-step and give your final answer in <answer></answer> tags."
        else:
            instruction += " Give your final answer in <answer></answer> tags."

        return [
            {"role": "user", "content": instruction + " " + question}
        ]

    def generate(self, question, use_cues, use_thinking, **kwargs):
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
            system_content += " Think step-by-step and give your final answer in <answer></answer> tags."
        else:
            system_content += " Give your final answer in <answer></answer> tags."

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
        sampling_params = self.SamplingParams(max_tokens=2048, temperature=self.temperature, top_p=self.top_p, stop=stop_sequences)

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
            system_content += " Think step-by-step and give your final answer in <answer></answer> tags."
        else:
            system_content += " Give your final answer in <answer></answer> tags."

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": question}
        ]

        if not use_thinking:
            # Prefill with closed think tags for no-thinking mode
            if self.custom_tokenizer_path is None:
                self.tokenizer.chat_template = """<｜begin▁of▁sentence｜>{% for message in messages %}<｜{{ message.role | capitalize }}｜>{{ message.content }}{% endfor %}<｜Assistant｜><think>\n\n</think>\n<answer>"""
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
                    forced_prompt = prompts[i] + text + "\n</think>\n<answer>"
                    answer_params = self.SamplingParams(
                        max_tokens=1028,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        n=1,
                    )
                    answer_out = self.llm.generate([forced_prompt], answer_params)[0]
                    final_text = text + "\n</think>\n<answer>" + answer_out.outputs[0].text
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
            system_prompt = "Think step-by-step and give your final answer in <answer></answer> tags."
        else:
            system_prompt = "Give your final answer in <answer></answer> tags."

        messages = [
            {"role": "user", "content": question + "\n" + system_prompt}
        ]

        if not use_thinking:
            # Prefill with closed think tags for no-thinking mode
            if self.custom_tokenizer_path is None:
                # R1 uses different chat template format
                self.tokenizer.chat_template = """{% for message in messages %}{{ message.content }}{% endfor %}<think>\n\n</think>\n<answer>"""
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

        sampling_params = self.SamplingParams(
            max_tokens=4096,
            temperature=self.temperature,
            top_p=self.top_p,
            n=1,
        )

        outputs = self.llm.generate(prompts, sampling_params)
        return [(out.outputs[0].text, len(out.outputs[0].token_ids)) for out in outputs]


def load_dataset(dataset_name):
    if dataset_name == "triviaqa":
        from datasets import load_dataset as hf_load_dataset
        raw_data = hf_load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
        sample = []
        for item in raw_data:
            sample.append({
                "question": item["question"],
                "answer_aliases": item["answer"]["normalized_aliases"]
            })
    elif dataset_name == "nq":
        sample = []
        # Download from: https://github.com/efficientqa/nq-open
        # Place NQ-open.dev.jsonl in the NQ/ directory
        nq_path = "NQ/NQ-open.dev.jsonl"
        with open(nq_path) as f:
            for line in f:
                item = json.loads(line)
                sample.append({
                    "question": item["question"],
                    "answer_aliases": item["answer"]
                })
    elif dataset_name == "hotpotqa":
        from datasets import load_dataset as hf_load_dataset
        raw_data = hf_load_dataset("hotpotqa/hotpot_qa", "fullwiki", split="validation")
        sample = []
        for item in raw_data:
            # HotPotQA has "question" and "answer" fields
            # Wrap answer in a list for consistency with other datasets
            sample.append({
                "question": item["question"],
                "answer_aliases": [item["answer"]]
            })
    elif dataset_name == "simpleqa":
        # Load SimpleQA from CSV (following official simple-evals implementation)
        try:
            df = pandas.read_csv(
                "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
            )
        except Exception as e:
            if "CERTIFICATE_VERIFY_FAILED" in str(e):
                print("\nSSL Certificate Error: Unable to download SimpleQA dataset.")
                print("To fix this on macOS, run:")
                print("  /Applications/Python\\ 3.12/Install\\ Certificates.command")
                print("Or install certifi: pip install --upgrade certifi")
                raise
            else:
                raise

        sample = []
        for _, row in df.iterrows():
            # SimpleQA has "problem" and "answer" fields
            # Wrap answer in a list for consistency with other datasets
            sample.append({
                "question": row["problem"],
                "answer_aliases": [row["answer"]]
            })
    elif dataset_name == "strategyqa":
        strategyqa_path = "StrategyQA/strategyqa_train.json"
        with open(strategyqa_path) as f:
            raw_data = json.load(f)
        sample = []
        for item in raw_data:
            sample.append({
                "question": item["question"],
                "answer_aliases": ["true"] if item["answer"] else ["false"]
            })
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return sample


def run_evaluation(model_name, dataset_name, use_cues, use_thinking, temperature, top_p, output_dir, checkpoint_uri=None, inference_name=None):
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
        output_file = Path(output_dir) / f"{dataset_name}_{model_short}_{cues_str}.json"
    else:
        output_file = Path(output_dir) / f"{dataset_name}_{model_short}_{cues_str}_{thinking_str}.json"

    print(f"Running evaluation: {model_name} on {dataset_name}")
    print(f"  Cues: {use_cues}, Thinking: {use_thinking}")
    if model_name == "gpt-5.2":
        print(f"  Temperature and Top-p settings not applicable to GPT-5.2")
    else: 
        print(f"  Temperature: {temperature}, Top-p: {top_p}")
    print(f"  Output: {output_file}")

    # Load dataset
    sample = load_dataset(dataset_name)
    print(f"Loaded {len(sample)} questions.")

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
    recalled_token_lengths = []
    not_recalled_token_lengths = []
    success = 0
    recall_hits = 0

    if batch_mode:
        batch_size = evaluator.batch_size
        for i in tqdm(range(0, len(sample), batch_size)):
            batch_items = sample[i:i + batch_size]
            batch_questions = [item["question"] for item in batch_items]

            try:
                batch_results = evaluator.generate_batch(batch_questions, use_cues, use_thinking)

                for item, (text, token_count) in zip(batch_items, batch_results):
                    try:
                        # For Olmo/R1 no-thinking mode, extract everything before </answer> tag
                        # since the prompt already includes <answer> opening tag
                        if not use_thinking and model_name in ["olmo-3-7b-think", "r1-distill-qwen-1.5b"]:
                            # Split on </answer> and take the first part
                            extracted = re.split(r"</answer>", text, flags=re.IGNORECASE)[0].strip()
                        else:
                            extracted = extract_answer(text)
                        ground_truths = item["answer_aliases"]

                        em = exact_match_score(extracted, ground_truths)
                        rec = recall_score(extracted, ground_truths)

                        token_lengths.append(token_count)
                        if rec:
                            recalled_token_lengths.append(token_count)
                        else:
                            not_recalled_token_lengths.append(token_count)

                        results.append({
                            "question": item["question"],
                            "ground_truth": ground_truths,
                            "raw_prediction": text,
                            "extracted_prediction": extracted,
                            "num_tokens": token_count,
                            "exact_match": em,
                            "recall": rec,
                        })

                        if em:
                            success += 1
                        if rec:
                            recall_hits += 1

                    except Exception as e:
                        errors.append({
                            "question": item["question"],
                            "error": str(e),
                        })

            except Exception as e:
                for item in batch_items:
                    errors.append({
                        "question": item["question"],
                        "error": str(e),
                    })

            # Save after each batch
            aggregate_metrics = {
                "total_questions": len(sample),
                "exact_match_successes": success,
                "recall_successes": recall_hits,
                "exact_match_rate": success / len(sample),
                "recall_rate": recall_hits / len(sample),
                "extraction_errors": len(errors),
                "avg_tokens_per_response": (
                    sum(token_lengths) / len(token_lengths)
                    if token_lengths else 0
                ),
                "avg_tokens_recalled": (
                    sum(recalled_token_lengths) / len(recalled_token_lengths)
                    if recalled_token_lengths else 0
                ),
                "avg_tokens_not_recalled": (
                    sum(not_recalled_token_lengths) / len(not_recalled_token_lengths)
                    if not_recalled_token_lengths else 0
                ),
                "num_recalled": len(recalled_token_lengths),
                "num_not_recalled": len(not_recalled_token_lengths),
            }

            with open(output_file, "w") as f:
                json.dump(
                    {
                        "results": results,
                        "aggregate_metrics": aggregate_metrics,
                        "errors": errors,
                        "success": success,
                        "recall_hits": recall_hits,
                    },
                    f,
                    indent=2,
                )
    else:
        for idx, item in tqdm(enumerate(sample), total=len(sample)):
            try:
                text, token_count = evaluator.generate(item["question"], use_cues, use_thinking)

                extracted = extract_answer(text)
                ground_truths = item["answer_aliases"]

                em = exact_match_score(extracted, ground_truths)
                rec = recall_score(extracted, ground_truths)

                token_lengths.append(token_count)
                if rec:
                    recalled_token_lengths.append(token_count)
                else:
                    not_recalled_token_lengths.append(token_count)

                results.append({
                    "question": item["question"],
                    "ground_truth": ground_truths,
                    "raw_prediction": text,
                    "extracted_prediction": extracted,
                    "num_tokens": token_count,
                    "exact_match": em,
                    "recall": rec,
                })

                if em:
                    success += 1
                if rec:
                    recall_hits += 1

            except Exception as e:
                errors.append({
                    "question": item["question"],
                    "error": str(e),
                })

            # Save after each question
            aggregate_metrics = {
                "total_questions": len(sample),
                "exact_match_successes": success,
                "recall_successes": recall_hits,
                "exact_match_rate": success / len(sample),
                "recall_rate": recall_hits / len(sample),
                "extraction_errors": len(errors),
                "avg_tokens_per_response": (
                    sum(token_lengths) / len(token_lengths)
                    if token_lengths else 0
                ),
                "avg_tokens_recalled": (
                    sum(recalled_token_lengths) / len(recalled_token_lengths)
                    if recalled_token_lengths else 0
                ),
                "avg_tokens_not_recalled": (
                    sum(not_recalled_token_lengths) / len(not_recalled_token_lengths)
                    if not_recalled_token_lengths else 0
                ),
                "num_recalled": len(recalled_token_lengths),
                "num_not_recalled": len(not_recalled_token_lengths),
            }

            with open(output_file, "w") as f:
                json.dump(
                    {
                        "results": results,
                        "aggregate_metrics": aggregate_metrics,
                        "errors": errors,
                        "success": success,
                        "recall_hits": recall_hits,
                    },
                    f,
                    indent=2,
                )

    print(f"Finished all questions. Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run QA evaluation on various models")
    parser.add_argument("--model", type=str, required=True,
                        choices=["gpt-5.2", "gpt-oss-20b", "olmo-3-7b-think", "r1-distill-qwen-1.5b"],
                        help="Model to evaluate")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["triviaqa", "nq", "hotpotqa", "simpleqa", "strategyqa", "all"],
                        help="Dataset to use ('all' to run on all 5 datasets)")
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

    datasets = ["triviaqa", "nq", "hotpotqa", "simpleqa", "strategyqa"] if args.dataset == "all" else [args.dataset]

    for dataset_name in datasets:
        run_evaluation(
            model_name=args.model,
            dataset_name=dataset_name,
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
