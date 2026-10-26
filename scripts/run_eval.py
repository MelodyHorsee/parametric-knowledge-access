#!/usr/bin/env python3
import re
import os
import random
import string
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import tinker
from tinker.types import SamplingParams as TinkerSamplingParams
from tinker_cookbook import renderers, tokenizer_utils
from tinker_cookbook.renderers.gpt_oss import GptOssRenderer, Message, Role
import pandas

# From https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py

def normalize_answer(s):
    def normalize_unicode(text):
        # Normalize unicode dashes/hyphens to regular hyphen
        text = re.sub(r"[‐‑‒–—―−-­]", "-", text)
        # Normalize unicode spaces to regular space
        text = re.sub(r"[            　]", " ", text)
        # Normalize unicode subscript digits to regular digits
        sub_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
        text = text.translate(sub_map)
        # Normalize unicode superscript digits to regular digits
        sup_map = {0xB2: "2", 0xB3: "3", 0xB9: "1", 0x2070: "0", 0x2074: "4", 0x2075: "5", 0x2076: "6", 0x2077: "7", 0x2078: "8", 0x2079: "9"}
        text = text.translate(sup_map)
        return text

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"'", u"'", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    def normalize_ampersand(text):
        # Treat & and "and" as equivalent by normalizing & -> and
        return re.sub(r'\s*&\s*', ' and ', text)

    def normalize_dates(text):
        # Normalize "Month DD, YYYY" and "Month DD YYYY" -> "DD Month YYYY"
        # so "November 20, 1989" and "20 November 1989" compare equal
        months = (
            r'january|february|march|april|may|june|july|august|september|october|november|december|'
            r'jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec'
        )
        # "November 20, 1989" or "November 20 1989" -> "20 november 1989"
        text = re.sub(
            rf'\b({months})\s+(\d{{1,2}}),?\s+(\d{{4}})\b',
            r'\2 \1 \3', text, flags=re.IGNORECASE
        )
        return text

    return white_space_fix(remove_articles(handle_punc(lower(normalize_dates(normalize_ampersand(normalize_unicode(replace_underscore(s)))))))).strip()


def _extract_boxed(text):
    """Return the content of the last \\boxed{...} in text (brace-balanced), or None."""
    marker = r'\boxed{'
    last_content = None
    idx = 0
    while True:
        start = text.find(marker, idx)
        if start == -1:
            break
        i = start + len(marker)
        depth = 1
        while i < len(text) and depth > 0:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            i += 1
        if depth == 0:
            content = text[start + len(marker):i - 1].strip()
            if content:
                last_content = content
        idx = start + len(marker)
    return last_content


def extract_answer(text):
    if isinstance(text, list):
        text = " ".join(c if isinstance(c, str) else c.get("text", "") for c in text)
    # Step 1: Isolate the answer portion by stripping all thinking tokens
    # GPT-OSS thinking: <|channel|>analysis<|message|>...<|end|>
    clean = re.sub(r"<\|channel\|>analysis<\|message\|>.*?<\|end\|>", "", text, flags=re.DOTALL)
    # R1/Olmo/Qwen3/Nemotron thinking: <think>...</think>
    clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL | re.IGNORECASE)
    # Some renderers put the opening <think> in the prompt prefix rather than
    # the completion, so only a stray closing tag shows up here. Strip
    # everything up through it too, otherwise example <answer>/\boxed{} content
    # used inside the reasoning gets mistaken for the real answer below.
    think_close = re.search(r"</think>", clean, re.IGNORECASE)
    if think_close:
        clean = clean[think_close.end():]

    clean = clean.strip()

    # Priority 1: <answer>...</answer> tags — take the last non-empty match,
    # since earlier ones can be leftover reasoning examples.
    matches = list(re.finditer(r"<answer>(.*?)</answer>", clean, re.IGNORECASE | re.DOTALL))
    for match in reversed(matches):
        content = match.group(1).strip()
        if content:
            return content

    # Priority 1b: unterminated <answer> tag — grab everything after the last one.
    answer_opens = list(re.finditer(r"<answer>", clean, re.IGNORECASE))
    if answer_opens:
        content = clean[answer_opens[-1].end():].strip()
        if content:
            return content

    # Priority 2: \boxed{...} — LaTeX-style final answer some reasoning models
    # (e.g. Nemotron) emit instead of <answer></answer> tags.
    boxed = _extract_boxed(clean)
    if boxed:
        return boxed

    # Priority 3: default to everything remaining after </think> (already
    # sliced above). Strip a leftover GPT-OSS final-channel marker prefix
    # first, since that's not part of the visible answer.
    final_marker = "<|start|>assistant<|channel|>final<|message|>"
    final_idx = clean.rfind(final_marker)
    if final_idx != -1:
        clean = clean[final_idx + len(final_marker):].strip()

    return clean.strip()


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


class GPTOSSEvaluator:
    def __init__(self, temperature, top_p, checkpoint_uri=None, inference_name=None, model_name='openai/gpt-oss-20b'):
        """
        Args:
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            checkpoint_uri: Optional Tinker URI to load from checkpoint (e.g., "tinker://.../:train:0/weights/001240")
            inference_name: Optional name for inference client when loading from checkpoint
            model_name: GPT-OSS variant to use (e.g. 'openai/gpt-oss-20b')
        """
        self.tokenizer = tokenizer_utils.get_tokenizer(model_name)
        self.service_client = tinker.ServiceClient()

        # Load from checkpoint if provided, otherwise use base model
        if checkpoint_uri and inference_name:
            training_client = self.service_client.create_training_client_from_state(checkpoint_uri)
            self.sampling_client = training_client.save_weights_and_get_sampling_client(inference_name)
        else:
            self.sampling_client = self.service_client.create_sampling_client(base_model=model_name)

        self.temperature = temperature
        self.top_p = top_p
        self.GptOssRenderer = GptOssRenderer
        self.SamplingParams = TinkerSamplingParams
        self.renderers = renderers

    def build_prompt(self, question, answers):
        system_content = "You will be given a question. Give your final answer in <answer></answer> tags."

        example_answer = answers[0]

        # return [
        #     {'role': 'system', 'content': system_content},
        #     {'role': 'user', 'content': question + "\n\n" + "This is an example for a response to the question:\n" + example_answer + "\n\n" + "Now answer with a response of your own, including the thinking process."}
        # ]

        return [
            {'role': 'system', 'content': system_content},
            {'role': 'user', 'content': question}
        ]

    def generate(self, question, answers, use_thinking, num_samples=4, system_instruction=None):
        messages = self.build_prompt(question, answers) if not system_instruction else [
            {'role': 'system', 'content': system_instruction},
            {'role': 'user', 'content': question}
        ]

        # GptOssRenderer's default generation suffix is just "<|start|>assistant" —
        # it does NOT auto-open a thinking channel (that's the model's own choice).
        # So no-thinking mode is achieved via prefill, not a custom renderer
        # subclass: same pattern as trivia_env.py's gpt_oss_no_reasoning mode,
        # which is what checkpoints loaded with renderer_name=gpt_oss_no_reasoning
        # were actually trained under.
        prefill = None
        if use_thinking:
            renderer = self.renderers.get_renderer('gpt_oss_low_reasoning', self.tokenizer)
        else:
            renderer = self.GptOssRenderer(self.tokenizer, use_system_prompt=True, reasoning_effort="low")
            prefill = "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>The answer is"

        stop_sequences = renderer.get_stop_sequences()
        sampling_params = self.SamplingParams(max_tokens=4096, temperature=self.temperature, top_p=self.top_p, stop=stop_sequences)

        prompt = renderer.build_generation_prompt(messages, prefill=prefill)
        output = self.sampling_client.sample(prompt, sampling_params=sampling_params, num_samples=num_samples).result()

        all_contents = []
        all_thinking_counts = []
        force_suffix = "<|end|><|start|>assistant<|channel|>final<|message|>"
        force_suffix_tokens = self.tokenizer.encode(force_suffix, add_special_tokens=False)
        for seq in output.sequences:
            # Full raw output including analysis + final channels
            raw_full = self.tokenizer.decode(seq.tokens)

            if not use_thinking:
                # No-thinking mode: the analysis channel (if any) is entirely in
                # the prompt/prefill, never in the sampled completion, so it will
                # never appear in raw_full — there's nothing to detect or retry.
                all_contents.append(raw_full)
                all_thinking_counts.append(0)
                continue

            # Count thinking tokens from analysis channel. A missing <|end|> is only
            # a truncation if the analysis channel was actually opened — if the model
            # never opened it at all, the response is already complete (this is the
            # model's own choice not to reason, not a cut-off), so don't force-continue.
            thinking_start = raw_full.find("<|channel|>analysis<|message|>")
            thinking_end = raw_full.find("<|end|>", thinking_start) if thinking_start != -1 else -1

            if thinking_start == -1:
                all_contents.append(raw_full)
                all_thinking_counts.append(0)
                continue

            if thinking_end != -1:
                thinking_text = raw_full[thinking_start + len("<|channel|>analysis<|message|>"):thinking_end]
                thinking_token_count = len(self.tokenizer.encode(thinking_text, add_special_tokens=False))
                all_contents.append(raw_full)
                all_thinking_counts.append(thinking_token_count)
                continue

            # Analysis channel was opened but never closed — genuinely truncated
            # mid-thinking. Force-close it and get an actual final answer.
            thinking_token_count = len(seq.tokens)
            forced_prompt = tinker.ModelInput.from_ints(
                list(prompt.to_ints()) + list(seq.tokens) + force_suffix_tokens
            )
            forced_params = self.SamplingParams(
                max_tokens=512, temperature=self.temperature, top_p=self.top_p, stop=stop_sequences
            )
            forced_output = self.sampling_client.sample(forced_prompt, sampling_params=forced_params, num_samples=1).result()
            forced_text = self.tokenizer.decode(forced_output.sequences[0].tokens)
            all_contents.append(raw_full + force_suffix + forced_text)
            all_thinking_counts.append(thinking_token_count)

        return all_contents, all_thinking_counts


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

    def build_prompt(self, question, system_instruction=None):
        if system_instruction:
            system_content = system_instruction
        else:
            system_content = "You will be given a question. Give your final answer in <answer></answer> tags."
        return [
            {'role': 'system', 'content': system_content},
            {'role': 'user', 'content': question}
        ]

    def generate(self, question, answers, use_thinking, num_samples=1, system_instruction=None):
        messages = self.build_prompt(question, system_instruction)

        renderer_name = 'qwen3' if use_thinking else 'qwen3_disable_thinking'
        renderer = renderers.get_renderer(renderer_name, self.tokenizer)

        stop_sequences = renderer.get_stop_sequences()
        sampling_params = self.SamplingParams(max_tokens=4096, temperature=self.temperature, top_p=self.top_p, stop=stop_sequences)

        prompt = renderer.build_generation_prompt(messages)
        output = self.sampling_client.sample(prompt, sampling_params=sampling_params, num_samples=num_samples).result()

        all_contents = []
        all_thinking_counts = []
        force_suffix = "</think>\n\n<answer>"
        force_suffix_tokens = self.tokenizer.encode(force_suffix, add_special_tokens=False)

        for seq in output.sequences:
            raw_full = self.tokenizer.decode(seq.tokens)

            # <think> is NOT prefilled in the prompt for the thinking-enabled 'qwen3'
            # renderer — the model decides whether to open it. A missing </think> is
            # only a truncation if <think> was actually opened.
            think_start = raw_full.find("<think>")
            think_end = raw_full.find("</think>", think_start) if think_start != -1 else -1

            if think_start == -1:
                all_contents.append(raw_full)
                all_thinking_counts.append(0)
                continue

            if think_end != -1:
                thinking_text = raw_full[think_start + len("<think>"):think_end]
                thinking_token_count = len(self.tokenizer.encode(thinking_text, add_special_tokens=False))
                all_contents.append(raw_full)
                all_thinking_counts.append(thinking_token_count)
                continue

            # Max tokens hit mid-thinking — force close thinking and prefill <answer>
            thinking_token_count = len(seq.tokens)
            forced_prompt = tinker.ModelInput.from_ints(
                list(prompt.to_ints()) + list(seq.tokens) + force_suffix_tokens
            )
            forced_params = self.SamplingParams(
                max_tokens=512, temperature=self.temperature, top_p=self.top_p, stop=stop_sequences
            )
            forced_output = self.sampling_client.sample(forced_prompt, sampling_params=forced_params, num_samples=1).result()
            forced_text = self.tokenizer.decode(forced_output.sequences[0].tokens)
            all_contents.append(raw_full + force_suffix + forced_text)
            all_thinking_counts.append(thinking_token_count)

        return all_contents, all_thinking_counts


class Nemotron3NanoEvaluator:
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

    def build_prompt(self, question, system_instruction=None):
        if system_instruction:
            system_content = system_instruction
        else:
            system_content = "You will be given a question. Give your final answer in <answer></answer> tags."
        return [
            {'role': 'system', 'content': system_content},
            {'role': 'user', 'content': question}
        ]

    def generate(self, question, answers, use_thinking, num_samples=1, system_instruction=None):
        messages = self.build_prompt(question, system_instruction)

        # Nemotron3Renderer(Qwen3_5Renderer): <think>\n...\n</think>\n formatting.
        # Disable variant prepends <think></think> to skip thinking.
        renderer_name = 'nemotron3' if use_thinking else 'nemotron3_disable_thinking'
        renderer = renderers.get_renderer(renderer_name, self.tokenizer)

        stop_sequences = renderer.get_stop_sequences()
        sampling_params = self.SamplingParams(max_tokens=4096, temperature=self.temperature, top_p=self.top_p, stop=stop_sequences)

        prompt = renderer.build_generation_prompt(messages)
        output = self.sampling_client.sample(prompt, sampling_params=sampling_params, num_samples=num_samples).result()

        all_contents = []
        all_thinking_counts = []
        force_suffix = "</think>\n\n<answer>"
        force_suffix_tokens = self.tokenizer.encode(force_suffix, add_special_tokens=False)

        for seq in output.sequences:
            raw_full = self.tokenizer.decode(seq.tokens)

            # The opening <think> tag is part of the prompt (generation prefix),
            # not the sampled completion, so only look for the closing tag here.
            think_end = raw_full.find("</think>")

            if think_end != -1:
                thinking_text = raw_full[:think_end]
                thinking_token_count = len(self.tokenizer.encode(thinking_text, add_special_tokens=False))
                all_contents.append(raw_full)
                all_thinking_counts.append(thinking_token_count)
            else:
                # Max tokens hit mid-thinking — force close thinking and prefill <answer>
                thinking_token_count = len(seq.tokens)
                forced_prompt = tinker.ModelInput.from_ints(
                    list(prompt.to_ints()) + list(seq.tokens) + force_suffix_tokens
                )
                forced_params = self.SamplingParams(
                    max_tokens=512, temperature=self.temperature, top_p=self.top_p, stop=stop_sequences
                )
                forced_output = self.sampling_client.sample(forced_prompt, sampling_params=forced_params, num_samples=1).result()
                forced_text = self.tokenizer.decode(forced_output.sequences[0].tokens)
                all_contents.append(raw_full + force_suffix + forced_text)
                all_thinking_counts.append(thinking_token_count)

        return all_contents, all_thinking_counts


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
        # Checked relative to this script's own location, so this works
        # regardless of which directory the repo is cloned into.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        nq_candidates = [
            os.path.join(script_dir, "NQ-open.dev.jsonl"),
            os.path.join(script_dir, "NQ", "NQ-open.dev.jsonl"),
        ]
        nq_path = next((p for p in nq_candidates if os.path.exists(p)), None)
        if nq_path is None:
            raise FileNotFoundError(
                f"NQ-open.dev.jsonl not found. Download from "
                f"https://github.com/efficientqa/nq-open and place it at one of: {nq_candidates}"
            )
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
        # Download from: https://github.com/eladsegal/strategyqa
        # Checked relative to this script's own location, so this works
        # regardless of which directory the repo is cloned into.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        strategyqa_candidates = [
            os.path.join(script_dir, "strategyqa_train.json"),
            os.path.join(script_dir, "StrategyQA", "strategyqa_train.json"),
        ]
        strategyqa_path = next((p for p in strategyqa_candidates if os.path.exists(p)), None)
        if strategyqa_path is None:
            raise FileNotFoundError(
                f"strategyqa_train.json not found. Download from "
                f"https://github.com/eladsegal/strategyqa and place it at one of: {strategyqa_candidates}"
            )
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


def run_evaluation(model_name, dataset_name, use_thinking, temperature, top_p, output_dir, checkpoint_uri=None, inference_name=None, limit=None, seed=None, num_samples=4):
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

    samples_str = f"pass{num_samples}"
    seed_str = f"_seed{seed}" if seed is not None else ""

    output_file = Path(output_dir) / f"{dataset_name}_{model_short}_{thinking_str}_{samples_str}{seed_str}.json"

    print(f"Running evaluation: {model_name} on {dataset_name}")
    print(f"  Thinking: {use_thinking}")
    print(f"  Temperature: {temperature}, Top-p: {top_p}")
    print(f"  Output: {output_file}")

    # Load dataset
    sample = load_dataset(dataset_name)
    if seed is not None:
        random.Random(seed).shuffle(sample)
    if limit:
        sample = sample[:limit]
    print(f"Loaded {len(sample)} questions.")

    # Build system instruction (custom for strategyqa)
    system_instruction = None
    if dataset_name == "strategyqa":
        system_instruction = "You will be given a yes/no question. Answer with True or False. Give your final answer in <answer></answer> tags."

    # Initialize evaluator
    if model_name == "gpt-oss-20b":
        evaluator = GPTOSSEvaluator(temperature, top_p, checkpoint_uri, inference_name, model_name='openai/gpt-oss-20b')
    elif model_name == "qwen3-8b":
        evaluator = Qwen3Evaluator(temperature, top_p, checkpoint_uri, inference_name)
    elif model_name == "nemotron3":
        evaluator = Nemotron3NanoEvaluator(temperature, top_p, checkpoint_uri, inference_name)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Resume from existing file if present
    completed_questions = set()
    results = []
    errors = []
    token_lengths = []
    recalled_token_lengths = []
    not_recalled_token_lengths = []
    success = 0
    recall_hits = 0
    pass_at_4_hits = 0
    pass_at_n_questions = 0  # questions where at least 1 of N samples recalled
    em_at_n_questions = 0    # questions where at least 1 of N samples exact matched

    if output_file.exists():
        with open(output_file) as f:
            existing = json.load(f)
        results = existing.get("results", [])
        errors = existing.get("errors", [])
        for r in results:
            completed_questions.add(r["question"])
            toks = r.get("num_tokens", 0)
            rec = r.get("recall_1", r.get("recall", False))
            em = r.get("exact_match_1", r.get("exact_match", False))
            token_lengths.append(toks)
            if rec:
                recalled_token_lengths.append(toks)
                recall_hits += 1
            else:
                not_recalled_token_lengths.append(toks)
            if em:
                success += 1
            pass_n_rec = r.get(f"pass_at_{num_samples}_recall", rec)
            pass_n_em = r.get(f"pass_at_{num_samples}_em", em)
            if pass_n_rec:
                pass_at_n_questions += 1
            if pass_n_em:
                em_at_n_questions += 1
        print(f"Resuming from {len(results)} completed questions, {len(errors)} errors.")

    from collections import defaultdict
    completed_counts = defaultdict(int)
    for r in results:
        completed_counts[r["question"]] += 1
    occurrence = defaultdict(int)
    remaining = []
    for item in sample:
        q = item["question"]
        if occurrence[q] < completed_counts[q]:
            occurrence[q] += 1
        else:
            remaining.append(item)
    sample = remaining
    print(f"Remaining: {len(sample)} questions.")

    for idx, item in tqdm(enumerate(sample), total=len(sample)):
        try:
            gen_output = evaluator.generate(item["question"], item["answer_aliases"], use_thinking, num_samples=num_samples, system_instruction=system_instruction)

            # GPTOSSEvaluator returns (list_of_contents, list_of_token_counts)
            # Other evaluators return (text, token_count)
            if isinstance(gen_output[0], list):
                all_texts, all_token_counts = gen_output
            else:
                all_texts = [gen_output[0]]
                all_token_counts = [gen_output[1]]

            # Primary sample (first one) used for main metrics
            text = all_texts[0]
            token_count = all_token_counts[0]

            extracted = extract_answer(text)
            ground_truths = item["answer_aliases"]

            em = exact_match_score(extracted, ground_truths)
            rec = recall_score(extracted, ground_truths)

            token_lengths.append(token_count)
            if rec:
                recalled_token_lengths.append(token_count)
            else:
                not_recalled_token_lengths.append(token_count)

            # Process all samples
            sample_recall_count = 0
            sample_em_count = 0
            result_entry = {
                "question": item["question"],
                "ground_truth": ground_truths,
            }
            for si, (s_text, s_tokens) in enumerate(zip(all_texts, all_token_counts), 1):
                s_extracted = extract_answer(s_text)
                s_rec = recall_score(s_extracted, ground_truths)
                s_em = exact_match_score(s_extracted, ground_truths)
                if s_rec:
                    sample_recall_count += 1
                if s_em:
                    sample_em_count += 1
                result_entry[f"raw_prediction_{si}"] = s_text
                result_entry[f"extracted_prediction_{si}"] = s_extracted
                result_entry[f"num_tokens_{si}"] = s_tokens
                result_entry[f"recall_{si}"] = s_rec
                result_entry[f"exact_match_{si}"] = s_em

            n_samples = len(all_texts)
            result_entry[f"recall_at_{n_samples}"] = f"{sample_recall_count}/{n_samples}"
            result_entry[f"em_at_{n_samples}"] = f"{sample_em_count}/{n_samples}"
            result_entry[f"pass_at_{n_samples}_recall"] = sample_recall_count > 0
            result_entry[f"pass_at_{n_samples}_em"] = sample_em_count > 0

            # Majority vote: most frequent extracted answer wins; tie → incorrect
            all_preds = [result_entry.get(f"extracted_prediction_{i}") for i in range(1, n_samples + 1)]
            from collections import Counter as _Counter
            norm_preds = [normalize_answer(p) if p else "" for p in all_preds]
            non_empty = [p for p in norm_preds if p]
            if non_empty:
                counts = _Counter(non_empty)
                top_count = counts.most_common(1)[0][1]
                top_answers = [a for a, c in counts.items() if c == top_count]
                maj_winner = None if len(top_answers) > 1 else top_answers[0]
            else:
                maj_winner = None
            result_entry[f"majority_at_{n_samples}_vote"] = maj_winner
            result_entry[f"majority_at_{n_samples}_em"] = exact_match_score(maj_winner, ground_truths) if maj_winner else False
            result_entry[f"majority_at_{n_samples}_recall"] = recall_score(maj_winner, ground_truths) if maj_winner else False

            # Keep top-level fields for backwards compatibility
            result_entry["raw_prediction"] = text
            result_entry["extracted_prediction"] = extracted
            result_entry["num_tokens"] = token_count
            result_entry["exact_match"] = em
            result_entry["recall"] = rec
            results.append(result_entry)

            if em:
                success += 1
            if rec:
                recall_hits += 1
            pass_at_4_hits += sample_recall_count
            if sample_recall_count > 0:
                pass_at_n_questions += 1
            if sample_em_count > 0:
                em_at_n_questions += 1

        except Exception as e:
            errors.append({
                "question": item["question"],
                "error": str(e),
            })

        # Save after each question
        n_completed = len(results) if results else 1
        aggregate_metrics = {
            "total_questions": len(sample),
            "num_samples_per_question": num_samples,
            # @1 metrics (first sample only)
            "exact_match_at_1": success,
            "recall_at_1": recall_hits,
            "exact_match_at_1_rate": success / len(sample),
            "recall_at_1_rate": recall_hits / len(sample),
            # pass@N metrics (at least 1 of N samples correct)
            f"pass_at_{num_samples}_recall": pass_at_n_questions,
            f"pass_at_{num_samples}_em": em_at_n_questions,
            f"pass_at_{num_samples}_recall_rate": pass_at_n_questions / n_completed,
            f"pass_at_{num_samples}_em_rate": em_at_n_questions / n_completed,
            # total hits across all samples
            f"total_recall_hits_at_{num_samples}": pass_at_4_hits,
            f"avg_recall_per_question_at_{num_samples}": pass_at_4_hits / n_completed,
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

        tmp_file = output_file.with_suffix(".tmp")
        with open(tmp_file, "w") as f:
            json.dump(
                {
                    "results": results,
                    "aggregate_metrics": aggregate_metrics,
                    "errors": errors,
                },
                f,
                indent=2,
            )
        os.replace(tmp_file, output_file)

    print(f"Finished all questions. Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run QA evaluation on various models")
    parser.add_argument("--model", type=str, required=True,
                        choices=["gpt-oss-20b", "qwen3-8b", "nemotron3"],
                        help="Model to evaluate")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["triviaqa", "nq", "hotpotqa", "simpleqa", "strategyqa", "all"],
                        help="Dataset to use ('all' to run on all 5 datasets)")
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
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples per question (default: 4)")

    args = parser.parse_args()

    use_thinking = args.thinking == "yes"

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    datasets = ["triviaqa", "nq", "hotpotqa", "simpleqa", "strategyqa"] if args.dataset == "all" else [args.dataset]

    for dataset_name in datasets:
        run_evaluation(
            model_name=args.model,
            dataset_name=dataset_name,
            use_thinking=use_thinking,
            temperature=args.temperature,
            top_p=args.top_p,
            output_dir=args.output_dir,
            checkpoint_uri=args.checkpoint_uri,
            inference_name=args.inference_name,
            limit=args.limit,
            seed=args.seed,
            num_samples=args.num_samples
        )


if __name__ == "__main__":
    main()
