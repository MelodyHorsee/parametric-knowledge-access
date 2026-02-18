#!/usr/bin/env python3
"""
Extract thinking token counts from existing evaluation results.
Uses refined recall labels from ex_ files.
"""
import re
import json
import argparse
from pathlib import Path
from transformers import AutoTokenizer
from tinker_cookbook import tokenizer_utils


def get_tokenizer(model_type):
    """Load the appropriate tokenizer for each model type."""
    if model_type == "gpt-oss-20b":
        return tokenizer_utils.get_tokenizer('openai/gpt-oss-20b')
    elif model_type == "olmo-3-7b-think":
        return AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Think")
    elif model_type == "r1-distill-qwen-1.5b":
        return AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def extract_thinking_text(raw_prediction, model_type):
    """Extract thinking portion from raw prediction."""
    if model_type == "gpt-oss-20b":
        # Format: <|channel|>analysis<|message|>...thinking...<|end|>
        match = re.search(
            r'<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>',
            raw_prediction,
            re.DOTALL
        )
        if match:
            return match.group(1)
        return ""

    elif model_type in ["olmo-3-7b-think", "r1-distill-qwen-1.5b"]:
        # Format: ...thinking...</think>
        # Extract everything BEFORE </think> (not including the tag)
        # Note: newlines after </think> vary (\n\n or \n), so just match up to </think>
        match = re.search(r'^(.*?)(?=</think>)', raw_prediction, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1)
        # If no </think> tag found, model hit token limit while still thinking
        # Treat entire output as thinking
        return raw_prediction

    return ""


def detect_model_type(filepath):
    """Detect model type from filename."""
    filename = Path(filepath).name.lower()
    if "gptoss" in filename or "gpt-oss" in filename:
        return "gpt-oss-20b"
    elif "olmo" in filename:
        return "olmo-3-7b-think"
    elif "r1qwen" in filename:
        return "r1-distill-qwen-1.5b"
    return None


def find_ex_file(original_path):
    """Find the corresponding ex_ file for an original results file."""
    path = Path(original_path)
    ex_filename = "ex_" + path.name
    ex_path = path.parent / ex_filename
    if ex_path.exists():
        return ex_path
    return None


def load_ex_recall_list(ex_path):
    """Load ex_ file and return list of recalled values (by index)."""
    with open(ex_path) as f:
        data = json.load(f)

    return [item.get("recalled", False) for item in data.get("results", [])]


def process_file_pair(original_path, model_type, use_tokenizer=True):
    """Process an original file paired with its ex_ file.

    Logic:
    - Match ex_ file with original by filename (ex_ prefix)
    - Use index-based matching (not question) since there are duplicate questions
    - Calculate thinking tokens from original file's raw_prediction
    - Use recalled labels from ex_ file
    """
    # Load original file
    with open(original_path) as f:
        original_data = json.load(f)

    # Find and load ex_ file
    ex_path = find_ex_file(original_path)
    if ex_path:
        print(f"  Found ex_ file: {ex_path.name}")
        recall_list = load_ex_recall_list(ex_path)
    else:
        print(f"  Warning: No ex_ file found, using original recall labels")
        recall_list = None

    # Load tokenizer if needed
    tokenizer = None
    if use_tokenizer:
        print(f"  Loading tokenizer for {model_type}...")
        tokenizer = get_tokenizer(model_type)

    results = original_data.get("results", [])

    # Verify lengths match if we have ex_ file
    if recall_list is not None and len(recall_list) != len(results):
        print(f"  Warning: Length mismatch - original: {len(results)}, ex_: {len(recall_list)}")

    all_thinking_tokens = []
    recalled_thinking_tokens = []
    not_recalled_thinking_tokens = []

    for idx, item in enumerate(results):
        raw_pred = item.get("raw_prediction", "")

        # Extract thinking text
        thinking_text = extract_thinking_text(raw_pred, model_type)

        # Get token count
        if tokenizer:
            thinking_tokens = len(tokenizer.encode(thinking_text, add_special_tokens=False))
        else:
            thinking_tokens = len(thinking_text)  # fallback to char count

        all_thinking_tokens.append(thinking_tokens)

        # Get recall/correct status from ex_ file by index, otherwise from original
        if recall_list is not None and idx < len(recall_list):
            recalled = recall_list[idx]
        else:
            # "recall" for QA datasets, "correct" for MATH
            recalled = item.get("recall", item.get("correct", False))

        if recalled:
            recalled_thinking_tokens.append(thinking_tokens)
        else:
            not_recalled_thinking_tokens.append(thinking_tokens)

    # Compute stats
    stats = {
        "total_samples": len(results),
        "avg_thinking_tokens": sum(all_thinking_tokens) / len(all_thinking_tokens) if all_thinking_tokens else 0,
        "num_recalled": len(recalled_thinking_tokens),
        "num_not_recalled": len(not_recalled_thinking_tokens),
        "avg_thinking_tokens_recalled": sum(recalled_thinking_tokens) / len(recalled_thinking_tokens) if recalled_thinking_tokens else 0,
        "avg_thinking_tokens_not_recalled": sum(not_recalled_thinking_tokens) / len(not_recalled_thinking_tokens) if not_recalled_thinking_tokens else 0,
    }

    return stats


def process_directory(results_dir, model_type=None, use_tokenizer=True):
    """Process all result files in a directory structure."""
    results_dir = Path(results_dir)
    all_stats = {}

    # Find all JSON files that are NOT ex_ files
    for json_file in sorted(results_dir.rglob("*.json")):
        if json_file.name.startswith("ex_"):
            continue

        # Auto-detect model type if not specified
        detected_type = model_type or detect_model_type(json_file)
        if not detected_type:
            print(f"Skipping {json_file.name} - could not detect model type")
            continue

        print(f"\nProcessing: {json_file}")
        stats = process_file_pair(json_file, detected_type, use_tokenizer)

        relative_path = json_file.relative_to(results_dir)
        all_stats[str(relative_path)] = stats

        # Print stats
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Avg thinking tokens (all): {stats['avg_thinking_tokens']:.2f}")
        print(f"  Recalled ({stats['num_recalled']}): {stats['avg_thinking_tokens_recalled']:.2f}")
        print(f"  Not recalled ({stats['num_not_recalled']}): {stats['avg_thinking_tokens_not_recalled']:.2f}")

    return all_stats


def main():
    parser = argparse.ArgumentParser(description="Extract thinking token counts from evaluation results")
    parser.add_argument("input", type=str, help="Path to results JSON file or directory")
    parser.add_argument("--model-type", type=str, choices=["gpt-oss-20b", "olmo", "r1-qwen"],
                        help="Model type (auto-detected from filename if not specified)")
    parser.add_argument("--no-tokenizer", action="store_true",
                        help="Skip tokenizer, only compute character counts")
    parser.add_argument("--output", type=str, help="Output file path for aggregated stats")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Path not found: {input_path}")
        return

    if input_path.is_dir():
        all_stats = process_directory(input_path, args.model_type, not args.no_tokenizer)
    else:
        model_type = args.model_type or detect_model_type(args.input)
        if not model_type:
            print("Error: Could not detect model type. Please specify with --model-type")
            return

        print(f"Processing: {input_path}")
        stats = process_file_pair(input_path, model_type, not args.no_tokenizer)
        all_stats = {input_path.name: stats}

        print(f"\n=== Thinking Token Stats ===")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Avg thinking tokens (all): {stats['avg_thinking_tokens']:.2f}")
        print(f"  Recalled ({stats['num_recalled']}): {stats['avg_thinking_tokens_recalled']:.2f}")
        print(f"  Not recalled ({stats['num_not_recalled']}): {stats['avg_thinking_tokens_not_recalled']:.2f}")

    # Save results if output specified
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_stats, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
