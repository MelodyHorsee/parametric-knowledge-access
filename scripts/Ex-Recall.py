#!/usr/bin/env python3
"""
Refine and evaluate extracted answers using GPT-5-mini.

This script takes evaluation results and uses GPT-5-mini to refine extracted
answers before computing recall scores against ground truth.
"""
import json
import os
import re
import string
import argparse
from openai import OpenAI
from tqdm import tqdm

# Default paths (relative to script location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "results")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "..", "extracted_results")

def normalize_answer(s):
    def normalize_unicode_dashes(text):
        return re.sub(r"[‐-‒–—―−]", " ", text)

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

    return white_space_fix(
        remove_articles(
            handle_punc(
                lower(
                    normalize_unicode_dashes(
                        replace_underscore(s)
                    )
                )
            )
        )
    ).strip()

def recall_score(prediction, ground_truths):
    """Check if any ground truth appears in the prediction"""
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

few_shot_prompt = """
You are given an answer that may contain one or multiple possibilities.
If it only contains one, just output it as is.
Otherwise, choose the answer that is stated with the most confidence, if there are multiple options.
DO NOT correct the answer, even if you think it's incorrect.
If the answer is just a question being repeated (not an actual answer), keep it exactly as is.
Examples:

A: While Leif Erikson reached North America earlier, Christopher Columbus is usually cited.
Refined Answer: Christopher Columbus

A: While some might think Saturn, the largest planet is Jupiter.
Refined Answer: Jupiter

A: It could be Paris, but some might mistakenly say Lyon.
Refined Answer: Paris

A: Leonardo da Vinci painted the Mona Lisa.
Refined Answer: Leonardo da Vinci painted the Mona Lisa.

A: Shanghai is the capital of China.
Refined Answer: Shanghai is the capital of China.

A: What is the capital of France?
Refined Answer: What is the capital of France?

Original Answer: {answer}
Refined Answer:
"""

def main():
    parser = argparse.ArgumentParser(
        description="Refine and evaluate extracted answers using GPT-5-mini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process specific files
  %(prog)s --target-files triviaqa_olmo3_no_cues_with_thinking.json nq_olmo3_no_cues_with_thinking.json

  # Process all .json files in results directory
  %(prog)s --all

  # Specify custom results directory
  %(prog)s --all --results-dir /path/to/results
        """
    )

    file_group = parser.add_mutually_exclusive_group(required=True)
    file_group.add_argument(
        "--target-files",
        nargs="+",
        help="Specific result files to process (e.g., triviaqa_olmo3_no_cues_with_thinking.json)"
    )
    file_group.add_argument(
        "--all",
        action="store_true",
        help="Process all .json files in the results directory"
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing result files (default: %(default)s)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save extracted results (default: %(default)s)"
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start processing from this index (useful for resuming after errors)"
    )

    args = parser.parse_args()

    # Initialize OpenAI client
    client = OpenAI()

    # Setup directories
    results_dir = args.results_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        exit(1)

    # Determine which files to process
    if args.all:
        target_files = None  # Process all
        print("Processing all .json files in results directory...")
    else:
        target_files = set(args.target_files)
        print(f"Processing {len(target_files)} specified file(s)...")

    processed_count = 0
    for file_name in os.listdir(results_dir):
        if not file_name.endswith(".json"):
            continue
        if target_files and file_name not in target_files:
            continue

        file_path = os.path.join(results_dir, file_name)
        output_path = os.path.join(output_dir, f"ex_{file_name}")

        print(f"\nProcessing {file_name}...")

        with open(file_path, "r") as f:
            data_json = json.load(f)

        data = data_json["results"]
        print(f"  Total items: {len(data)}")

        # Load existing results if file exists and we're resuming
        if os.path.exists(output_path) and args.start_index > 0:
            print(f"  Loading existing results from {output_path}")
            with open(output_path, "r") as f:
                existing_data = json.load(f)
                results = existing_data.get("results", [])
            print(f"  Resuming from index {args.start_index} (already have {len(results)} results)")
        else:
            results = []

        # Create progress bar starting from start_index
        pbar = tqdm(total=len(data), desc=f"Processing {file_name}", initial=args.start_index)

        for idx, item in enumerate(data):
            # Skip items before start_index
            if idx < args.start_index:
                continue
            extracted = item.get("extracted_prediction", "")
            ground_truth = item.get("ground_truth", [])

            # Skip if no extracted prediction
            if not extracted or extracted == "":
                record = {
                    "question": item.get("question", ""),
                    "original_extracted": extracted,
                    "refined_answer": "",
                    "ground_truth": ground_truth,
                    "recalled": False,
                    "skipped": True
                }
                results.append(record)
                continue

            prompt = few_shot_prompt.format(answer=extracted)

            try:
                response = client.responses.create(
                    model="gpt-5-mini-2025-08-07",
                    reasoning={"effort": "medium"},
                    input=[{"role": "user", "content": prompt}]
                )

                refined_answer = response.output_text.strip()
                recalled = recall_score(refined_answer, ground_truth)

                record = {
                    "question": item.get("question", ""),
                    "original_extracted": extracted,
                    "refined_answer": refined_answer,
                    "ground_truth": ground_truth,
                    "recalled": recalled,
                    "skipped": False
                }

            except Exception as e:
                print(f"  Error processing item {idx}: {e}")
                record = {
                    "question": item.get("question", ""),
                    "original_extracted": extracted,
                    "refined_answer": "",
                    "ground_truth": ground_truth,
                    "recalled": False,
                    "error": str(e)
                }

            results.append(record)

            # Update progress bar
            pbar.update(1)

            # Save progress every 10 items
            if (idx + 1) % 10 == 0:
                with open(output_path, "w") as out_f:
                    json.dump({"results": results}, out_f, indent=2)

        # Close progress bar
        pbar.close()

        # Final write with all results
        with open(output_path, "w") as out_f:
            json.dump({"results": results}, out_f, indent=2)

        print(f"✓ Finished {file_name} -> {output_path}")
        processed_count += 1

    print(f"\n{'='*50}")
    print(f"All done! Processed {processed_count} files.")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
