#!/usr/bin/env python3
"""
Refine and evaluate extracted answers using GPT-5-mini.

For pass1 files: refines extracted_prediction, reports ex_recall@1.
For passN files: refines each extracted_prediction_1..N, reports ex_recall@N
  (pass_at_N = True if any of the N refined answers is recalled).
"""
import json
import os
import re
import string
import time
import argparse
from openai import OpenAI
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(SCRIPT_DIR, "colm")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "eacl_ex_recall")
BATCH_MODEL = "gpt-5-mini-2025-08-07"

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
    pred = normalize_answer(prediction)
    for gt in ground_truths:
        gt_normalized = normalize_answer(gt)
        gt_escaped = re.escape(gt_normalized)
        gt_pattern = gt_escaped.replace(r'\ ', r'\s+')
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

def refine(client, extracted):
    """Call GPT-5-mini to refine a single extracted answer. Returns refined string."""
    if not extracted or extracted == "":
        return None
    prompt = few_shot_prompt.format(answer=extracted)
    response = client.responses.create(
        model="gpt-5-mini-2025-08-07",
        reasoning={"effort": "medium"},
        input=[{"role": "user", "content": prompt}]
    )
    return response.output_text.strip()


def iter_refine_targets(data, num_samples):
    """Yield (idx, si, extracted) for every (question, sample) slot that needs
    refining. si is always 1 for pass1 files. Shared by submit/collect so the
    two phases can't drift out of sync on which slots exist.
    """
    for idx, item in enumerate(data):
        if num_samples == 1:
            yield idx, 1, item.get("extracted_prediction", "")
        else:
            for si in range(1, num_samples + 1):
                yield idx, si, item.get(f"extracted_prediction_{si}", "")


def extract_output_text(response_body):
    """Reconstruct the equivalent of response.output_text from a raw Responses
    API JSON body (batch output doesn't give you the SDK convenience property).
    """
    for item in response_body.get("output", []):
        if item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if part.get("type") == "output_text":
                return part.get("text", "")
    return ""


def submit_batch(client, file_name, data, num_samples, output_dir):
    """Build and submit one Batch API job covering every refine() call needed
    for this file. Saves a small state file with the batch_id; use --collect
    later (batch jobs can take anywhere from minutes to 24h) to fetch results.
    """
    batch_state_path = os.path.join(output_dir, f"{file_name}.batch.json")
    if os.path.exists(batch_state_path):
        with open(batch_state_path) as f:
            existing = json.load(f)
        print(f"  Batch already submitted for {file_name}: {existing['batch_id']} "
              f"(status at submit time: {existing.get('status_at_submit', 'unknown')}). "
              f"Delete {batch_state_path} to resubmit.")
        return

    requests_path = os.path.join(output_dir, f"{file_name}.batch_input.jsonl")
    n_requests = 0
    with open(requests_path, "w") as f:
        for idx, si, extracted in iter_refine_targets(data, num_samples):
            if not extracted:
                continue  # matches refine()'s early-return for empty input; no API call needed
            prompt = few_shot_prompt.format(answer=extracted)
            line = {
                "custom_id": f"{idx}_{si}",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": BATCH_MODEL,
                    "reasoning": {"effort": "medium"},
                    "input": [{"role": "user", "content": prompt}],
                },
            }
            f.write(json.dumps(line) + "\n")
            n_requests += 1

    print(f"  Built {n_requests} batch requests -> {requests_path}")

    with open(requests_path, "rb") as f:
        batch_input_file = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"source_file": file_name},
    )

    with open(batch_state_path, "w") as f:
        json.dump({
            "batch_id": batch.id,
            "input_file_id": batch_input_file.id,
            "requests_path": requests_path,
            "num_samples": num_samples,
            "n_requests": n_requests,
            "status_at_submit": batch.status,
        }, f, indent=2)

    print(f"  Submitted batch {batch.id} ({n_requests} requests, status: {batch.status})")
    print(f"  State saved to {batch_state_path} — run with --collect once it's done.")


def collect_batch(client, file_name, data, num_samples, output_dir, wait=False, poll_seconds=30):
    """Check a submitted batch's status; if completed, download results,
    reconstruct the same record/aggregate schema the synchronous path
    produces, and write the final output file.
    """
    batch_state_path = os.path.join(output_dir, f"{file_name}.batch.json")
    if not os.path.exists(batch_state_path):
        print(f"  No submitted batch found for {file_name} (expected {batch_state_path}). "
              f"Run with --batch first.")
        return

    with open(batch_state_path) as f:
        state = json.load(f)

    batch = client.batches.retrieve(state["batch_id"])
    print(f"  Batch {batch.id} status: {batch.status} "
          f"({batch.request_counts.completed}/{batch.request_counts.total} completed, "
          f"{batch.request_counts.failed} failed)")

    if wait:
        while batch.status in ("validating", "in_progress", "finalizing"):
            time.sleep(poll_seconds)
            batch = client.batches.retrieve(state["batch_id"])
            print(f"  Batch {batch.id} status: {batch.status} "
                  f"({batch.request_counts.completed}/{batch.request_counts.total} completed, "
                  f"{batch.request_counts.failed} failed)")

    if batch.status in ("validating", "in_progress", "finalizing"):
        print("  Not done yet — rerun with --collect (or --collect --wait) later.")
        return
    if batch.status != "completed":
        print(f"  Batch ended with status '{batch.status}' — check the OpenAI dashboard for details.")
        return

    output_content = client.files.content(batch.output_file_id).text
    refined_by_custom_id = {}
    errors_by_custom_id = {}
    for line in output_content.splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        custom_id = entry["custom_id"]
        if entry.get("error"):
            errors_by_custom_id[custom_id] = str(entry["error"])
            continue
        response = entry.get("response") or {}
        if response.get("status_code") != 200:
            errors_by_custom_id[custom_id] = f"status_code={response.get('status_code')}"
            continue
        refined_by_custom_id[custom_id] = extract_output_text(response["body"]).strip()

    results = []
    ex_recall_hits = 0
    pass_at_n_hits = 0

    for idx, item in enumerate(data):
        ground_truth = item.get("ground_truth", [])
        question = item.get("question", "")

        if num_samples == 1:
            extracted = item.get("extracted_prediction", "")
            custom_id = f"{idx}_1"
            if not extracted:
                record = {"question": question, "original_extracted": extracted,
                          "refined_answer": "", "ground_truth": ground_truth,
                          "recalled": False, "skipped": True}
            elif custom_id in errors_by_custom_id:
                record = {"question": question, "original_extracted": extracted,
                          "refined_answer": "", "ground_truth": ground_truth,
                          "recalled": False, "error": errors_by_custom_id[custom_id]}
            else:
                refined = refined_by_custom_id.get(custom_id, "")
                recalled = recall_score(refined, ground_truth)
                record = {"question": question, "original_extracted": extracted,
                          "refined_answer": refined, "ground_truth": ground_truth,
                          "recalled": recalled, "skipped": False}
                if recalled:
                    ex_recall_hits += 1
                pass_at_n_hits += record["recalled"]
        else:
            record = {"question": question, "ground_truth": ground_truth}
            any_recalled = False
            for si in range(1, num_samples + 1):
                extracted = item.get(f"extracted_prediction_{si}", "")
                custom_id = f"{idx}_{si}"
                record[f"original_extracted_{si}"] = extracted
                if not extracted:
                    record[f"refined_answer_{si}"] = ""
                    record[f"recalled_{si}"] = False
                    record[f"skipped_{si}"] = True
                elif custom_id in errors_by_custom_id:
                    record[f"refined_answer_{si}"] = ""
                    record[f"recalled_{si}"] = False
                    record[f"error_{si}"] = errors_by_custom_id[custom_id]
                else:
                    refined = refined_by_custom_id.get(custom_id, "")
                    recalled = recall_score(refined, ground_truth)
                    record[f"refined_answer_{si}"] = refined
                    record[f"recalled_{si}"] = recalled
                    if recalled:
                        any_recalled = True

            record["recalled"] = record.get("recalled_1", False)
            record[f"pass_at_{num_samples}_ex_recall"] = any_recalled
            if record["recalled"]:
                ex_recall_hits += 1
            if any_recalled:
                pass_at_n_hits += 1

        results.append(record)

    n_done = len(results)
    aggregate = {
        "total_questions": len(data),
        "num_samples": num_samples,
        "ex_recall_at_1": ex_recall_hits,
        "ex_recall_at_1_rate": ex_recall_hits / n_done if n_done else 0,
        f"pass_at_{num_samples}_ex_recall": pass_at_n_hits,
        f"pass_at_{num_samples}_ex_recall_rate": pass_at_n_hits / n_done if n_done else 0,
    }

    output_path = os.path.join(output_dir, f"ex_{file_name}")
    with open(output_path, "w") as out_f:
        json.dump({"results": results, "aggregate": aggregate}, out_f, indent=2)

    print(f"  ✓ {file_name} -> {output_path}")
    print(f"    ex_recall@1: {ex_recall_hits}/{n_done} = {aggregate['ex_recall_at_1_rate']:.3f}")
    print(f"    pass@{num_samples} ex_recall: {pass_at_n_hits}/{n_done} = {aggregate[f'pass_at_{num_samples}_ex_recall_rate']:.3f}")
    if errors_by_custom_id:
        print(f"    {len(errors_by_custom_id)} request(s) errored — see 'error'/'error_N' fields in results.")


def detect_num_samples(data):
    """Detect N from keys like extracted_prediction_1 .. extracted_prediction_N."""
    if not data:
        return 1
    first = data[0]
    if "extracted_prediction_1" not in first:
        return 1
    n = 1
    while f"extracted_prediction_{n + 1}" in first:
        n += 1
    return n

def main():
    parser = argparse.ArgumentParser(
        description="Refine and evaluate extracted answers using GPT-5-mini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ex-recall.py --target-files triviaqa_qwen38b_no_cues_with_thinking_pass4.json
  python ex-recall.py --all --results-dir colm
        """
    )

    file_group = parser.add_mutually_exclusive_group(required=True)
    file_group.add_argument("--target-files", nargs="+",
                            help="Specific result files to process")
    file_group.add_argument("--all", action="store_true",
                            help="Process all .json files in the results directory")

    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-index", type=int, default=0,
                        help="Resume from this question index")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--batch", action="store_true",
                            help="Submit refine requests via the OpenAI Batch API instead of "
                                 "calling synchronously. Batch jobs run async (minutes to 24h); "
                                 "this just submits and saves state — use --collect to fetch results.")
    mode_group.add_argument("--collect", action="store_true",
                            help="Check status of a batch submitted with --batch, and if complete, "
                                 "download results and write the final output file.")

    parser.add_argument("--wait", action="store_true",
                        help="With --collect, poll until the batch finishes instead of checking once "
                             "and exiting. Prints status every --poll-seconds.")
    parser.add_argument("--poll-seconds", type=int, default=30,
                        help="Polling interval in seconds when using --collect --wait (default: 30)")

    args = parser.parse_args()

    client = OpenAI()
    results_dir = args.results_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        exit(1)

    if args.all:
        target_files = None
        print("Processing all .json files in results directory...")
    else:
        target_files = set(args.target_files)
        print(f"Processing {len(target_files)} specified file(s)...")

    processed_count = 0
    for file_name in sorted(os.listdir(results_dir)):
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
        num_samples = detect_num_samples(data)
        print(f"  Total items: {len(data)}, samples per question: {num_samples}")

        if args.batch:
            submit_batch(client, file_name, data, num_samples, output_dir)
            processed_count += 1
            continue
        if args.collect:
            collect_batch(client, file_name, data, num_samples, output_dir,
                          wait=args.wait, poll_seconds=args.poll_seconds)
            processed_count += 1
            continue

        # Resume support
        if os.path.exists(output_path) and args.start_index > 0:
            with open(output_path, "r") as f:
                existing = json.load(f)
            results = existing.get("results", [])
            print(f"  Resuming from index {args.start_index} ({len(results)} already done)")
        else:
            results = []

        pbar = tqdm(total=len(data), desc=file_name, initial=args.start_index)

        ex_recall_hits = sum(r["recalled"] for r in results)
        pass_at_n_hits = sum(r.get(f"pass_at_{num_samples}_ex_recall", r["recalled"]) for r in results)

        for idx, item in enumerate(data):
            if idx < args.start_index:
                continue

            ground_truth = item.get("ground_truth", [])
            question = item.get("question", "")

            if num_samples == 1:
                extracted = item.get("extracted_prediction", "")
                try:
                    refined = refine(client, extracted)
                    if refined is None:
                        record = {"question": question, "original_extracted": extracted,
                                  "refined_answer": "", "ground_truth": ground_truth,
                                  "recalled": False, "skipped": True}
                    else:
                        recalled = recall_score(refined, ground_truth)
                        record = {"question": question, "original_extracted": extracted,
                                  "refined_answer": refined, "ground_truth": ground_truth,
                                  "recalled": recalled, "skipped": False}
                        if recalled:
                            ex_recall_hits += 1
                        pass_at_n_hits += record["recalled"]
                except Exception as e:
                    record = {"question": question, "original_extracted": extracted,
                              "refined_answer": "", "ground_truth": ground_truth,
                              "recalled": False, "error": str(e)}
            else:
                record = {"question": question, "ground_truth": ground_truth}
                any_recalled = False
                for si in range(1, num_samples + 1):
                    extracted = item.get(f"extracted_prediction_{si}", "")
                    try:
                        refined = refine(client, extracted)
                        if refined is None:
                            record[f"refined_answer_{si}"] = ""
                            record[f"recalled_{si}"] = False
                            record[f"skipped_{si}"] = True
                        else:
                            recalled = recall_score(refined, ground_truth)
                            record[f"original_extracted_{si}"] = extracted
                            record[f"refined_answer_{si}"] = refined
                            record[f"recalled_{si}"] = recalled
                            if recalled:
                                any_recalled = True
                    except Exception as e:
                        record[f"refined_answer_{si}"] = ""
                        record[f"recalled_{si}"] = False
                        record[f"error_{si}"] = str(e)

                record["recalled"] = record.get("recalled_1", False)
                record[f"pass_at_{num_samples}_ex_recall"] = any_recalled
                if record["recalled"]:
                    ex_recall_hits += 1
                if any_recalled:
                    pass_at_n_hits += 1

            results.append(record)
            pbar.update(1)

            if (idx + 1) % 10 == 0:
                n_done = len(results)
                aggregate = {
                    "total_questions": len(data),
                    "num_samples": num_samples,
                    "ex_recall_at_1": ex_recall_hits,
                    "ex_recall_at_1_rate": ex_recall_hits / n_done,
                    f"pass_at_{num_samples}_ex_recall": pass_at_n_hits,
                    f"pass_at_{num_samples}_ex_recall_rate": pass_at_n_hits / n_done,
                }
                with open(output_path, "w") as out_f:
                    json.dump({"results": results, "aggregate": aggregate}, out_f, indent=2)

        pbar.close()

        n_done = len(results)
        aggregate = {
            "total_questions": len(data),
            "num_samples": num_samples,
            "ex_recall_at_1": ex_recall_hits,
            "ex_recall_at_1_rate": ex_recall_hits / n_done if n_done else 0,
            f"pass_at_{num_samples}_ex_recall": pass_at_n_hits,
            f"pass_at_{num_samples}_ex_recall_rate": pass_at_n_hits / n_done if n_done else 0,
        }
        with open(output_path, "w") as out_f:
            json.dump({"results": results, "aggregate": aggregate}, out_f, indent=2)

        print(f"✓ {file_name} -> {output_path}")
        print(f"  ex_recall@1: {ex_recall_hits}/{n_done} = {aggregate['ex_recall_at_1_rate']:.3f}")
        print(f"  pass@{num_samples} ex_recall: {pass_at_n_hits}/{n_done} = {aggregate[f'pass_at_{num_samples}_ex_recall_rate']:.3f}")
        processed_count += 1

    print(f"\n{'='*50}")
    print(f"Done! Processed {processed_count} files. Results in: {output_dir}")


if __name__ == "__main__":
    main()
