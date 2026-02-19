#!/usr/bin/env python3
"""
Script to perform McNemar's test on paired binary outcomes between two results
trace files.

This script:
1. Loads two JSON or JSONL files with results
2. Pairs results between files (by question/problem text or by index)
3. Builds the 2×2 McNemar contingency table for each available metric
4. Applies McNemar's test (with continuity correction; exact binomial for small n)
5. Reports the test statistic, p-value, and significance

Supports multiple file formats:
- JSON with 'results' array: {"results": [...]}
- JSONL: each line is a JSON object

Supports multiple datasets:
- NQ/TriviaQA: Uses 'recall'/'recalled' and optionally 'exact_match'/'em'
  Pairs by 'question' field
- MATH: Uses 'correct' field for accuracy
  Pairs by 'problem' field
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from scipy import stats


def load_json_file(file_path: str) -> List[Dict]:
    """
    Load and return results from a JSON or JSONL file.
    
    Supports two formats:
    1. JSON with a 'results' array: {"results": [...]}
    2. JSONL: each line is a JSON object
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(path, 'r') as f:
        # Try to read as standard JSON first
        try:
            data = json.load(f)
            # If it's a dict with 'results' key, return that
            if isinstance(data, dict) and 'results' in data:
                return data.get('results', [])
            # If it's already a list, return it
            elif isinstance(data, list):
                return data
            # Otherwise, wrap in list
            else:
                return [data]
        except json.JSONDecodeError:
            # If JSON parsing fails, try JSONL format
            f.seek(0)
            results = []
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON line {line_num}: {e}")
            return results


def pair_results(results1: List[Dict], results2: List[Dict],
                 pair_by_question: bool = True) -> List[Tuple[Dict, Dict]]:
    """
    Pair results from two files.

    Args:
        results1: Results from first file
        results2: Results from second file
        pair_by_question: If True, pair by question text; if False, pair by index

    Returns:
        List of (result1, result2) tuples
    """
    if pair_by_question:
        # Detect question field name (either 'question' for NQ/TriviaQA or 'problem' for MATH)
        sample_result = results1[0] if results1 else {}
        question_field = 'problem' if 'problem' in sample_result else 'question'

        # Create a mapping from question to result for file2
        results2_map = {}
        for r2 in results2:
            question = r2.get(question_field, '')
            if question:
                results2_map[question] = r2

        # Pair results by question
        pairs = []
        for r1 in results1:
            question = r1.get(question_field, '')
            if question in results2_map:
                pairs.append((r1, results2_map[question]))
            else:
                print(f"Warning: {question_field.capitalize()} not found in second file: {question[:50]}...")

        if len(pairs) < len(results1):
            print(f"Warning: Only {len(pairs)}/{len(results1)} results paired by {question_field}")
    else:
        # Pair by index
        min_len = min(len(results1), len(results2))
        pairs = [(results1[i], results2[i]) for i in range(min_len)]

        if len(results1) != len(results2):
            print(f"Warning: Files have different lengths ({len(results1)} vs {len(results2)}). "
                  f"Only paired {min_len} results.")

    return pairs


def get_metric_value(result: Dict, metric: str) -> float:
    """
    Get metric value from a result, handling different field names.

    Args:
        result: Result dictionary
        metric: Either 'recall', 'exact_match' (or 'em'), or 'correct'

    Returns:
        Float value (1.0 for True, 0.0 for False)
    """
    if metric == 'em':
        metric = 'exact_match'

    if metric == 'recall':
        # Check both 'recall' and 'recalled' fields
        if 'recalled' in result:
            return 1.0 if result.get('recalled', False) else 0.0
        else:
            return 1.0 if result.get('recall', False) else 0.0
    elif metric == 'exact_match':
        # Check both 'exact_match' and 'em' fields
        if 'exact_match' in result:
            return 1.0 if result.get('exact_match', False) else 0.0
        elif 'em' in result:
            return 1.0 if result.get('em', False) else 0.0
        else:
            return 0.0  # Field doesn't exist
    elif metric == 'correct':
        # For MATH dataset
        return 1.0 if result.get('correct', False) else 0.0
    else:
        return 1.0 if result.get(metric, False) else 0.0


def build_contingency_table(pairs: List[Tuple[Dict, Dict]],
                            metric: str) -> Tuple[int, int, int, int]:
    """
    Build the 2×2 McNemar contingency table for a binary metric.

    The four cells are:
      a = both correct          (file1=1, file2=1)
      b = only file1 correct    (file1=1, file2=0)
      c = only file2 correct    (file1=0, file2=1)
      d = both incorrect        (file1=0, file2=0)

    Args:
        pairs: List of (result1, result2) tuples
        metric: Metric name

    Returns:
        (a, b, c, d) counts
    """
    a = b = c = d = 0
    for r1, r2 in pairs:
        v1 = int(get_metric_value(r1, metric))
        v2 = int(get_metric_value(r2, metric))
        if v1 == 1 and v2 == 1:
            a += 1
        elif v1 == 1 and v2 == 0:
            b += 1
        elif v1 == 0 and v2 == 1:
            c += 1
        else:
            d += 1
    return a, b, c, d


def mcnemar_test(pairs: List[Tuple[Dict, Dict]],
                 metric: str) -> Dict:
    """
    Perform McNemar's test on paired binary outcomes.

    Uses the mid-p exact binomial test when b+c < 25 (sparse discordant cells),
    and the chi-squared approximation with continuity correction otherwise.

    Args:
        pairs: List of (result1, result2) tuples
        metric: Metric name

    Returns:
        Dictionary with keys: a, b, c, d, n, statistic, p_value, method
    """
    a, b, c, d = build_contingency_table(pairs, metric)
    n = a + b + c + d
    discordant = b + c

    if discordant == 0:
        # Perfect agreement on discordant pairs — no test possible
        return dict(a=a, b=b, c=c, d=d, n=n,
                    statistic=None, p_value=None, method='none')

    if discordant < 25:
        # Exact binomial test: under H0, b ~ Binomial(b+c, 0.5)
        # Two-sided p-value
        p_value = stats.binom_test(b, discordant, 0.5, alternative='two-sided')
        method = 'exact (binomial)'
        statistic = None
    else:
        # Chi-squared with continuity correction
        chi2 = (abs(b - c) - 1) ** 2 / discordant
        p_value = 1 - stats.chi2.cdf(chi2, df=1)
        method = 'chi-squared (continuity correction)'
        statistic = chi2

    return dict(a=a, b=b, c=c, d=d, n=n,
                statistic=statistic, p_value=p_value, method=method)


def print_results(metric: str, result: Dict, alpha: float = 0.05):
    """Print formatted McNemar's test results for a metric."""
    if metric == 'recall':
        metric_label = 'Recall'
    elif metric == 'exact_match':
        metric_label = 'Exact Match (EM)'
    elif metric == 'correct':
        metric_label = 'Accuracy'
    else:
        metric_label = metric.capitalize()

    a, b, c, d = result['a'], result['b'], result['c'], result['d']
    n = result['n']
    p_value = result['p_value']
    statistic = result['statistic']
    method = result['method']

    acc1 = (a + b) / n if n > 0 else 0.0
    acc2 = (a + c) / n if n > 0 else 0.0

    print(f"\n{'=' * 80}")
    print(f"{metric_label} — McNemar's Test")
    print(f"{'=' * 80}")
    print(f"Number of paired examples : {n}")
    print(f"  File1 rate              : {acc1:.4f}  ({a + b}/{n})")
    print(f"  File2 rate              : {acc2:.4f}  ({a + c}/{n})")
    print(f"\nContingency table (File1 \\ File2):")
    print(f"                 File2 correct   File2 incorrect")
    print(f"  File1 correct      {a:>6}           {b:>6}   (b)")
    print(f"  File1 incorrect    {c:>6}   (c)      {d:>6}")
    print(f"\nDiscordant pairs: b={b}, c={c}  (b+c={b+c})")
    print(f"Method: {method}")

    if p_value is None:
        print("  → b+c = 0: both systems agree on every example; test not applicable.")
    else:
        if statistic is not None:
            print(f"χ²  = {statistic:.4f}")
        print(f"p   = {p_value:.6f}")
        if p_value < alpha:
            winner = "File1" if b > c else "File2"
            print(f"  → Significant difference at α={alpha}: {winner} performs better "
                  f"(p={p_value:.4f} < {alpha})")
        else:
            print(f"  → No significant difference at α={alpha} "
                  f"(p={p_value:.4f} ≥ {alpha})")

    print(f"{'=' * 80}")


def compute_summary_stats(results: List[Dict], file_name: str):
    """Compute and print summary statistics for a file."""
    if not results:
        print(f"No results in {file_name}")
        return

    # Detect dataset type (MATH uses 'correct', NQ/TriviaQA use 'recall'/'recalled')
    sample_result = results[0] if results else {}
    is_math_dataset = 'correct' in sample_result

    print(f"\n{file_name}:")
    print(f"  Total examples: {len(results)}")

    if is_math_dataset:
        # MATH dataset - only has 'correct' field
        corrects = [get_metric_value(r, 'correct') for r in results]
        correct_mean = np.mean(corrects)
        print(f"  Accuracy: {correct_mean:.4f} ({sum(corrects):.0f}/{len(results)})")
    else:
        # NQ/TriviaQA dataset - has 'recall'/'recalled' and optionally 'exact_match'/'em'
        recalls = [get_metric_value(r, 'recall') for r in results]
        ems = [get_metric_value(r, 'exact_match') for r in results]

        recall_mean = np.mean(recalls)
        em_mean = np.mean(ems) if any(ems) else None

        print(f"  Recall rate: {recall_mean:.4f} ({sum(recalls):.0f}/{len(results)})")
        if em_mean is not None:
            print(f"  EM rate: {em_mean:.4f} ({sum(ems):.0f}/{len(results)})")
        else:
            print(f"  EM rate: N/A (field not found in results)")


def main():
    parser = argparse.ArgumentParser(
        description="Perform McNemar's test on paired binary outcomes "
                    "(recall/EM/accuracy) between two results trace files."
    )
    parser.add_argument(
        "input_file1",
        type=str,
        help="Path to the first JSON file containing reasoning traces"
    )
    parser.add_argument(
        "input_file2",
        type=str,
        help="Path to the second JSON file containing reasoning traces"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level α (default: 0.05)"
    )
    parser.add_argument(
        "--pair-by-index",
        action="store_true",
        help="Pair results by index instead of by question text (default: pair by question)"
    )

    args = parser.parse_args()

    # Load both files
    print(f"Reading {args.input_file1}...")
    try:
        results1 = load_json_file(args.input_file1)
        print(f"Found {len(results1)} results")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print(f"\nReading {args.input_file2}...")
    try:
        results2 = load_json_file(args.input_file2)
        print(f"Found {len(results2)} results")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Print summary statistics
    compute_summary_stats(results1, args.input_file1)
    compute_summary_stats(results2, args.input_file2)

    # Pair results
    pair_method = "index" if args.pair_by_index else "question"
    print(f"\nPairing results by {pair_method}...")
    pairs = pair_results(results1, results2, pair_by_question=not args.pair_by_index)

    if len(pairs) == 0:
        print("Error: No pairs could be formed. Check that files have matching questions or indices.")
        return

    print(f"Successfully paired {len(pairs)} results")

    # Check which metrics are available
    available_metrics = []
    sample_result = pairs[0][0] if pairs else {}

    # Check for MATH dataset ('correct' field)
    if 'correct' in sample_result:
        available_metrics.append('correct')
    else:
        # NQ/TriviaQA dataset - check for recall/recalled
        if 'recall' in sample_result or 'recalled' in sample_result:
            available_metrics.append('recall')

        # Check for exact_match/em
        if 'exact_match' in sample_result or 'em' in sample_result:
            available_metrics.append('exact_match')

    if not available_metrics:
        print("Warning: No recognized metric fields found. Looking for 'correct' (MATH), "
              "'recall'/'recalled', 'exact_match', or 'em'")
        return

    # Run McNemar's test for each available metric
    for metric in available_metrics:
        result = mcnemar_test(pairs, metric)
        print_results(metric, result, alpha=args.alpha)


if __name__ == "__main__":
    main()
