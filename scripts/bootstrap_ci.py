#!/usr/bin/env python3
"""
Script to compute paired bootstrap-resample confidence intervals on the difference
of means between two results trace files.

This script:
1. Loads two JSON or JSONL files with results
2. Pairs results between files (by question/problem text or by index)
3. Computes paired differences for available metrics
4. Performs bootstrap resampling to estimate confidence intervals
5. Reports the mean difference and confidence intervals

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


def compute_paired_differences(pairs: List[Tuple[Dict, Dict]], metric: str) -> np.ndarray:
    """
    Compute paired differences for a given metric.
    
    Args:
        pairs: List of (result1, result2) tuples
        metric: Either 'recall' or 'exact_match' (or 'em')
    
    Returns:
        Array of differences (value1 - value2) for each pair
    """
    differences = []
    for r1, r2 in pairs:
        val1 = get_metric_value(r1, metric)
        val2 = get_metric_value(r2, metric)
        differences.append(val1 - val2)
    
    return np.array(differences)


def bootstrap_resample(data: np.ndarray, n_bootstrap: int = 10000, 
                      confidence_level: float = 0.95) -> Tuple[float, float, float]:
    """
    Perform bootstrap resampling to compute confidence intervals.
    
    Args:
        data: Array of paired differences
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
    
    Returns:
        Tuple of (mean_difference, lower_bound, upper_bound)
    """
    n = len(data)
    if n == 0:
        raise ValueError("Cannot bootstrap empty data")
    
    # Compute observed mean difference
    observed_mean = np.mean(data)
    
    # Bootstrap resampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_mean = np.mean(bootstrap_sample)
        bootstrap_means.append(bootstrap_mean)
    
    bootstrap_means = np.array(bootstrap_means)
    
    # Compute confidence interval using percentile method
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_means, lower_percentile)
    upper_bound = np.percentile(bootstrap_means, upper_percentile)
    
    return observed_mean, lower_bound, upper_bound


def print_results(metric: str, mean_diff: float, lower_bound: float, upper_bound: float,
                  n_pairs: int, confidence_level: float = 0.95):
    """Print formatted results for a metric."""
    if metric == 'recall':
        metric_label = 'Recall'
    elif metric == 'exact_match':
        metric_label = 'Exact Match (EM)'
    elif metric == 'correct':
        metric_label = 'Accuracy'
    else:
        metric_label = metric.capitalize()

    confidence_pct = int(confidence_level * 100)

    print(f"\n{'=' * 80}")
    print(f"{metric_label} - Paired Bootstrap Confidence Intervals")
    print(f"{'=' * 80}")
    print(f"Number of paired examples: {n_pairs}")
    print(f"Mean difference (File1 - File2): {mean_diff:.6f}")
    print(f"{confidence_pct}% Confidence Interval: [{lower_bound:.6f}, {upper_bound:.6f}]")

    # Check if interval contains zero
    if lower_bound <= 0 <= upper_bound:
        print(f"  → Interval contains zero: difference is not statistically significant")
    elif mean_diff > 0:
        print(f"  → File1 has significantly higher {metric_label.lower()} (p < {1-confidence_level:.3f})")
    else:
        print(f"  → File2 has significantly higher {metric_label.lower()} (p < {1-confidence_level:.3f})")

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
        description="Compute paired bootstrap-resample confidence intervals on the "
                    "difference of means of recall/EM between two results trace files."
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
        "--n-bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap samples (default: 10000)"
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level for intervals (default: 0.95 for 95% CI)"
    )
    parser.add_argument(
        "--pair-by-index",
        action="store_true",
        help="Pair results by index instead of by question text (default: pair by question)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)"
    )
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
    
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
        print("Warning: No recognized metric fields found. Looking for 'correct' (MATH), 'recall'/'recalled', 'exact_match', or 'em'")
        return
    
    # Compute paired differences and bootstrap CIs for available metrics
    for metric in available_metrics:
        differences = compute_paired_differences(pairs, metric)
        mean_diff, lower_bound, upper_bound = bootstrap_resample(
            differences, 
            n_bootstrap=args.n_bootstrap,
            confidence_level=args.confidence_level
        )
        print_results(metric, mean_diff, lower_bound, upper_bound, 
                     len(pairs), args.confidence_level)


if __name__ == "__main__":
    main()