"""
Evaluation Script - Similarity Only

Evaluates A/B test results by computing TF-IDF cosine similarity
between the agent response and the ground truth expert answer.
No LLM judge is used — fast, deterministic, no API quota consumed.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
from typing import Dict, List, Any


def load_test_results(results_file: str) -> List[Dict[str, Any]]:
    """Load A/B test results from JSON file."""
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Compute TF-IDF cosine similarity between two texts.

    Uses unigrams + bigrams with English stop-word removal.
    Returns a score in [0.0, 1.0].
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    if not text1.strip() or not text2.strip():
        return 0.0

    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
        tfidf = vectorizer.fit_transform([text1, text2])
        v1 = tfidf[0].toarray().flatten()
        v2 = tfidf[1].toarray().flatten()
        norm1, norm2 = float(v1 @ v1) ** 0.5, float(v2 @ v2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float((v1 @ v2) / (norm1 * norm2))
    except Exception as e:
        print(f"  [Similarity error: {e}]", end="")
        return 0.0


def evaluate_all_results(results_file: str, output_file: str = None) -> str:
    """
    Evaluate all test results using TF-IDF cosine similarity only.

    Args:
        results_file: Path to A/B test results JSON.
        output_file:  Output path (optional; defaults to <results_stem>_evaluated.json).

    Returns:
        Path to the saved evaluated results file.
    """
    print("=" * 70)
    print("CounselChat RAG Experiment - Similarity Evaluation")
    print("=" * 70)

    print(f"\nLoading results from: {results_file}")
    results = load_test_results(results_file)
    print(f"Loaded {len(results)} results\n")

    evaluated_results = []
    scores_computed = 0

    for i, result in enumerate(results, 1):
        query_id = result.get("query_id", f"unknown_{i}")
        group    = result.get("group", "unknown")
        response = result.get("response", "")
        gt       = result.get("ground_truth", "")

        print(f"[{i}/{len(results)}] {query_id} ({group})", end="  ")

        # Skip error items
        if "error" in result and not response:
            print("⚠ skipped (test error)")
            evaluated_results.append({**result, "evaluation": None})
            continue

        # Compute similarity
        if gt and response:
            sim = calculate_similarity(response, gt)
            scores_computed += 1
            print(f"similarity={sim:.4f}")
        else:
            sim = None
            print("similarity=N/A (no ground truth)")

        evaluated_results.append({
            **result,
            "evaluation": {
                "ground_truth_similarity": sim
            }
        })

    # Determine output path
    if not output_file:
        report_dir = Path("experiments/report")
        report_dir.mkdir(parents=True, exist_ok=True)
        base_name  = Path(results_file).stem
        output_file = str(report_dir / f"{base_name}_evaluated.json")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(evaluated_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"Done! {scores_computed}/{len(results)} similarity scores computed.")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}")

    return str(output_path)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python evaluate_results.py <results_file.json>")
        print("\nExample:")
        print("  python experiments/evaluate_results.py experiments/report/ab_test_results_20260213_172501.json")
        sys.exit(1)

    results_file = sys.argv[1]
    output_file  = evaluate_all_results(results_file)
    print(f"\nNext: python experiments/generate_report.py {output_file}")


if __name__ == "__main__":
    main()
