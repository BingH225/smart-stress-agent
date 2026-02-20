"""
Report Generator for A/B Test Results (Similarity-Only Evaluation)

Compares Control vs Experimental group similarity scores and
generates a Markdown report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime
from statistics import mean, stdev
from typing import Dict, List, Any, Optional, Tuple


def load_evaluated_results(results_file: str) -> List[Dict[str, Any]]:
    """Load evaluated results from JSON file."""
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_group_statistics(results: List[Dict[str, Any]], group_name: str) -> Dict[str, Any]:
    """Calculate similarity statistics for a test group."""
    group_results = [r for r in results if r.get("group") == group_name]

    if not group_results:
        return {"error": "No results for this group"}

    similarities = []
    for result in group_results:
        ev = result.get("evaluation") or {}
        sim = ev.get("ground_truth_similarity")
        if isinstance(sim, float) and sim > 0:
            similarities.append(sim)

    n = len(group_results)
    return {
        "n": n,
        "n_valid": len(similarities),
        "similarity": {
            "mean":  round(mean(similarities), 4)  if similarities else 0.0,
            "stdev": round(stdev(similarities), 4) if len(similarities) > 1 else 0.0,
            "min":   round(min(similarities), 4)   if similarities else 0.0,
            "max":   round(max(similarities), 4)   if similarities else 0.0,
        },
        # Per-category breakdown
        "by_category": _category_means(group_results),
        # Raw scores list (for t-test)
        "raw_scores": similarities,
    }


def _category_means(group_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Return mean similarity per category."""
    cat_scores: Dict[str, List[float]] = {}
    for r in group_results:
        cat = r.get("category", "unknown")
        ev  = r.get("evaluation") or {}
        sim = ev.get("ground_truth_similarity")
        if isinstance(sim, float):
            cat_scores.setdefault(cat, []).append(sim)
    return {cat: round(mean(vals), 4) for cat, vals in sorted(cat_scores.items())}


def run_ttest(
    scores_a: List[float],
    scores_b: List[float],
) -> Tuple[Optional[float], Optional[float], str]:
    """
    Run Welch's independent-samples t-test (unequal variance assumed).

    Returns:
        (t_stat, p_value, interpretation_string)
    """
    try:
        from scipy import stats

        if len(scores_a) < 2 or len(scores_b) < 2:
            return None, None, "Insufficient data for t-test"

        t_stat, p_value = stats.ttest_ind(scores_a, scores_b, equal_var=False)

        if p_value < 0.001:
            sig = "highly significant (p < 0.001)"
        elif p_value < 0.01:
            sig = "significant (p < 0.01)"
        elif p_value < 0.05:
            sig = "significant (p < 0.05)"
        elif p_value < 0.10:
            sig = "marginally significant (p < 0.10)"
        else:
            sig = "not significant (p ≥ 0.10)"

        return float(t_stat), float(p_value), sig
    except ImportError:
        return None, None, "scipy not installed — t-test skipped"
    except Exception as e:
        return None, None, f"t-test error: {e}"


def generate_markdown_report(results_file: str, output_file: str = None) -> str:
    """Generate a Markdown similarity comparison report with t-test."""
    print("=" * 70)
    print("Generating Similarity Comparison Report")
    print("=" * 70)

    print(f"\nLoading: {results_file}")
    results = load_evaluated_results(results_file)
    print(f"Loaded {len(results)} evaluated results")

    ctrl = calculate_group_statistics(results, "Control")
    exp  = calculate_group_statistics(results, "Experimental")

    ctrl_mean = ctrl["similarity"]["mean"]
    exp_mean  = exp["similarity"]["mean"]
    improvement = ((exp_mean - ctrl_mean) / ctrl_mean * 100) if ctrl_mean > 0 else 0.0

    # T-test
    t_stat, p_value, significance = run_ttest(
        ctrl["raw_scores"], exp["raw_scores"]
    )

    lines = []

    # ── Header ──────────────────────────────────────────────────────────────
    lines += [
        "# CounselChat RAG Enhancement — A/B Test Report\n\n",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
        "---\n\n",
    ]

    # ── Executive Summary ───────────────────────────────────────────────────
    lines += ["## Executive Summary\n\n"]
    lines += [
        "Comparison of the SmartStress Agent **with** and **without** "
        "CounselChat RAG enhancement, measured by TF-IDF cosine similarity "
        "to ground-truth expert answers.\n\n",
        f"| | Queries | Valid Scores | Mean Similarity |\n",
        f"|---|---|---|---|\n",
        f"| **Control (No RAG)** | {ctrl['n']} | {ctrl['n_valid']} | {ctrl_mean:.4f} |\n",
        f"| **Experimental (RAG k=3)** | {exp['n']} | {exp['n_valid']} | {exp_mean:.4f} |\n",
        "\n",
    ]

    # T-test summary table in executive summary
    if t_stat is not None:
        lines += [
            "### Statistical Significance (Welch's t-test)\n\n",
            f"| t-statistic | p-value | Result |\n",
            f"|---|---|---|\n",
            f"| {t_stat:.4f} | {p_value:.4f} | {significance} |\n",
            "\n",
        ]
    lines += ["\n---\n\n"]

    # ── Overall Comparison ──────────────────────────────────────────────────
    lines += ["## Overall Similarity Scores\n\n"]
    lines += [
        "| Metric | Control | Experimental | Δ |\n",
        "|--------|---------|--------------|---|\n",
    ]
    for stat in ["mean", "stdev", "min", "max"]:
        c_val = ctrl["similarity"][stat]
        e_val = exp["similarity"][stat]
        delta = e_val - c_val
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        lines.append(f"| {stat.capitalize()} | {c_val:.4f} | {e_val:.4f} | {delta_str} |\n")
    lines += ["\n"]

    pct_str = f"+{improvement:.1f}%" if improvement >= 0 else f"{improvement:.1f}%"
    lines += [f"**Mean similarity change (RAG vs No-RAG):** {pct_str}\n\n", "---\n\n"]

    # ── Per-Category Breakdown ──────────────────────────────────────────────
    all_cats = sorted(set(list(ctrl["by_category"]) + list(exp["by_category"])))
    if all_cats:
        lines += ["## Per-Category Similarity\n\n"]
        lines += ["| Category | Control | Experimental | Δ |\n", "|---|---|---|---|\n"]
        for cat in all_cats:
            c = ctrl["by_category"].get(cat, 0.0)
            e = exp["by_category"].get(cat, 0.0)
            d = e - c
            d_str = f"+{d:.4f}" if d >= 0 else f"{d:.4f}"
            lines.append(f"| {cat} | {c:.4f} | {e:.4f} | {d_str} |\n")
        lines += ["\n---\n\n"]

    # ── Key Findings ────────────────────────────────────────────────────────
    lines += ["## Key Findings\n\n"]
    if improvement > 5:
        lines += [f"✅ **Positive Impact:** RAG improves response similarity by {improvement:.1f}%.\n\n"]
    elif improvement < -5:
        lines += [f"⚠️ **Negative Impact:** RAG decreases similarity by {abs(improvement):.1f}%.\n\n"]
    else:
        lines += [f"ℹ️ **Neutral Impact:** Minimal change ({improvement:+.1f}%).\n\n"]

    # T-test finding
    if t_stat is not None:
        sig_symbol = "✅" if p_value < 0.05 else "❌"
        lines += [
            f"{sig_symbol} **Statistical Test:** Welch's t-test — "
            f"t = {t_stat:.4f}, p = {p_value:.4f} — **{significance}**\n\n",
        ]

    # Best/worst categories for RAG
    cat_diffs = {
        cat: (exp["by_category"].get(cat, 0.0) - ctrl["by_category"].get(cat, 0.0))
        for cat in all_cats
    }
    if cat_diffs:
        best_cat  = max(cat_diffs, key=cat_diffs.get)
        worst_cat = min(cat_diffs, key=cat_diffs.get)
        lines += [
            f"- **Best category for RAG:** `{best_cat}` ({cat_diffs[best_cat]:+.4f})\n",
            f"- **Worst category for RAG:** `{worst_cat}` ({cat_diffs[worst_cat]:+.4f})\n",
            "\n",
        ]

    lines += ["---\n\n"]

    # ── Recommendations ─────────────────────────────────────────────────────
    lines += ["## Recommendations\n\n"]
    if improvement > 5:
        lines += [
            "1. **Deploy RAG:** Results support production deployment.\n",
            "2. **Monitor:** Continue tracking similarity scores in production.\n",
            "3. **Expand Knowledge Base:** Add more high-quality counseling resources.\n",
        ]
    else:
        lines += [
            "1. **Review RAG Implementation:** Investigate why improvement is limited.\n",
            "2. **Tune Retrieval:** Adjust `k` value and similarity thresholds.\n",
            "3. **Improve Context Integration:** Review prompt template for RAG context.\n",
        ]

    lines += ["\n---\n\n"]

    # ── Methodology ─────────────────────────────────────────────────────────
    lines += [
        "## Methodology\n\n",
        "- **Evaluation Metric:** TF-IDF cosine similarity (unigrams + bigrams) to expert ground truth\n",
        "- **Similarity Scale:** 0.0 – 1.0 (higher = more lexically similar to expert answer)\n",
        "- **Statistical Test:** Welch's independent-samples t-test (unequal variance assumed, two-tailed)\n",
        f"- **Control Queries:** {ctrl['n']} | **Experimental Queries:** {exp['n']}\n",
        "- **RAG Configuration:** k=3 documents retrieved per query, CounselChat dataset\n",
        "- **Test Data Source:** `counselchat-data.csv` (held-out from RAG ingestion)\n",
    ]

    # ── Save ─────────────────────────────────────────────────────────────────
    if not output_file:
        report_dir = Path("experiments/report")
        report_dir.mkdir(parents=True, exist_ok=True)
        base_name  = Path(results_file).stem
        output_file = str(report_dir / f"{base_name}_report.md")

    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines), encoding="utf-8")

    print(f"\n{'='*70}")
    print(f"Report saved to: {out}")
    print(f"Mean similarity — Control: {ctrl_mean:.4f} | Experimental: {exp_mean:.4f} | Δ: {pct_str}")
    print(f"{'='*70}")

    return str(out)


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_report.py <evaluated_results.json>")
        sys.exit(1)
    report = generate_markdown_report(sys.argv[1])
    print(f"\n📊 Report: {report}")


if __name__ == "__main__":
    main()
