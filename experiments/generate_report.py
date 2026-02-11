"""
Report Generator for A/B Test Results

Generates a comprehensive comparison report with statistics and visualizations.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from statistics import mean, stdev


def load_evaluated_results(results_file: str) -> List[Dict[str, Any]]:
    """Load evaluated results from JSON file."""
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_group_statistics(results: List[Dict[str, Any]], group_name: str) -> Dict[str, Any]:
    """Calculate statistics for a test group."""
    group_results = [r for r in results if r.get("group") == group_name]
    
    if not group_results:
        return {"error": "No results for this group"}
    
    # Extract scores
    scores = {metric: [] for metric in ["groundedness", "stressor_identification", "safety_compliance", "response_quality"]}
    
    for result in group_results:
        eval_data = result.get("evaluation")
        if eval_data and isinstance(eval_data, dict):
            for metric in scores.keys():
                score = eval_data.get(metric)
                if isinstance(score, (int, float)) and score > 0:
                    scores[metric].append(score)
    
    # Calculate statistics
    stats = {
        "n": len(group_results),
        "metrics": {}
    }
    
    for metric, values in scores.items():
        if values:
            stats["metrics"][metric] = {
                "mean": round(mean(values), 2),
                "stdev": round(stdev(values), 2) if len(values) > 1 else 0,
                "min": min(values),
                "max": max(values),
                "n_valid": len(values)
            }
        else:
            stats["metrics"][metric] = {
                "mean": 0,
                "stdev": 0,
                "min": 0,
                "max": 0,
                "n_valid": 0
            }
    
    return stats


def generate_markdown_report(results_file: str, output_file: str = None) -> str:
    """
    Generate a comprehensive Markdown report.
    
    Returns:
        Path to generated report
    """
    print("=" * 70)
    print("Generating Comparison Report")
    print("=" * 70)
    
    # Load results
    print(f"\nLoading evaluated results from {results_file}...")
    results = load_evaluated_results(results_file)
    print(f"Loaded {len(results)} evaluated results")
    
    # Calculate statistics for both groups
    control_stats = calculate_group_statistics(results, "Control")
    experimental_stats = calculate_group_statistics(results, "Experimental")
    
    # Generate report
    report_lines = []
    report_lines.append("# CounselChat RAG Enhancement - A/B Test Report\n")
    report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append("---\n")
    
    # Executive Summary
    report_lines.append("## Executive Summary\n")
    report_lines.append("This report presents the results of an A/B test comparing the SmartStress Agent's performance ")
    report_lines.append("with and without CounselChat RAG enhancement.\n")
    
    control_n = control_stats.get("n", 0)
    experimental_n = experimental_stats.get("n", 0)
    
    report_lines.append(f"- **Control Group (No RAG):** {control_n} queries tested\n")
    report_lines.append(f"- **Experimental Group (CounselChat RAG, k=3):** {experimental_n} queries tested\n")
    report_lines.append(f"- **Total Test Queries:** {control_n} unique scenarios\n")
    report_lines.append("\n---\n")
    
    # Metric Comparison Table
    report_lines.append("## Performance Comparison\n")
    report_lines.append("### Overall Metrics (Mean Scores, 1-5 scale)\n")
    report_lines.append("| Metric | Control (No RAG) | Experimental (With RAG) | Improvement |\n")
    report_lines.append("|--------|------------------|-------------------------|-------------|\n")
    
    metrics_display = {
        "groundedness": "Groundedness",
        "stressor_identification": "Stressor Identification",
        "safety_compliance": "Safety Compliance",
        "response_quality": "Response Quality"
    }
    
    total_improvement = 0
    metric_count = 0
    
    for metric_key, metric_name in metrics_display.items():
        control_mean = control_stats.get("metrics", {}).get(metric_key, {}).get("mean", 0)
        exp_mean = experimental_stats.get("metrics", {}).get(metric_key, {}).get("mean", 0)
        
        if control_mean > 0:
            improvement = ((exp_mean - control_mean) / control_mean) * 100
            total_improvement += improvement
            metric_count += 1
        else:
            improvement = 0
        
        improvement_str = f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%"
        if improvement > 5:
            improvement_str = f"✓ {improvement_str}"
        
        report_lines.append(f"| {metric_name} | {control_mean:.2f} | {exp_mean:.2f} | {improvement_str} |\n")
    
    avg_improvement = total_improvement / metric_count if metric_count > 0 else 0
    report_lines.append(f"\n**Average Improvement:** {avg_improvement:+.1f}%\n")
    report_lines.append("\n---\n")
    
    # Detailed Statistics
    report_lines.append("## Detailed Statistics\n")
    
    for group_name, stats in [("Control", control_stats), ("Experimental", experimental_stats)]:
        report_lines.append(f"\n### {group_name} Group\n")
        report_lines.append(f"**Sample Size:** {stats.get('n', 0)}\n\n")
        report_lines.append("| Metric | Mean | Std Dev | Min | Max |\n")
        report_lines.append("|--------|------|---------|-----|-----|\n")
        
        for metric_key, metric_name in metrics_display.items():
            m_stats = stats.get("metrics", {}).get(metric_key, {})
            report_lines.append(
                f"| {metric_name} | {m_stats.get('mean', 0):.2f} | "
                f"{m_stats.get('stdev', 0):.2f} | {m_stats.get('min', 0)} | {m_stats.get('max', 0)} |\n"
            )
    
    report_lines.append("\n---\n")
    
    # Key Findings
    report_lines.append("## Key Findings\n")
    
    if avg_improvement > 5:
        report_lines.append(f"✅ **Positive Impact:** The CounselChat RAG enhancement shows an average improvement of {avg_improvement:.1f}% across all metrics.\n\n")
    elif avg_improvement < -5:
        report_lines.append(f"⚠️ **Negative Impact:** The CounselChat RAG enhancement shows an average decline of {abs(avg_improvement):.1f}% across all metrics.\n\n")
    else:
        report_lines.append(f"ℹ️ **Neutral Impact:** The CounselChat RAG enhancement shows minimal change ({avg_improvement:.1f}%).\n\n")
    
    # Metric-specific findings
    for metric_key, metric_name in metrics_display.items():
        control_mean = control_stats.get("metrics", {}).get(metric_key, {}).get("mean", 0)
        exp_mean = experimental_stats.get("metrics", {}).get(metric_key, {}).get("mean", 0)
        
        if control_mean > 0:
            improvement = ((exp_mean - control_mean) / control_mean) * 100
            
            if improvement > 10:
                report_lines.append(f"- **{metric_name}:** Strong improvement (+{improvement:.1f}%)\n")
            elif improvement > 5:
                report_lines.append(f"- **{metric_name}:** Moderate improvement (+{improvement:.1f}%)\n")
            elif improvement < -5:
                report_lines.append(f"- **{metric_name}:** Decline ({improvement:.1f}%)\n")
    
    report_lines.append("\n---\n")
    
    # Recommendations
    report_lines.append("## Recommendations\n")
    
    if avg_improvement > 5:
        report_lines.append("1. **Deploy RAG Enhancement:** The positive results support deploying the CounselChat RAG system to production.\n")
        report_lines.append("2. **Monitor Performance:** Continue tracking these metrics in production to ensure sustained improvement.\n")
        report_lines.append("3. **Expand Knowledge Base:** Consider adding more high-quality counseling resources to further enhance performance.\n")
    else:
        report_lines.append("1. **Review RAG Implementation:** Investigate why the RAG enhancement did not show expected improvements.\n")
        report_lines.append("2. **Refine Retrieval:** Consider adjusting retrieval parameters (k value, similarity threshold, etc.).\n")
        report_lines.append("3. **Improve Context Integration:** Review how retrieved context is integrated into agent responses.\n")
    
    report_lines.append("\n---\n")
    
    # Methodology
    report_lines.append("## Methodology\n")
    report_lines.append("- **Evaluation Method:** LLM-as-a-judge using Gemini\n")
    report_lines.append("- **Scoring Scale:** 1-5 for each metric\n")
    report_lines.append("- **Test Queries:** 25 diverse stress-related scenarios\n")
    report_lines.append("- **RAG Configuration:** k=3 documents retrieved per query\n")
    report_lines.append("- **Knowledge Source:** CounselChat dataset (professional mental health Q&A)\n")
    
    # Save report
    if not output_file:
        # Save to report subdirectory
        report_dir = Path("experiments/report")
        report_dir.mkdir(parents=True, exist_ok=True)
        base_name = Path(results_file).stem
        output_file = str(report_dir / f"{base_name}_report.md")
    
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(''.join(report_lines))
    
    print(f"\n{'='*70}")
    print(f"Report Generated!")
    print(f"Report saved to: {output_path}")
    print(f"\nAverage Improvement: {avg_improvement:+.1f}%")
    print(f"{'='*70}")
    
    return str(output_path)


def main():
    """Main entry point."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python generate_report.py <evaluated_results_file.json>")
        print("\nExample:")
        print("  python experiments/generate_report.py experiments/ab_test_results_20260210_120000_evaluated.json")
        sys.exit(1)
    
    results_file = sys.argv[1]
    report_file = generate_markdown_report(results_file)
    print(f"\n📊 View the report at: {report_file}")


if __name__ == "__main__":
    main()
