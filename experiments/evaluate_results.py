"""
Automated Evaluation Script using LLM-as-a-Judge

Evaluates A/B test results using Gemini as a judge.
"""

from __future__ import annotations

import sys
from pathlib import Path
import os

# Add project root to Python path - use absolute path  
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
import re
from datetime import datetime
from typing import Dict, List, Any

from smartstress_langgraph.llm import generate_chat

# Import config
import importlib.util
spec = importlib.util.spec_from_file_location("ab_test_config", Path(__file__).parent / "ab_test_config.py")
ab_test_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab_test_config)

JUDGE_PROMPT_TEMPLATE = ab_test_config.JUDGE_PROMPT_TEMPLATE
EVALUATION_METRICS = ab_test_config.EVALUATION_METRICS


def load_test_results(results_file: str) -> List[Dict[str, Any]]:
    """Load A/B test results from JSON file."""
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate_response(query: str, response: str, context: str = "", query_id: str = "") -> Dict[str, Any]:
    """
    Evaluate a single response using LLM-as-a-judge.
    
    Returns:
        Dictionary with scores and justification
    """
    # Format the judge prompt
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        query=query,
        response=response,
        context=context if context else "No context provided (control group)"
    )
    
    print(f"    Evaluating with LLM judge...", end="")
    
    try:
        # Call Gemini as judge
        judge_response = generate_chat(
            messages=[{"role": "user", "content": prompt}],
            generation_config={"temperature": 0.1}  # Lower temperature for consistency
        )
        
        # Extract JSON from response
        json_match = re.search(r'\{[^{}]*\}', judge_response, re.DOTALL)
        if json_match:
            evaluation = json.loads(json_match.group())
            print(" ✓")
            return evaluation
        else:
            print(" ✗ (JSON parse error)")
            return {
                "groundedness": 0,
                "stressor_identification": 0,
                "safety_compliance": 0,
                "response_quality": 0,
                "justification": "Failed to parse judge response",
                "raw_response": judge_response
            }
    
    except Exception as e:
        print(f" ✗ (Error: {e})")
        return {
            "groundedness": 0,
            "stressor_identification": 0,
            "safety_compliance": 0,
            "response_quality": 0,
            "justification": f"Evaluation error: {str(e)}",
            "error": str(e)
        }


def evaluate_all_results(results_file: str, output_file: str = None):
    """
    Evaluate all test results using LLM-as-a-judge.
    
    Args:
        results_file: Path to A/B test results JSON
        output_file: Path to save evaluation results (optional)
    """
    import time
    
    print("=" * 70)
    print("CounselChat RAG Experiment - Automated Evaluation")
    print("=" * 70)
    
    # Load results
    print(f"\nLoading test results from {results_file}...")
    results = load_test_results(results_file)
    print(f"Loaded {len(results)} test results")
    
    # Evaluate each result
    evaluated_results = []
    
    for i, result in enumerate(results, 1):
        query_id = result.get("query_id", f"unknown_{i}")
        group = result.get("group", "unknown")
        
        print(f"\n[{i}/{len(results)}] {query_id} ({group} group):")
        
        # Skip if error occurred during A/B test
        if "error" in result and "response" not in result:
            print(f"    ⚠ Skipping (test error): {result.get('error')}")
            evaluated_results.append({
                **result,
                "evaluation": None,
                "evaluation_error": result.get("error")
            })
            continue
        
        # Evaluate the response
        evaluation = evaluate_response(
            query=result.get("query", ""),
            response=result.get("response", ""),
            context=result.get("context_used", ""),
            query_id=query_id
        )
        
        evaluated_results.append({
            **result,
            "evaluation": evaluation
        })
        
        # Add 15-second delay to avoid API quota issues (except for last one)
        if i < len(results):
            print(f"    Waiting 5s before next evaluation...")
            time.sleep(5)
    
    # Save evaluated results to report directory
    if not output_file:
        # Save to report subdirectory
        report_dir = Path("experiments/report")
        report_dir.mkdir(parents=True, exist_ok=True)
        base_name = Path(results_file).stem
        output_file = str(report_dir / f"{base_name}_evaluated.json")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(evaluated_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print(f"Evaluation Complete!")
    print(f"Evaluated results saved to: {output_path}")
    print(f"{'='*70}")
    
    return str(output_path)


def main():
    """Main entry point."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate_results.py <results_file.json>")
        print("\nExample:")
        print("  python experiments/evaluate_results.py experiments/ab_test_results_20260210_120000.json")
        sys.exit(1)
    
    results_file = sys.argv[1]
    output_file = evaluate_all_results(results_file)
    print(f"\nNext step: Generate report with: python experiments/generate_report.py {output_file}")


if __name__ == "__main__":
    main()
