"""
A/B Test Runner for CounselChat RAG Experiment

Runs test queries through both control and experimental groups.
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
from datetime import datetime
from typing import Dict, List, Any

from smartstress_langgraph.api import start_monitoring_session, continue_session

# Import config from experiments directory
import importlib.util
spec = importlib.util.spec_from_file_location("ab_test_config", Path(__file__).parent / "ab_test_config.py")
ab_test_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab_test_config)

CONTROL_GROUP = ab_test_config.CONTROL_GROUP
EXPERIMENTAL_GROUP = ab_test_config.EXPERIMENTAL_GROUP
EVALUATION_METRICS = ab_test_config.EVALUATION_METRICS


def load_test_queries(queries_file: str) -> List[Dict[str, Any]]:
    """Load test queries from JSON file."""
    with open(queries_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_single_test(query_data: Dict[str, Any], group_config, session_id: str = None) -> Dict[str, Any]:
    """
    Run a single test query through the specified group.
    
    The agent handles RAG internally based on the use_rag state field.
    
    Returns:
        Test result dictionary
    """
    query = query_data["query"]
    query_id = query_data["id"]
    
    print(f"  Processing query {query_id} with {group_config.group_name} group (RAG={group_config.use_rag})...")
    
    # Start agent session
    if not session_id:
        session_id = f"ab_test_{group_config.group_name.lower()}_{query_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Import required models
    from smartstress_langgraph.io_models import StartSessionRequest, UserInfo, ContinueSessionRequest, SessionHandleModel, ChatMessage
    
    try:
        # Start session
        request = StartSessionRequest(
            user=UserInfo(
                user_id=f"ab_test_user",
                session_id=session_id
            )
        )
        
        handle, _ = start_monitoring_session(request)
        
        # Override the use_rag flag in state based on group config
        from smartstress_langgraph.api import APP, _load_cached_state
        from smartstress_langgraph.state import SessionHandle
        
        sh = handle.to_handle()
        cached_state = _load_cached_state(sh)
        cached_state["use_rag"] = group_config.use_rag
        
        # Save the updated state by re-invoking with the use_rag flag
        # We do this by continuing the session with the user message
        continue_request = ContinueSessionRequest(
            session_handle=handle,
            user_message=ChatMessage(role="user", content=query)
        )
        
        # Temporarily patch the state to include use_rag
        from smartstress_langgraph import api as api_module
        original_load = api_module._load_cached_state
        
        def patched_load(h):
            s = original_load(h)
            s["use_rag"] = group_config.use_rag
            return s
        
        api_module._load_cached_state = patched_load
        try:
            _, state_view = continue_session(continue_request)
        finally:
            api_module._load_cached_state = original_load
        
        # Extract response
        agent_response = ""
        if state_view and hasattr(state_view, 'conversation_history'):
            for msg in reversed(state_view.conversation_history):
                if msg.get("role") == "assistant":
                    agent_response = msg.get("content", "")
                    break
                    
    except Exception as e:
        print(f"    Warning: Agent error - {e}")
        import traceback
        traceback.print_exc()
        agent_response = f"[Agent Error: {str(e)}]"
    
    return {
        "query_id": query_id,
        "query": query,
        "category": query_data.get("category", "unknown"),
        "group": group_config.group_name,
        "use_rag": group_config.use_rag,
        "rag_k": group_config.rag_k,
        "response": agent_response,
        "session_id": session_id,
        "timestamp": datetime.now().isoformat()
    }


def run_ab_test(queries_file: str, output_file: str = None):
    """
    Run complete A/B test on all queries.
    
    Args:
        queries_file: Path to test queries JSON file
        output_file: Path to save raw results (optional)
    """
    print("=" * 70)
    print("CounselChat RAG A/B Test Runner")
    print("=" * 70)
    
    # Load queries
    print(f"\nLoading test queries from {queries_file}...")
    queries = load_test_queries(queries_file)
    print(f"Loaded {len(queries)} test queries")
    
    # Run tests for both groups
    all_results = []
    
    for group_config in [CONTROL_GROUP, EXPERIMENTAL_GROUP]:
        print(f"\n{'='*70}")
        print(f"Running {group_config.group_name} Group: {group_config.description}")
        print(f"{'='*70}")
        
        for i, query_data in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}] Query: {query_data['id']}")
            
            try:
                result = run_single_test(query_data, group_config)
                all_results.append(result)
                print(f"  ✓ Completed")
            except Exception as e:
                print(f"  ✗ Error: {e}")
                all_results.append({
                    "query_id": query_data["id"],
                    "query": query_data["query"],
                    "group": group_config.group_name,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
    
    # Save raw results to report directory
    if not output_file:
        report_dir = Path("experiments/report")
        report_dir.mkdir(parents=True, exist_ok=True)
        output_file = str(report_dir / f"ab_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print(f"A/B Test Complete!")
    print(f"Results saved to: {output_path}")
    print(f"Total tests run: {len(all_results)} ({len(all_results)//2} queries × 2 groups)")
    print(f"{'='*70}")
    
    return str(output_path)


def main():
    """Main entry point."""
    queries_file = "experiments/test_queries.json"
    output_file = run_ab_test(queries_file)
    print(f"\nNext step: Run evaluation with: python experiments/evaluate_results.py {output_file}")


if __name__ == "__main__":
    main()
