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

from smartstress_langgraph.api import start_monitoring_session
from smartstress_langgraph.rag.tidb_vector_store import get_tidb_vector_store

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


def get_rag_context(query: str, k: int = 3, tags: List[str] = None) -> tuple[str, List[Dict]]:
    """
    Retrieve relevant documents from RAG for a query.
    
    Returns:
        (context_text, retrieved_docs)
    """
    if k == 0:
        return "", []
    
    vs = get_tidb_vector_store()
    results = vs.similarity_search(query, k=k)
    vs.close()
    
    # Format context for injection
    context_parts = []
    retrieved_docs = []
    
    for i, (doc, score) in enumerate(results, 1):
        context_parts.append(f"[Reference {i}]:\n{doc.content}\n")
        retrieved_docs.append({
            "rank": i,
            "similarity": float(score),
            "source": doc.source,
            "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
        })
    
    context_text = "\n".join(context_parts)
    return context_text, retrieved_docs


def run_single_test(query_data: Dict[str, Any], group_config, session_id: str = None) -> Dict[str, Any]:
    """
    Run a single test query through the specified group.
    
    Returns:
        Test result dictionary
    """
    query = query_data["query"]
    query_id = query_data["id"]
    
    print(f"  Processing query {query_id} with {group_config.group_name} group...")
    
    # Get RAG context if enabled
    context_text = ""
    retrieved_docs = []
    
    if group_config.use_rag:
        context_text, retrieved_docs = get_rag_context(
            query,
            k=group_config.rag_k,
            tags=group_config.rag_tags
        )
    
    # Create augmented query with context for experimental group
    if context_text:
        augmented_query = f"""Here is some professional guidance that may be relevant:

{context_text}

User query: {query}

Please provide a helpful, empathetic response drawing on the professional guidance above where appropriate."""
    else:
        augmented_query = query
    
    # Start agent session
    if not session_id:
        session_id = f"ab_test_{group_config.group_name.lower()}_{query_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Import required models
    from smartstress_langgraph.io_models import StartSessionRequest, UserInfo, ContinueSessionRequest, SessionHandleModel, ChatMessage
    
    try:
        # Start session first (without initial message)
        request = StartSessionRequest(
            user=UserInfo(
                user_id=f"ab_test_user",
                session_id=session_id
            )
        )
        
        handle, _ = start_monitoring_session(request)
        
        # Then continue with the user message
        continue_request = ContinueSessionRequest(
            session_handle=handle,
            user_message=ChatMessage(role="user", content=augmented_query)
        )
        
        from smartstress_langgraph.api import continue_session
        _, state_view = continue_session(continue_request)
        
        # Extract response
        agent_response = ""
        if state_view and hasattr(state_view, 'conversation_history'):
            for msg in reversed(state_view.conversation_history):
                if msg.get("role") == "assistant":
                    agent_response = msg.get("content", "")
                    break
                    
    except Exception as e:
        print(f"    Warning: Agent error - {e}")
        agent_response = f"[Agent Error: {str(e)}]"
    
    return {
        "query_id": query_id,
        "query": query,
        "category": query_data.get("category", "unknown"),
        "group": group_config.group_name,
        "use_rag": group_config.use_rag,
        "rag_k": group_config.rag_k,
        "retrieved_docs": retrieved_docs,
        "context_used": context_text if group_config.use_rag else None,
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
