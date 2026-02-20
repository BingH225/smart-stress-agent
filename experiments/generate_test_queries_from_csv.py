"""
Generate Test Queries from CounselChat CSV

Extracts a random sample of conversations from the test CSV file
and creates test_queries.json for A/B testing with ground truth answers.
"""

from __future__ import annotations

import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd


def clean_html(text: str) -> str:
    """Remove HTML tags and clean up text formatting."""
    if pd.isna(text):
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', str(text))
    
    # Convert HTML entities
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&quot;', '"')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def generate_test_queries(
    csv_path: str,
    num_queries: int = 50,
    random_seed: int = 42,
    stratify_by_topic: bool = True
) -> List[Dict[str, Any]]:
    """
    Generate test queries from CSV file.
    
    Args:
        csv_path: Path to CSV file
        num_queries: Number of test queries to generate
        random_seed: Random seed for reproducibility
        stratify_by_topic: Whether to stratify sampling by topic
        
    Returns:
        List of test query dictionaries
    """
    print(f"\nLoading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Total rows in CSV: {len(df)}")
    
    # Extract and clean fields
    df['question_text'] = df.get('questionText', '').apply(clean_html)
    df['answer_text'] = df.get('answerText', '').apply(clean_html)
    df['topic'] = df.get('topic', df.get('topics', 'General'))
    df['question_id'] = df.get('questionID', df.index)
    
    # Filter out rows with empty or too short content
    df = df[
        (df['question_text'].str.len() >= 20) &
        (df['answer_text'].str.len() >= 30)
    ]
    
    print(f"Valid rows after filtering: {len(df)}")
    
    if len(df) < num_queries:
        print(f"Warning: Only {len(df)} valid rows available, requested {num_queries}")
        num_queries = len(df)
    
    # Sample rows
    if stratify_by_topic and 'topic' in df.columns:
        # Stratified sampling by topic
        print(f"\nStratified sampling by topic...")
        
        # Count samples per topic
        topic_counts = df['topic'].value_counts()
        print(f"Found {len(topic_counts)} unique topics")
        
        # Calculate samples per topic (proportional)
        samples_per_topic = {}
        for topic, count in topic_counts.items():
            samples = max(1, int(num_queries * count / len(df)))
            samples_per_topic[topic] = min(samples, count)
        
        # Adjust to ensure we get exactly num_queries
        total_samples = sum(samples_per_topic.values())
        if total_samples < num_queries:
            # Add more samples to largest topics
            sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
            for topic, _ in sorted_topics:
                if total_samples >= num_queries:
                    break
                if samples_per_topic[topic] < topic_counts[topic]:
                    samples_per_topic[topic] += 1
                    total_samples += 1
        
        # Sample from each topic
        sampled_rows = []
        for topic, n_samples in samples_per_topic.items():
            topic_df = df[df['topic'] == topic]
            sampled = topic_df.sample(n=n_samples, random_state=random_seed)
            sampled_rows.append(sampled)
        
        sample_df = pd.concat(sampled_rows, ignore_index=True)
        
        # Print topic distribution
        print("\nTopic distribution in sample:")
        for topic, count in sample_df['topic'].value_counts().items():
            print(f"  {topic}: {count}")
    else:
        # Simple random sampling
        print(f"\nRandom sampling {num_queries} rows...")
        sample_df = df.sample(n=num_queries, random_state=random_seed)
    
    # Generate test queries
    test_queries = []
    for i, row in sample_df.iterrows():
        query = {
            "id": f"test_{row['question_id']}",
            "query": row['question_text'],
            "category": str(row['topic']) if not pd.isna(row['topic']) else "General",
            "ground_truth": row['answer_text']
        }
        test_queries.append(query)
    
    print(f"\nGenerated {len(test_queries)} test queries")
    return test_queries


def main():
    """Main test query generation process."""
    parser = argparse.ArgumentParser(description="Generate test queries from CounselChat CSV")
    parser.add_argument(
        "--csv",
        type=str,
        default="rag_docs/counselchat-data.csv",
        help="Path to CSV file for test data (default: rag_docs/counselchat-data.csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/test_queries.json",
        help="Output path for test_queries.json (default: experiments/test_queries.json)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Number of test queries to generate (default: 50)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Disable stratified sampling by topic"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("CounselChat Test Query Generator")
    print("=" * 70)
    
    # Resolve paths
    base_dir = Path(__file__).parent.parent
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = base_dir / csv_path
    
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = base_dir / output_path
    
    # Check CSV exists
    if not csv_path.exists():
        print(f"\nError: CSV file not found: {csv_path}")
        return
    
    # Generate test queries
    test_queries = generate_test_queries(
        str(csv_path),
        num_queries=args.count,
        random_seed=args.seed,
        stratify_by_topic=not args.no_stratify
    )
    
    # Save to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(test_queries, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 70)
    print("Test Query Generation Complete!")
    print("=" * 70)
    print(f"CSV source: {csv_path}")
    print(f"Output file: {output_path}")
    print(f"Total queries: {len(test_queries)}")
    print("\nNext steps:")
    print("1. Run A/B test: python experiments/run_ab_test.py")
    print("2. Evaluate results: python experiments/evaluate_results.py <results_file>")
    print("=" * 70)


if __name__ == "__main__":
    main()
