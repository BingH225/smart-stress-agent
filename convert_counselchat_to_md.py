from __future__ import annotations

"""
CounselChat CSV to Markdown Converter

Converts counselchat CSV files to structured Markdown documents for RAG ingestion.
Handles HTML cleaning, deduplication, and metadata tagging.
"""

import re
from pathlib import Path
from typing import Set

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


def generate_filename(question_id: str, title: str, index: int) -> str:
    """Generate a clean filename from question ID and title."""
    # Use question ID if available, otherwise use index
    base = str(question_id) if question_id and not pd.isna(question_id) else f"cc_{index:04d}"
    
    # Clean title for filename (optional, for readability)
    if title and not pd.isna(title):
        clean_title = re.sub(r'[^\w\s-]', '', str(title).lower())
        clean_title = re.sub(r'[-\s]+', '_', clean_title)[:50]
        if clean_title:
            base = f"{base}_{clean_title}"
    
    return f"{base}.md"


def convert_csv_to_md(
    csv_path: str, 
    output_dir: str,
    seen_questions: Set[str],
    file_prefix: str = ""
) -> int:
    """
    Convert a single CSV file to Markdown documents.
    
    Args:
        csv_path: Path to the CSV file
        output_dir: Directory to write markdown files
        seen_questions: Set of question texts to track duplicates
        file_prefix: Prefix for filenames to distinguish sources
        
    Returns:
        Number of documents created
    """
    df = pd.read_csv(csv_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    created_count = 0
    skipped_count = 0
    
    print(f"\nProcessing {csv_path}...")
    print(f"Total rows: {len(df)}")
    
    for i, row in df.iterrows():
        # Extract fields (handle different column names)
        question_id = row.get('questionID', i)
        question_title = row.get('questionTitle', '')
        question_text = clean_html(row.get('questionText', ''))
        answer_text = clean_html(row.get('answerText', ''))
        
        # Handle topic/topics field
        topic = row.get('topic', row.get('topics', 'General'))
        if pd.isna(topic):
            topic = 'General'
        
        # Skip if content is too short or empty
        if len(question_text) < 20 or len(answer_text) < 30:
            skipped_count += 1
            continue
        
        # Check for duplicates
        question_key = question_text.lower()[:100]  # Use first 100 chars as key
        if question_key in seen_questions:
            skipped_count += 1
            continue
        seen_questions.add(question_key)
        
        # Generate markdown content
        content = f"""# Topic: {topic}

## Question

{question_text}

## Expert Answer

{answer_text}

---
*Source: CounselChat Dataset*  
*Topic: {topic}*  
*Document ID: {question_id}*
"""
        
        # Write to file
        filename = generate_filename(question_id, question_title, i)
        if file_prefix:
            filename = f"{file_prefix}_{filename}"
        
        file_path = output_path / filename
        file_path.write_text(content, encoding='utf-8')
        created_count += 1
        
        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1} rows... ({created_count} created, {skipped_count} skipped)")
    
    print(f"Completed: {created_count} documents created, {skipped_count} skipped")
    return created_count


def main():
    """Main conversion process."""
    print("=" * 60)
    print("CounselChat CSV to Markdown Converter")
    print("=" * 60)
    
    # Define paths
    base_dir = Path(__file__).parent
    rag_docs_dir = base_dir / "rag_docs"
    output_dir = rag_docs_dir / "counselchat"
    
    csv_files = [
        (rag_docs_dir / "20220401_counsel_chat.csv", "cc2022"),
        (rag_docs_dir / "counselchat-data.csv", "ccdata"),
    ]
    
    # Track duplicates across files
    seen_questions: Set[str] = set()
    total_created = 0
    
    # Convert each CSV file
    for csv_path, prefix in csv_files:
        if not csv_path.exists():
            print(f"\nWarning: {csv_path} not found, skipping...")
            continue
        
        created = convert_csv_to_md(
            str(csv_path),
            str(output_dir),
            seen_questions,
            file_prefix=prefix
        )
        total_created += created
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Conversion Complete!")
    print(f"Total documents created: {total_created}")
    print(f"Output directory: {output_dir}")
    print(f"Duplicates avoided: {len(seen_questions) - total_created}")
    print("=" * 60)
    
    # Verify sample files
    print("\nSample verification:")
    md_files = list(output_dir.glob("*.md"))
    if md_files:
        sample = md_files[0]
        print(f"\nFirst file: {sample.name}")
        print(f"Content preview:\n{sample.read_text(encoding='utf-8')[:300]}...")
    else:
        print("No markdown files found!")


if __name__ == "__main__":
    main()
