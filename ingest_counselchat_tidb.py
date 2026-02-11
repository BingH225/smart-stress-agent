from __future__ import annotations

"""
TiDB-based CounselChat RAG Ingestion Script

Ingests counselchat documents into TiDB vector store in batches.
"""

import time
from pathlib import Path
from smartstress_langgraph.rag.ingestion import load_documents_from_folder
from smartstress_langgraph.rag.tidb_vector_store import get_tidb_vector_store


def ingest_in_batches(folder_path: str, batch_size: int = 10, delay_seconds: int = 1, tags: list[str] | None = None):
    """
    Ingest documents in batches to TiDB, skipping already-ingested documents.
    
    Args:
        folder_path: Path to folder containing documents
        batch_size: Number of documents per batch
        delay_seconds: Seconds to wait between batches
        tags: Optional tags to add to documents
    """
    print("=" * 60)
    print("TiDB CounselChat RAG Ingestion (Skip Existing)")
    print("=" * 60)
    
    # Load all documents
    print(f"\nLoading documents from {folder_path}...")
    docs = load_documents_from_folder(folder_path)
    total_docs = len(docs)
    
    if total_docs == 0:
        print("No documents found!")
        return
    
    print(f"Found {total_docs} documents")
    
    # Add tags if provided
    if tags:
        for doc in docs:
            doc.tags.extend(tags)
        print(f"Added tags: {tags}")
    
    # Get TiDB vector store
    print("\nConnecting to TiDB...")
    vector_store = get_tidb_vector_store()
    
    # Check which documents already exist
    print("\nChecking for existing documents in TiDB...")
    cursor = vector_store.connection.cursor()
    cursor.execute("SELECT id FROM rag_documents")
    existing_ids = {row[0] for row in cursor.fetchall()}
    cursor.close()
    
    print(f"Found {len(existing_ids)} existing documents in TiDB")
    
    # Filter out already-ingested documents
    new_docs = [doc for doc in docs if doc.id not in existing_ids]
    skipped_count = len(docs) - len(new_docs)
    
    print(f"Skipping {skipped_count} already-ingested documents")
    print(f"Will ingest {len(new_docs)} new documents")
    
    if len(new_docs) == 0:
        print("\n✓ All documents already in TiDB, nothing to ingest!")
        vector_store.close()
        return
    
    # Process in batches
    total_ingested = 0
    num_batches = (len(new_docs) + batch_size - 1) // batch_size
    
    print(f"\nProcessing {num_batches} batches of up to {batch_size} documents...")
    print(f"Delay between batches: {delay_seconds} seconds\n")
    
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(new_docs))
        batch = new_docs[start_idx:end_idx]
        
        print(f"Batch {batch_num + 1}/{num_batches}: Processing documents {start_idx + 1}-{end_idx}...")
        
        try:
            vector_store.add_documents(batch)
            total_ingested += len(batch)
            print(f"  ✓ Successfully ingested {len(batch)} documents (total new: {total_ingested}/{len(new_docs)})")
            
            # Delay between batches (except for the last one)
            if batch_num < num_batches - 1:
                time.sleep(delay_seconds)
        
        except Exception as e:
            print(f"  ✗ Error ingesting batch: {str(e)}")
            print(f"  Continuing with next batch...")
    
    # Clean up
    vector_store.close()
    
    # Summary
    print("\n" + "=" * 60)
    print("Ingestion Complete!")
    print(f"Previously existing: {skipped_count}")
    print(f"Newly ingested: {total_ingested}/{len(new_docs)}")
    print(f"Total in TiDB: {len(existing_ids) + total_ingested}")
    
    if total_ingested == len(new_docs):
        print("✓ All new documents successfully ingested to TiDB")
    else:
        print(f"⚠ Warning: {len(new_docs) - total_ingested} documents failed to ingest")
    
    print("=" * 60)


def main():
    """Main ingestion process."""
    base_dir = Path(__file__).parent
    counselchat_dir = base_dir / "rag_docs" / "counselchat"
    
    # Check if directory exists
    if not counselchat_dir.exists():
        print(f"\nError: Directory not found: {counselchat_dir}")
        return
    
    # Ingest with batching
    # Using batch size of 10 with 1 second delay
    ingest_in_batches(
        folder_path=str(counselchat_dir),
        batch_size=10,  # Process 10 documents at a time
        delay_seconds=1,  # 1 second delay between batches
        tags=["psychoeducation", "counselchat"]
    )


if __name__ == "__main__":
    main()
