"""
TiDB Vector Store Implementation

Replaces ChromaDB with TiDB for RAG document and embedding storage.
"""

from __future__ import annotations

import os
import json
from typing import Any, List, Sequence, Tuple
from pathlib import Path

import mysql.connector
from mysql.connector import Error as MySQLError
from dotenv import load_dotenv

from .schemas import RagDocument
from ..llm import embed_documents


class TiDBVectorStore:
    """
    Vector store implementation using TiDB for persistence.
    
    Stores documents and their embeddings in TiDB tables with vector similarity search support.
    """
    
    def __init__(self):
        """Initialize TiDB connection from environment variables."""
        load_dotenv()
        
        self.config = {
            'host': os.getenv('DB_HOST'),
            'port': int(os.getenv('DB_PORT', 4000)),
            'user': os.getenv('DB_USERNAME', '').strip("'\""),
            'password': os.getenv('DB_PASSWORD', '').strip("'\""),
            'database': os.getenv('DB_DATABASE', 'test'),
            'ssl_disabled': False
        }
        
        self.connection = None
        self._connect()
        self._create_tables()
    
    def _connect(self):
        """Establish connection to TiDB."""
        try:
            self.connection = mysql.connector.connect(**self.config)
            if self.connection.is_connected():
                print(f"✓ Connected to TiDB at {self.config['host']}")
        except MySQLError as e:
            print(f"✗ Error connecting to TiDB: {e}")
            raise
    
    def _create_tables(self):
        """Create tables for documents and embeddings if they don't exist."""
        cursor = self.connection.cursor()
        
        # Documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_documents (
                id VARCHAR(255) PRIMARY KEY,
                content TEXT NOT NULL,
                source VARCHAR(500),
                section VARCHAR(255),
                created_at DATETIME,
                tags TEXT,
                embedding_id VARCHAR(255),
                INDEX idx_source (source),
                INDEX idx_created (created_at)
            )
        """)
        
        # Embeddings table (stores vectors as JSON for now)
        # TiDB can use JSON or VECTOR type depending on version
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_embeddings (
                id VARCHAR(255) PRIMARY KEY,
                document_id VARCHAR(255) NOT NULL,
                embedding JSON NOT NULL,
                dimension INT,
                INDEX idx_document (document_id),
                FOREIGN KEY (document_id) REFERENCES rag_documents(id) ON DELETE CASCADE
            )
        """)
        
        self.connection.commit()
        cursor.close()
        print("✓ TiDB tables created/verified")
    
    def add_documents(self, docs: Sequence[RagDocument]) -> None:
        """
        Add documents to TiDB with their embeddings.
        
        Args:
            docs: Sequence of RagDocument objects to add
        """
        if not docs:
            return
        
        # Generate embeddings for all documents
        texts = [d.content for d in docs]
        print(f"  Generating embeddings for {len(texts)} documents...")
        embeddings = embed_documents(texts)
        
        cursor = self.connection.cursor()
        
        for i, doc in enumerate(docs):
            doc_id = doc.id or f"doc-{i}"
            embedding_id = f"emb-{doc_id}"
            
            try:
                # Insert document
                cursor.execute("""
                    INSERT INTO rag_documents (id, content, source, section, created_at, tags, embedding_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE 
                        content = VALUES(content),
                        source = VALUES(source),
                        section = VALUES(section),
                        created_at = VALUES(created_at),
                        tags = VALUES(tags),
                        embedding_id = VALUES(embedding_id)
                """, (
                    doc_id,
                    doc.content,
                    doc.source,
                    doc.section,
                    doc.created_at,
                    ", ".join(doc.tags) if doc.tags else None,
                    embedding_id
                ))
                
                # Insert embedding
                embedding_json = json.dumps(embeddings[i])
                cursor.execute("""
                    INSERT INTO rag_embeddings (id, document_id, embedding, dimension)
                    VALUES (%s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        embedding = VALUES(embedding),
                        dimension = VALUES(dimension)
                """, (
                    embedding_id,
                    doc_id,
                    embedding_json,
                    len(embeddings[i])
                ))
                
            except MySQLError as e:
                print(f"  Warning: Error inserting document {doc_id}: {e}")
                continue
        
        self.connection.commit()
        cursor.close()
    
    def similarity_search(
        self, query: str, k: int = 5
    ) -> List[Tuple[RagDocument, float]]:
        """
        Search for similar documents using cosine similarity.
        
        Args:
            query: Query text to search for
            k: Number of results to return
            
        Returns:
            List of (document, similarity_score) tuples
        """
        # Generate query embedding
        query_embedding = embed_documents([query])[0]
        
        cursor = self.connection.cursor(dictionary=True)
        
        # Fetch all documents with embeddings
        cursor.execute("""
            SELECT d.id, d.content, d.source, d.section, d.created_at, d.tags, e.embedding
            FROM rag_documents d
            JOIN rag_embeddings e ON d.embedding_id = e.id
        """)
        
        results = []
        for row in cursor.fetchall():
            doc_embedding = json.loads(row['embedding'])
            
            # Compute cosine similarity
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            
            doc = RagDocument(
                id=row['id'],
                content=row['content'],
                source=row['source'],
                section=row['section'],
                created_at=row['created_at'],
                tags=row['tags'].split(", ") if row['tags'] else []
            )
            
            results.append((doc, similarity))
        
        cursor.close()
        
        # Sort by similarity and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def close(self):
        """Close the database connection."""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("✓ TiDB connection closed")


def get_tidb_vector_store() -> TiDBVectorStore:
    """
    Get a TiDB-based vector store instance.
    
    Returns:
        TiDBVectorStore instance
    """
    return TiDBVectorStore()
