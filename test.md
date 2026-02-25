Here are the detailed PostgreSQL schemas optimized for `pgvector` and BM25, followed by the Python logic for the Retrieval API response formatting.

### 1. PostgreSQL Schema Design (Documents, Chunks, and Vectors)

To achieve "blazing fast" reads and writes, we use `JSONB` for flexible metadata, `tsvector` for sparse keyword search (BM25), and `vector` for our BERT dense embeddings.

**SQL**

```
-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- ==========================================
-- TABLE: documents (Parent Documents)
-- ==========================================
CREATE TABLE documents (
    parent_document_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    s3_path VARCHAR(1024) NOT NULL,
    status VARCHAR(50) DEFAULT 'INGESTING', -- INGESTING, PROCESSING, COMPLETED, FAILED
    metadata JSONB DEFAULT '{}'::jsonb,      -- Stores original filename, author, upload date, etc.
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexing metadata for fast NER/NLP pre-filtering (e.g., finding documents by a specific company)
CREATE INDEX idx_documents_metadata ON documents USING GIN (metadata);

-- ==========================================
-- TABLE: chunks (Document Chunks)
-- ==========================================
-- Note: In a production 1TB/hr system, this table should be partitioned 
-- (e.g., PARTITION BY RANGE (created_at)) to maintain performance over time.
CREATE TABLE chunks (
    chunk_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    parent_document_id UUID NOT NULL REFERENCES documents(parent_document_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,            -- Order of the chunk in the document
    text_content TEXT NOT NULL,              -- The actual cleaned text
    text_search TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text_content)) STORED, -- For BM25
    embedding VECTOR(768),                   -- BERT embedding (Standard bert-base is 768 dimensions)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index for BM25 Sparse Search
CREATE INDEX idx_chunks_text_search ON chunks USING GIN (text_search);

-- Index for Dense Vector Search (HNSW)
-- ef_construction and m are tuned for high recall. 
-- vector_cosine_ops is used for Cosine distance (<=>).
CREATE INDEX idx_chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)
WITH (m = 64, ef_construction = 256);

-- Index for fast lookups of chunks by parent document
CREATE INDEX idx_chunks_parent_id ON chunks(parent_document_id);
```

---

### 2. Python Pseudo-code: Dynamic Top-K vs. Top-N Logic

This logic lives at the very end of the Retrieval API pipeline. It assumes the query has already gone through Dense/Sparse search, RRF, MMR, and the final Cross-Encoder re-ranking.

The input `ranked_candidates` is a list of dictionary objects, strictly sorted by the Cross-Encoder score in descending order.

**Python**

```
from typing import List, Dict, Any

def format_retrieval_response(
    ranked_candidates: List[Dict[str, Any]], 
    request_type: str, 
    limit: int
) -> Dict[str, Any]:
    """
    Filters and formats the final re-ranked chunks based on Top-K chunks or Top-N documents.
  
    Args:
        ranked_candidates: List of chunk dictionaries, pre-sorted by Cross-Encoder score (highest to lowest).
                           Expected keys: 'chunk_id', 'parent_document_id', 'text_content', 'score', 's3_path'
        request_type: String, either 'top_k_chunks' or 'top_n_documents'.
        limit: Integer representing K or N.
      
    Returns:
        Dictionary containing the formatted response payload.
    """
    final_results = []
  
    if request_type == "top_k_chunks":
        # Simply take the highest scoring K chunks, regardless of which document they come from.
        final_results = ranked_candidates[:limit]
      
    elif request_type == "top_n_documents":
        # We need chunks representing N UNIQUE parent documents.
        # Since the list is already sorted by score, the first time we see a document ID, 
        # it is guaranteed to be that document's highest-scoring chunk.
        seen_document_ids = set()
      
        for candidate in ranked_candidates:
            doc_id = candidate.get("parent_document_id")
          
            if doc_id not in seen_document_ids:
                seen_document_ids.add(doc_id)
                final_results.append(candidate)
              
            # Break early once we hit the requested number of unique documents
            if len(seen_document_ids) == limit:
                break
              
    else:
        raise ValueError("Invalid request_type. Must be 'top_k_chunks' or 'top_n_documents'.")

    # Construct the final API response payload
    response_payload = {
        "status": "success",
        "retrieved_count": len(final_results),
        "request_type": request_type,
        "results": [
            {
                "chunk_id": res["chunk_id"],
                "parent_document_id": res["parent_document_id"],
                "s3_path": res["s3_path"],
                "text": res["text_content"],
                "relevance_score": res["score"]
            }
            for res in final_results
        ]
    }
  
    return response_payload
```
