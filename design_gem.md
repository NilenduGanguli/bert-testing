## 1. High-Level System Architecture

To handle ~28 documents per second (100k/hr) and ~277 MB/s throughput, the pipeline is entirely decoupled into microservices communicating via **RabbitMQ** (using Quorum Queues for high availability) and utilizing **Redis** for distributed caching. The database layer uses **PostgreSQL** with **PGVector** for storing metadata, chunks, and dense vectors.

### Core Components

* **API Gateway & Ingestion REST API:** Handles incoming PDF payloads and metadata.
* **Object Storage (S3):** Acts as the immutable source of truth for raw documents.
* **Message Broker (RabbitMQ):** Orchestrates tasks between ingestion, OCR, chunking, and embedding.
* **Compute Workers:** Independent, horizontally scaled worker nodes for Ingestion, OCR, Chunking, and Embedding.
* **Retrieval REST API:** Handles user queries, NLP preprocessing, and hybrid search.

---

## 2. Phase A: Ingestion and Routing

The ingestion phase focuses on fast acknowledgment and shifting heavy processing to background workers.

### 2.1 The Ingestion API

The REST endpoint receives the PDF. To prevent blocking:

1. **Stream to S3:** The payload is directly streamed to S3 to avoid memory exhaustion on the API container.
2. **Database Initialization:** A record is created in the PostgreSQL `documents` table, generating a `parent_document_id`.
3. **Queue Dispatch:** The API pushes an ingestion task (containing the S3 path and `parent_document_id`) to the `ingestion_queue` in RabbitMQ.
4. **Response:** The API immediately returns a `202 Accepted` with the tracking ID.

### 2.2 The Ingestion Worker & OCR Router

Workers continuously consume from the `ingestion_queue`.

* **Ingestion Router:** The worker streams the PDF from S3 into memory and analyzes the pages. It separates readable text layers from image blocks.
* **OCR Delegation:** Images are separated, batched, and sent to the dedicated **OCR Service Container** (treated as a high-throughput black box).
* **Preprocessing:** The raw text and OCR-extracted text are merged. The worker applies aggressive text cleaning (removing null bytes, normalizing unicode, stripping excessive whitespace).

---

## 3. Phase B: Modular Chunking Engine

Given the requirement for future-proofing, the chunking mechanism operates behind a standardized interface.

### 3.1 Chunking Strategy

* **Recursive Character Splitting:** The default implementation splits the cleaned text using hierarchical separators (e.g., `\n\n`, `\n`, , `""`) with a defined chunk size (e.g., 512 tokens) and overlap (e.g., 50 tokens) to preserve context.
* **Modularity:** The chunker is designed as an injectable dependency (e.g., `BaseChunker` interface). If we later require semantic chunking, we simply register a new class implementing this interface.

### 3.2 Database & Queue Handoff

1. **Database Update:** The chunker generates `chunk_id`s and performs a massive `COPY` or bulk `INSERT` into the PostgreSQL `chunks` table, linking them to the `parent_document_id`.
2. **Task Delegation:** Instead of passing the actual text through the message broker (which would choke RabbitMQ at 1 TB/hr), the worker pushes batches of `chunk_id`s to the `chunking_queue`.

---

## 4. Phase C: Embedding and Storage

This phase transforms the textual chunks into dense vectors using the specified model.

### 4.1 Embedding Worker

* The worker pops `chunk_id` batches from the `chunking_queue`.
* It fetches the corresponding text chunks from PostgreSQL.
* **Redis Caching:** To optimize, it checks Redis. If a duplicate chunk (exact text hash) was processed recently, it fetches the cached embedding, skipping the model inference.
* **Model Execution:** It processes unseen chunks through the `bert-embedding` model using high-throughput batched inference (e.g., batch size of 64 or 128 depending on GPU memory).

### 4.2 PGVector Storage & Indexing

The vectors are stored in PostgreSQL using PGVector. For blazing-fast retrieval at this scale, we use the **HNSW (Hierarchical Navigable Small World)** index.

* **Bulk Upsert:** Vectors are written using bulk operations.
* **Index Tuning:** Given the dataset size, we configure the PGVector HNSW index with `m = 64` (max connections per layer) and `ef_construction = 256` (candidate list size during build) to maximize recall without sacrificing query speed.

---

## 5. Phase D: Retrieval API & Hybrid Search Pipeline

The Retrieval API is a masterclass in combining sparse and dense techniques, rigorously adhering to the constraint of only using BERT models.

### 5.1 Query Preprocessing & Audit

1. **Logging:** The raw query is instantly asynchronously logged to the `retrieval_audit` table.
2. **NER/NLP:** The query is passed through `bert-multilingual` configured for Named Entity Recognition (NER). This extracts critical entities (dates, organizations, locations) which can be used to apply hard pre-filters in the PostgreSQL query (e.g., `WHERE metadata->>'company' = 'Google'`), drastically reducing the search space.

### 5.2 Hybrid Search (Dense + Sparse)

We execute two searches in parallel against the database:

1. **Dense Search:** The query is embedded using `bert-embedding` (functioning as a bi-encoder). It queries PGVector using cosine distance (`<=>`).
2. **Sparse Search:** A standard BM25 query is executed against a `tsvector` column in PostgreSQL for exact keyword matching.

### 5.3 Reciprocal Rank Fusion (RRF)

The results from both searches are unified using RRF to balance semantic meaning with keyword precision.

$$
RRFScore(d) = \frac{1}{k + rank_{dense}(d)} + \frac{1}{k + rank_{sparse}(d)}
$$

*(Where **$k$** is a smoothing constant, typically 60).*

### 5.4 Maximal Marginal Relevance (MMR)

To ensure the retrieved chunks aren't redundant (e.g., 10 chunks saying the exact same thing), we apply MMR. This penalizes chunks that are too similar to those already selected, promoting diversity in the context window.

$$
MMR = \arg\max_{D_i \in R \setminus S} \left[ \lambda \cdot Sim_1(D_i, Q) - (1 - \lambda) \cdot \max_{D_j \in S} Sim_2(D_i, D_j) \right]
$$

### 5.5 Final Re-ranking (Cross-Encoder)

The top candidates surviving the MMR filter are paired with the original query and fed back into `bert-embedding` (now configured as a cross-encoder). The cross-encoder calculates a highly accurate relevance score for each Query-Chunk pair, generating the absolute final ranking.

### 5.6 Response Formatting (Top-K vs. Top-N)

The API logic dynamically handles the user's request format:

* **If Top-K Chunks:** It simply returns the highest-ranked **$K$** chunks.
* **If Top-N Documents:** It iterates through the cross-encoder ranked list, picking the single highest-scoring chunk for a given `parent_document_id`. It stops once **$N$** unique parent documents are represented, packaging them with their S3 paths, metadata, and audit trace IDs.

---

## 6. Infrastructure & Performance Tuning (1 TB / hr)

To guarantee resilience and speed at this massive scale, the infrastructure requires specific tuning:

| **Component**    | **Tuning Strategy for Scale**                                                                                                                                                                                                                                                                 |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **RabbitMQ**     | Use**Quorum Queues**for fault tolerance. Enable**Lazy Queues**(default in newer versions) to spool messages to disk immediately, preventing RAM exhaustion during massive ingestion spikes. Use the Consistent Hash Exchange to evenly distribute messages across thousands of workers. |
| **Redis**        | Deploy as a cluster. Cache the BM25 term frequencies, recent query embeddings, and parent document paths. Set a strict LRU (Least Recently Used) eviction policy.                                                                                                                                   |
| **PostgreSQL**   | Implement**Table Partitioning**on the `chunks`and `audit`tables by date or hash. Ensure `maintenance_work_mem`is heavily allocated (e.g., 8GB+) so the HNSW index can build efficiently in memory.                                                                                      |
| **BERT Workers** | Run all `bert-embedding`and `bert-multilingual`workers on GPU-accelerated nodes using TensorRT or ONNX Runtime for optimized inference. Batch inputs aggressively.                                                                                                                              |


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

