# Day 4 — Indexing — Vector Store Index

## What this script does
Covers all index types available in LlamaIndex with live demos of each.
Demonstrates building indexes from documents vs from nodes, using Qdrant
as a persistent vector store backend, inserting and deleting documents
without rebuilding, and the persist/load pattern for production use.
Includes the incremental indexing pipeline with a JSON tracker to prevent
duplicate indexing across runs.

## Concepts Covered
- What an index is and why it is needed
- VectorStoreIndex — similarity search, your default for all RAG
- SummaryIndex — sequential read for full document summarization
- KeywordTableIndex — LLM-based keyword matching
- SimpleKeywordTableIndex — regex-based keyword matching (no LLM)
- TreeIndex — hierarchical tree of summaries
- DocumentSummaryIndex — LLM summary per document for routing
- KnowledgeGraphIndex — entity and relationship extraction
- Building index from documents vs from nodes
- Qdrant as external vector store backend
- StorageContext with Qdrant
- index.insert() — add documents without rebuilding
- index.delete_ref_doc() — remove documents from index
- index.refresh_ref_docs() — update changed documents
- Persist and load pattern
- Incremental indexing with JSON tracker file
- Metadata filtering during retrieval

## Real-World Use Case
**Enterprise Document Management System**
A financial institution maintains thousands of regulatory documents that
change monthly. VectorStoreIndex with Qdrant stores all embeddings persistently.
The incremental indexing pipeline (tracker JSON) ensures new documents are
added without re-processing existing ones. When a regulation is updated,
delete_ref_doc removes the old version and insert adds the new one — no
full rebuild needed. DocumentSummaryIndex routes queries to the correct
regulatory category automatically.

## How to Run
```bash
# Main index types demo
python day4.py

# Incremental indexing pipeline
python incremental_indexing.py
# Option 1: Add documents
# Option 3: Query the index
# Option 4: View tracked files
```

## Dependencies
llama-index-core
llama-index-vector-stores-qdrant
qdrant-client
pypdf

## Prerequisites
Qdrant must be running in Docker:
```bash
docker run -p 6333:6333 qdrant/qdrant
```