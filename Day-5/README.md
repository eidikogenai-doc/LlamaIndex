# Day 5 — Storage & Persistence + Retrieval

## What this script does
Deep dive into how LlamaIndex stores and retrieves data. Covers StorageContext
internals, disk persistence, Qdrant as a vector store backend, explicit retrieval
without LLM calls, Top-K tuning, and metadata filtering across multiple documents.
Supports loading any PDF or multiple PDFs with custom metadata assignment.

## Concepts Covered
- StorageContext — manages DocStore, IndexStore, VectorStore
- Persisting index to disk — docstore.json, index_store.json, vector_store.json
- Loading index from disk — no re-embedding needed
- Load-or-build pattern — check if exists, build only if not
- Qdrant as persistent vector store backend
- VectorIndexRetriever — retrieve nodes WITHOUT calling LLM
- similarity_top_k — tuning how many nodes to retrieve
- Top-K comparison — 1 vs 2 vs 3 vs 5 nodes
- MetadataFilter — single filter condition
- MetadataFilters — multiple filters with AND / OR
- FilterCondition.AND — all conditions must match
- FilterCondition.OR — any condition can match
- Filter operators — EQ, NE, GT, GTE, LT, LTE, IN, NIN
- Interactive query with filter selection

## Real-World Use Case
**Insurance Claims Processing System**
An insurance company has thousands of policy documents, claim forms, and
exclusion schedules indexed in Qdrant. On startup the application loads
the persisted index in seconds instead of rebuilding. When a claims officer
queries "fire damage coverage", metadata filters restrict search to only
policy documents from the current year — not historical versions or claim
forms. VectorIndexRetriever is used by the QA team to debug exactly which
chunks are being retrieved before they reach the LLM.

## How to Run
```bash
python day5.py
# Option 1: Load documents (enter filenames + metadata)
# Option 2: Build index (disk) or Option 3: Build index (Qdrant)
# Option 5: Top-K comparison
# Option 6: Metadata filtering
# Option 7: Interactive query with filters
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