# Day 8 — Advanced Retrieval — Re-ranking & Hybrid Search

## What this script does
Implements production-grade retrieval techniques that go beyond basic
similarity search. Covers re-ranking with three different re-rankers,
BM25 keyword search, hybrid search combining vector + BM25, SentenceWindow
retrieval for precise sentence retrieval with surrounding context, and
a full retrieval strategy comparison demo.

## Concepts Covered
- Why similarity search alone is not enough
- Bi-encoder vs cross-encoder — how re-ranking works differently
- Re-ranking flow — retrieve Top-10, re-rank, return Top-3
- LLMRerank — uses LLM to score each node (most accurate, expensive)
- SentenceTransformerRerank — local cross-encoder (recommended, free)
- CohereRerank — Cohere API re-ranker (fast, paid)
- SimilarityPostprocessor — score threshold filter
- BM25Retriever — keyword-based retrieval, no embeddings needed
- Reciprocal Rank Fusion (RRF) — merging ranked lists from multiple retrievers
- QueryFusionRetriever — combines vector + BM25 with query variations
- num_queries — generates query variations for better coverage
- SentenceWindowNodeParser — indexes sentences, stores window in metadata
- MetadataReplacementPostProcessor — replaces sentence with window for LLM
- HierarchicalNodeParser — multi-level chunking (2048/512/128)
- AutoMergingRetriever — merges child nodes into parent when enough match
- Retrieval strategy comparison across all approaches

## Real-World Use Case
**Legal Document Search System**
A law firm searches across thousands of case documents and statutes.
Vector search finds semantically similar clauses. BM25 ensures exact
legal terms like "Section 47B" or "mens rea" are always matched precisely.
SentenceTransformerRerank re-scores retrieved clauses using a cross-encoder
trained on legal text — dramatically improving precision. SentenceWindow
retrieval finds the exact sentence containing a ruling but sends the
surrounding paragraph to the LLM for context. The full production pipeline
(Hybrid → Score filter → Re-rank) gives lawyers the most relevant
clauses every time.

## How to Run
```bash
python day8.py
# Option 1: Load documents
# Option 2: Build index
# Option 3: Baseline (run first for comparison)
# Option 5: SentenceTransformerRerank (recommended)
# Option 8: Hybrid search
# Option 9: Full production pipeline
# Option 11: Strategy comparison (most educational)
```

## Dependencies
```
llama-index-core
llama-index-postprocessor-sbert-rerank
llama-index-retrievers-bm25
rank-bm25
sentence-transformers
pypdf
```