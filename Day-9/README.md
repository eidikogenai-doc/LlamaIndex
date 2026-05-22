# Day 9 — Multi-document & Metadata Filtering

## What this script does
Solves the multi-document problem where one large document dominates
results when multiple documents are in one index. Implements metadata
tagging during loading, single/AND/OR filter combinations, automatic
query routing with RouterQueryEngine, manual SubQuestion querying across
multiple documents, DocumentSummaryIndex for large collections, and
separate indexes per document for complete isolation.

## Concepts Covered
- Volume imbalance problem — why large documents dominate small ones
- Metadata tagging during loading — source_file, document_type, category, year, department
- Metadata flow from Document to Node automatically
- MetadataFilter — single condition (key, value, operator)
- MetadataFilters — multiple conditions combined
- FilterCondition.AND — all conditions must match
- FilterCondition.OR — any condition can match
- All filter operators — EQ, NE, GT, GTE, LT, LTE, IN, NIN, CONTAINS
- RouterQueryEngine — LLM automatically routes to correct engine
- LLMSingleSelector — routes to one engine
- LLMMultiSelector — can route to multiple engines
- QueryEngineTool — wraps engine with name and description
- SubQuestionQueryEngine — breaks query into sub-questions per document
- Manual SubQuestion approach — per-engine querying + LLM synthesis
- DocumentSummaryIndex — LLM summary per document for routing
- Separate indexes per document — complete isolation
- Filter comparison — same question with and without filter

## Real-World Use Case
**Multi-Department Enterprise Knowledge Base**
A large corporation has HR policies, legal contracts, finance reports,
and IT documentation all indexed together. Without metadata filtering
the IT documentation (thousands of pages) dominates every search.
With metadata filtering, HR queries only search HR documents, legal
queries only search contracts. RouterQueryEngine automatically detects
whether a query is about HR, legal, or finance and routes it to the
correct specialized engine. SubQuestion engine handles cross-department
queries like "Compare HR leave policy with the legal definition of
working hours" by querying both departments simultaneously and
synthesizing a combined answer.

## How to Run
```bash
python day9.py
# Option 1: Load documents — enter 2+ files with DIFFERENT categories
# Option 2: Build combined index
# Option 4: Demo 1 — baseline (no filter, see imbalance)
# Option 5: Demo 2 — single filter (fix the imbalance)
# Option 12: Demo 9 — filter comparison (most educational)
# Option 8: Demo 5 — RouterQueryEngine
# Option 9: Demo 6 — SubQuestion (manual approach)
# Option 11: Demo 8 — separate indexes
```

## Testing Tips
- Load both PDFs with DIFFERENT categories (e.g. technology vs general)
- Run Demo 1 (no filter) first — notice which document dominates
- Run Demo 9 (filter comparison) with same question — see the difference
- For RouterQueryEngine ask questions from each domain separately

## Dependencies
```
llama-index-core
llama-index-embeddings-huggingface
sentence-transformers
pypdf
```