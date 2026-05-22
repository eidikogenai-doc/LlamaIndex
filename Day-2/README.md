# Day 2 — Documents, Nodes & Chunking

## What this script does
Demonstrates how raw documents are split into smaller searchable chunks (nodes).
Covers all 7 chunking strategies available in LlamaIndex, compares their output,
shows how node relationships work, and demonstrates metadata flow from document to node.

## Concepts Covered
- Document object — text, metadata, doc_id, excluded metadata keys
- Node object — chunk of a document with ID, embedding, relationships
- Document vs Node differences
- Chunking — why it is needed (LLM context limits + retrieval quality)
- chunk_size and chunk_overlap — tuning for retrieval quality
- SentenceSplitter — splits on sentence boundaries (default choice)
- TokenTextSplitter — splits on exact token count
- SemanticSplitterNodeParser — splits when topic/meaning changes
- SentenceWindowNodeParser — each sentence + surrounding window in metadata
- HierarchicalNodeParser — multi-level chunks (2048/512/128 tokens)
- CodeSplitter — splits code by function/class boundaries
- MarkdownNodeParser — splits markdown by headers
- Node relationships — SOURCE, PREVIOUS, NEXT
- Metadata inheritance from Document to Node
- Effect of chunk size on node count

## Real-World Use Case
**Legal Document Processing Pipeline**
A law firm processes hundreds of contracts. Different chunking strategies
are needed for different document types — MarkdownNodeParser for structured
legal templates, SemanticSplitter for long narrative contracts where topics
change frequently, and SentenceWindowNodeParser for precise clause retrieval
with surrounding context. Getting chunking right directly determines whether
lawyers find the right clause or miss it entirely.

## How to Run
```bash
python day2.py
# Select LLM provider
# Choose any splitter demo from the menu
# Observe node counts and text previews for each strategy
```

## Dependencies
```
llama-index-core
sentence-transformers
pypdf
```