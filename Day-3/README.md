# Day 3 — Embeddings & Vector Basics + Data Ingestion

## What this script does
Explains what embeddings are by generating real vectors and computing cosine
similarity manually. Demonstrates all data loading approaches including folder
loading, single file loading, web loading, custom metadata attachment, and the
IngestionPipeline pattern for combining loading + chunking + metadata extraction.

## Concepts Covered
- Embeddings — converting text to numerical vectors
- Embedding dimensions — all-MiniLM-L6-v2 produces 384-dimensional vectors
- Cosine similarity — mathematical basis of semantic search
- Why similar text produces similar vectors
- SimpleDirectoryReader — auto-detects PDF, txt, docx, csv, json
- Loading specific files vs entire folders
- Web loading using requests + Document object
- Custom metadata — adding source, category, author, project fields
- excluded_llm_metadata_keys — keeping prompts clean
- excluded_embed_metadata_keys — keeping embeddings focused
- IngestionPipeline — load + chunk + extract metadata in one step
- TitleExtractor — LLM extracts document title per node
- KeywordExtractor — LLM extracts top keywords per node
- Manual Document creation in code

## Real-World Use Case
**Multi-Source Knowledge Base Ingestion**
A hospital system ingests medical guidelines from PDFs, live updates from
health authority websites, and structured data from CSV patient records —
all into one unified searchable knowledge base. Custom metadata tags each
source with department, date, and document type so doctors can filter
results by source during retrieval. Keywords extracted during ingestion
enable both semantic and keyword-based search later.

## How to Run
```bash
python day3.py
# Select LLM provider
# Run demos from menu
# Demo 1 shows raw embedding vectors and similarity scores
# Demo 7 shows manual document creation
```

## Dependencies
```
llama-index-core
llama-index-embeddings-huggingface
sentence-transformers
numpy
requests
pypdf
```