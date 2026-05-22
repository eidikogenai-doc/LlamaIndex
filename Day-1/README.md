# Day 1 — Introduction & Architecture + LLMs & Settings Configuration

## What this script does
Sets up the complete LlamaIndex foundation. Configures the global Settings object
with Groq (cloud) or LM Studio (local) as the LLM and HuggingFace all-MiniLM-L6-v2
as the embedding model. Loads a PDF document, builds an in-memory VectorStoreIndex,
creates a QueryEngine, and answers questions with source node inspection.

## Concepts Covered
- LlamaIndex 5-stage pipeline: Load → Index → Store → Query → Respond
- Core building blocks: Document, Node, Index, Retriever, QueryEngine
- Settings object (LlamaIndex 0.10+) — replaces old ServiceContext
- Groq cloud LLM configuration
- LM Studio local LLM configuration
- HuggingFace embedding model (all-MiniLM-L6-v2, 384 dimensions)
- SimpleDirectoryReader for loading PDFs
- VectorStoreIndex — similarity-based indexing
- QueryEngine — retriever + response synthesizer combined
- Source node inspection — see which chunks were used to answer

## Real-World Use Case
**Enterprise Document Q&A System**
A company has hundreds of internal policy PDFs. Instead of employees
manually reading through documents, they ask natural language questions
and get instant answers grounded in the actual document content.
Example: "What is the leave policy for new employees?" returns the
exact policy text with source page references.

## Key Files
- `day1.py` — Main script
- `settings.py` — Reusable global LLM + embedding configuration

## How to Run
```bash
python day1.py
# Select 1 for Groq (cloud) or 2 for LM Studio (local)
# Ask questions about your PDF
# Type 'exit' to quit
```

## Dependencies
```
llama-index-core
llama-index-llms-groq
llama-index-llms-openai-like
llama-index-embeddings-huggingface
sentence-transformers
python-dotenv
pypdf
```