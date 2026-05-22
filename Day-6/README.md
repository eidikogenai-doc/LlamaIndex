# Day 6 — Query Engine + Response Synthesis

## What this script does
Covers the complete QueryEngine layer — how queries are processed, how
retrieved nodes are synthesized into answers, all response modes with
live comparison, custom prompt templates, streaming responses, and deep
source node inspection. Includes score threshold to prevent out-of-domain
answers from LLM training knowledge.

## Concepts Covered
- QueryEngine = Retriever + Response Synthesizer
- index.as_query_engine() — simple creation
- RetrieverQueryEngine — explicit control over retriever and synthesizer
- get_response_synthesizer — create synthesizer directly
- Response modes:
  - compact — packs nodes into minimum LLM calls (default)
  - refine — iteratively refines answer node by node
  - tree_summarize — builds tree of summaries (best for many nodes)
  - simple_summarize — single LLM call, truncates if needed
  - no_text — returns nodes without calling LLM (debugging)
  - accumulate — one LLM call per node, combines results
- PromptTemplate — custom string prompt with {context_str} {query_str}
- ChatPromptTemplate — system + user message structure for chat models
- ChatMessage and MessageRole — SYSTEM, USER, ASSISTANT
- Streaming responses — response_gen token iterator
- Source node inspection — score, file, page, full text
- SimilarityPostprocessor — cutoff=0.3 prevents out-of-domain answers

## Real-World Use Case
**Customer Support Chatbot with Strict Domain Control**
A telecom company deploys a support bot that must only answer from their
product documentation — never from the LLM's general knowledge. compact
mode minimizes API costs for simple queries. refine mode handles complex
multi-part questions about billing. SimilarityPostprocessor(cutoff=0.3)
ensures the bot responds "This information is not in our documentation"
for questions outside the product domain (e.g. competitor questions).
Custom ChatPromptTemplate sets the bot's persona as a telecom expert.

## How to Run
```bash
python day6.py
# Option 1: Load documents
# Option 2: Build index
# Option 5: Response mode comparison (most educational)
# Option 6: Custom prompt vs default prompt comparison
# Option 8: Streaming response
# Option 9: Source node deep inspection
```

## Dependencies
```
llama-index-core
llama-index-embeddings-huggingface
sentence-transformers
pypdf
```