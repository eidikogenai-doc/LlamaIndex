# 🦙 LlamaIndex Learning Journal — Complete 9-Day Course

> **Your personal, organized LlamaIndex learning path** — built from your real project files, code, and practice work.

---

## 📁 Folder Structure

```
LlamaIndex_Learning/
│
├── settings.py                        ← Shared LLM + Embedding config (used by all days)
├── .env                               ← GROQ_API_KEY
├── data/                              ← Your PDF files
│   ├── Telangana_Overview_Styled.pdf
│   └── A_comprehensive_review_on_applications_of_Raspberry_Pi.pdf
│
├── Day-1/                             ← Introduction, Architecture & LLMs/Settings
│   └── day1.py
│
├── Day-2/                             ← Documents, Nodes & Chunking Strategies
│   └── day2.py
│
├── Day-3/                             ← Embeddings, Vector Basics & Data Ingestion
│   └── day3.py
│
├── Day-4/                             ← Indexing (VectorStoreIndex & All Index Types)
│   ├── day4.py
│   └── incremental_indexing.py
│
├── Day-5/                             ← Storage, Persistence & Retrieval
│   └── day5.py
│
├── Day-6/                             ← Query Engine & Response Synthesis
│   └── day6.py
│
├── Day-7/                             ← Chat Engine with Memory & Context
│   └── day7.py
│
├── Day-8/                             ← Advanced Retrieval (Re-ranking & Hybrid Search)
│   └── day8.py
│
└── Day-9/                             ← Multi-document & Metadata Filtering
    └── day9.py
```

---

## 🗺️ Learning Roadmap

| Day | Topic | Key Concepts | Main Script |
|-----|-------|-------------|-------------|
| 1 | **Introduction & Architecture** | Pipeline, Settings, Groq, LM Studio, HuggingFace | `day1.py` |
| 2 | **Documents, Nodes & Chunking** | Document, Node, SentenceSplitter, SemanticSplitter, HierarchicalParser | `day2.py` |
| 3 | **Embeddings & Data Ingestion** | Vectors, Cosine Similarity, SimpleDirectoryReader, IngestionPipeline | `day3.py` |
| 4 | **Indexing** | VectorStoreIndex, SummaryIndex, KeywordIndex, Qdrant, Insert/Delete | `day4.py` |
| 5 | **Storage & Retrieval** | StorageContext, Persist/Load, VectorIndexRetriever, MetadataFilter | `day5.py` |
| 6 | **Query Engine & Synthesis** | ResponseModes, PromptTemplate, ChatPromptTemplate, Streaming | `day6.py` |
| 7 | **Chat Engine & Memory** | ChatMemoryBuffer, condense_plus_context, stream_chat, Reset | `day7.py` |
| 8 | **Advanced Retrieval** | SentenceTransformerRerank, BM25, HybridSearch, SentenceWindow | `day8.py` |
| 9 | **Multi-document & Filtering** | MetadataFilters, RouterQueryEngine, SubQuestion, DocumentSummaryIndex | `day9.py` |

---

## 🔑 Key Libraries Used

```
llama-index-core==0.14.21          — Core framework: Documents, Nodes, Index, QueryEngine, ChatEngine
llama-index-llms-groq              — Groq cloud LLM provider (fast inference, free tier)
llama-index-llms-openai-like       — LM Studio local LLM (OpenAI-compatible server)
llama-index-embeddings-huggingface — HuggingFace embedding models (all-MiniLM-L6-v2, 384 dims)
llama-index-vector-stores-qdrant   — Qdrant vector store integration
llama-index-postprocessor-sbert-rerank — SentenceTransformer cross-encoder re-ranker
llama-index-retrievers-bm25        — BM25 keyword-based retriever
sentence-transformers              — Runs HuggingFace models locally on CPU
qdrant-client                      — Qdrant Python client
python-dotenv                      — Load .env API keys
pypdf                              — PDF text extraction
numpy                              — Vector math (cosine similarity)
rank-bm25                          — BM25 algorithm implementation
requests                           — Web page loading
```

---

## ⚡ The Full RAG Pipeline (What You Built)

```
📄 PDF / CSV / Web Page
        ↓
[SimpleDirectoryReader]          Day 3  (auto-detects file type)
        ↓
[Metadata Enrichment]            Day 3 + 9  (source_file, category, year, department)
        ↓
[SentenceSplitter]               Day 2  (chunk_size=512, chunk_overlap=50)
        ↓
[Node Validation]                Day 3  (filter empty/short nodes)
        ↓
[HuggingFace Embedding]          Day 3  (all-MiniLM-L6-v2 → 384-dim vectors)
        ↓
[VectorStoreIndex + Qdrant]      Day 4 + 5  (persistent vector storage)
        ↓
[MetadataFilter]                 Day 5 + 9  (restrict to relevant documents)
        ↓
[Hybrid Retrieval: Vector + BM25] Day 8  (semantic + keyword combined)
        ↓
[SimilarityPostprocessor]        Day 6  (cutoff=0.3 — blocks out-of-domain answers)
        ↓
[SentenceTransformerRerank]      Day 8  (cross-encoder re-scores Top-10 → Top-3)
        ↓
[Response Synthesis]             Day 6  (compact mode + custom ChatPromptTemplate)
        ↓
[ChatMemoryBuffer]               Day 7  (condense_plus_context, token_limit=3000)
        ↓
Final Answer ✅
```

---

## 🚀 Quick Reference — Most Used Patterns

### Settings Configuration
```python
from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.llm = Groq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.chunk_size = 512
Settings.chunk_overlap = 50
```

### Load Documents
```python
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("data").load_data()                    # entire folder
documents = SimpleDirectoryReader(input_files=["data/file.pdf"]).load_data()  # specific file
```

### Build and Persist Index
```python
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage

if os.path.exists("storage/"):
    index = load_index_from_storage(StorageContext.from_defaults(persist_dir="storage/"))
else:
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist("storage/")
```

### Query Engine with Score Threshold
```python
from llama_index.core.postprocessor import SimilarityPostprocessor

query_engine = index.as_query_engine(
    similarity_top_k=3,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.3)]
)
response = query_engine.query("Your question here")
print(response)
```

### Chat Engine with Memory
```python
from llama_index.core.memory import ChatMemoryBuffer

memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    memory=memory,
    system_prompt="You are a document assistant. Answer only from context.",
    verbose=True
)
response = chat_engine.chat("What is Telangana known for?")
print(response.response)
```

### Streaming Response
```python
chat_engine = index.as_chat_engine(chat_mode="condense_plus_context", streaming=True)
response = chat_engine.stream_chat("Your question")
for token in response.response_gen:
    print(token, end="", flush=True)
```

### Metadata Filtering
```python
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterCondition

filters = MetadataFilters(filters=[
    MetadataFilter(key="source_file", value="Telangana_Overview_Styled.pdf"),
    MetadataFilter(key="year", value="2025")
], condition=FilterCondition.AND)

query_engine = index.as_query_engine(similarity_top_k=3, filters=filters)
```

### Hybrid Search + Re-ranking (Production)
```python
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer

vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=10)
bm25_retriever = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=10)

hybrid_retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    similarity_top_k=10,
    num_queries=4,
    mode="reciprocal_rerank",
    use_async=False
)
reranker = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=3)
query_engine = RetrieverQueryEngine(
    retriever=hybrid_retriever,
    response_synthesizer=get_response_synthesizer(response_mode="compact"),
    node_postprocessors=[reranker]
)
```

### Router Query Engine
```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool

tool1 = QueryEngineTool.from_defaults(query_engine=engine1, name="insurance_tool",
    description="Use for insurance policy questions.")
tool2 = QueryEngineTool.from_defaults(query_engine=engine2, name="tech_tool",
    description="Use for technology and Raspberry Pi questions.")

router = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[tool1, tool2],
    verbose=True
)
```

---

## 🗂️ Day-wise Summary

### Day 1 — Introduction, Architecture & LLMs/Settings Configuration
- LlamaIndex 5-stage pipeline: **Load → Index → Store → Query → Respond**
- `Settings` object replaces old `ServiceContext` in LlamaIndex 0.10+
- Configure **Groq** (cloud) or **LM Studio** (local) as LLM with one flag
- `use_local=False` → Groq | `use_local=True` → LM Studio
- **Real-world use:** Enterprise document Q&A — employees ask questions, get answers from company PDFs

### Day 2 — Documents, Nodes & Chunking Strategies
- **Document** = entire page or file | **Node** = small chunk with embedding
- 7 splitter types: SentenceSplitter, TokenText, Semantic, SentenceWindow, Hierarchical, Code, Markdown
- `chunk_size=512, chunk_overlap=50` is your production default
- Metadata flows automatically from Document → every Node
- **Real-world use:** Legal document processing — different splitters for contracts vs structured templates

### Day 3 — Embeddings, Vector Basics & Data Ingestion
- Embedding = text → 384-dimensional vector | Similar text = similar numbers
- Cosine similarity score: 1.0 (identical) → 0.0 (unrelated)
- `SimpleDirectoryReader` handles all file types in all LlamaIndex versions
- `IngestionPipeline` = load + chunk + extract metadata in one step
- **Real-world use:** Hospital knowledge base ingesting PDFs, CSVs, and web pages into one searchable system

### Day 4 — Indexing (VectorStoreIndex & All Index Types)
- 8 index types: VectorStore, Summary, KeywordTable, SimpleKeyword, Tree, DocumentSummary, KnowledgeGraph, PropertyGraph
- **VectorStoreIndex** is your default for all RAG — always start here
- Build from documents (quick) vs from nodes (production — full control)
- `insert()`, `delete_ref_doc()`, `refresh_ref_docs()` — update without rebuilding
- **Real-world use:** Financial institution managing thousands of regulatory documents with incremental updates

### Day 5 — Storage, Persistence & Retrieval
- `StorageContext` manages DocStore + IndexStore + VectorStore
- Load-or-build pattern: first run = minutes, every run after = seconds
- `VectorIndexRetriever` — retrieve nodes WITHOUT calling LLM (debug tool)
- `MetadataFilter` — filter which nodes get searched BEFORE similarity runs
- **Real-world use:** Insurance claims system filtering by document_type + year before every query

### Day 6 — Query Engine & Response Synthesis
- `QueryEngine = Retriever + Response Synthesizer`
- 6 response modes: compact (default), refine, tree_summarize, simple_summarize, no_text, accumulate
- `SimilarityPostprocessor(cutoff=0.3)` — prevents LLM answering from training knowledge
- `ChatPromptTemplate` with `MessageRole.SYSTEM` sets domain persona
- **Real-world use:** Telecom support bot — compact mode for cost, score threshold blocks competitor questions

### Day 7 — Chat Engine with Memory & Context
- `ChatEngine = QueryEngine + conversation memory`
- `condense_plus_context` — best production mode, handles follow-up questions correctly
- `ChatMemoryBuffer(token_limit=3000)` — prevents context window overflow
- `stream_chat()` — tokens appear live for better user experience
- **Real-world use:** HR policy chatbot — employees ask follow-up questions using "it" and "its" naturally

### Day 8 — Advanced Retrieval (Re-ranking & Hybrid Search)
- Re-ranking flow: retrieve Top-10 → cross-encoder re-scores → return Top-3
- `SentenceTransformerRerank` — free, local, production choice
- BM25 catches exact keyword matches that vector search misses
- `QueryFusionRetriever` combines vector + BM25 with query variations (RRF)
- **Real-world use:** Legal search — vector finds semantically similar clauses, BM25 matches exact legal terms like "Section 47B"

### Day 9 — Multi-document & Metadata Filtering
- Volume imbalance problem: large docs dominate small ones without filtering
- Tag every document at load time — metadata flows to all nodes automatically
- `FilterCondition.AND` / `FilterCondition.OR` for complex filter logic
- `RouterQueryEngine` auto-routes queries to correct engine using LLM
- **Real-world use:** Enterprise knowledge base — HR queries only search HR docs, legal queries only search contracts

---

## 🛠️ Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| Binary PDF content | Scanned image-based PDF | Use text-based PDFs only |
| Out-of-domain LLM answers | No score threshold set | Add `SimilarityPostprocessor(cutoff=0.3)` |
| Follow-up questions fail | Using QueryEngine not ChatEngine | Switch to `condense_plus_context` mode |
| Duplicate Qdrant results | Same doc indexed multiple times | Use tracker JSON, delete collection, rebuild |
| Slow embedding every run | No persistence | Use load-or-build pattern with `persist()` |
| Import errors on readers | LlamaIndex version mismatch | Use `SimpleDirectoryReader` for all loading |
| BM25 import error | Version incompatibility | `pip install llama-index-retrievers-bm25 --upgrade` |
| SubQuestion Uber/Lyft bug | Hardcoded examples in LLM prompt | Use manual per-engine querying + LLM synthesis |
| Score shows None | Non-vector index (Summary/Keyword) | `f"{node.score:.4f}" if node.score else "N/A"` |
| OneDrive file corruption | OneDrive sync locks binary files | Move project to `C:\Projects\` outside OneDrive |

---

## 🏗️ Prerequisites

### 1. Install Dependencies
```bash
pip install llama-index-core==0.14.21
pip install llama-index-llms-groq
pip install llama-index-llms-openai-like
pip install llama-index-embeddings-huggingface
pip install llama-index-vector-stores-qdrant
pip install llama-index-postprocessor-sbert-rerank
pip install llama-index-retrievers-bm25
pip install sentence-transformers qdrant-client python-dotenv pypdf numpy rank-bm25
```

### 2. Set up `.env` file
```
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Start Qdrant (Docker)
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 4. Run any day
```bash
python Day-1/day1.py
# Select 1 for Groq or 2 for LM Studio
```

---

## 🧠 LLM Provider Reference

| Provider | Type | When to Use | Config |
|----------|------|-------------|--------|
| Groq | Cloud (fast) | Online development | `api_key` from `.env` |
| LM Studio | Local (offline) | No internet / air-gapped | Run LM Studio, load model, start server |

```python
# Switch providers with one line in any script
configure(use_local=False)   # Groq
configure(use_local=True)    # LM Studio
```

---

## 📚 Read the code comments in each script!

Every day's Python file has:
- ✅ What the script does (top comment block)
- ✅ What each function demonstrates
- ✅ Why each parameter is set the way it is
- ✅ Inline comments explaining each step
- ✅ Alternatives mentioned where relevant

---

*Happy learning! You've covered a full professional LlamaIndex curriculum from basics to production. 🎓*
