# day4_indexing.py
# Day 4 — All Index Types in LlamaIndex

import sys
import os
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    SummaryIndex,
    KeywordTableIndex,
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.indices.document_summary import DocumentSummaryIndex
from llama_index.core.indices.tree.base import TreeIndex
from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from llama_index.vector_stores.qdrant import QdrantVectorStore
from settings import configure

logging.basicConfig(level=logging.WARNING)  # suppress info logs for cleaner output
logger = logging.getLogger(__name__)

PDF_PATH = r"C:\Users\jaina\OneDrive\Desktop\Learning Llama\data\Telangana_Overview_Styled.pdf"
QDRANT_URL = "http://localhost:6333"
EMBEDDING_DIM = 384
PERSIST_DIR = "storage/"


# -------------------------------------------------------
# Menu
# -------------------------------------------------------
def show_menu():
    print("\n" + "=" * 55)
    print("       LlamaIndex - Day 4")
    print("       All Index Types")
    print("=" * 55)
    print("Select LLM Provider:")
    print("  1 - Groq (cloud)")
    print("  2 - LM Studio (local)")
    print("=" * 55)
    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice == "1":
            return False
        elif choice == "2":
            return True
        else:
            print("Invalid. Enter 1 or 2.")


def load_documents():
    print("\nLoading documents...")
    documents = SimpleDirectoryReader(
        input_files=[PDF_PATH]
    ).load_data()
    print(f"Loaded {len(documents)} pages")
    return documents


def ask_question(query_engine, question: str):
    print(f"\nQuestion : {question}")
    response = query_engine.query(question)
    print(f"Answer   : {response}")
    print(f"\nSource nodes ({len(response.source_nodes)}):")
    for i, node in enumerate(response.source_nodes):
        score = f"{node.score:.4f}" if node.score is not None else "N/A"
        print(f"  Node {i+1} | Score: {score} | {node.text[:120]}...")


# -------------------------------------------------------
# INDEX 1 — VectorStoreIndex (in-memory)
# Definition: Stores embeddings in memory, uses cosine
# similarity to find relevant nodes at query time.
# Best for: All RAG and Q&A use cases — your default.
# -------------------------------------------------------
def demo_vector_index_memory(documents):
    print("\n" + "=" * 55)
    print("INDEX 1 — VectorStoreIndex (in-memory)")
    print("Uses: Similarity search | Embeddings: Yes")
    print("Best for: RAG, Q&A, semantic search")
    print("=" * 55)

    # Way 1 — from documents (quick)
    print("\n[Way 1] Building from documents...")
    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True
    )
    query_engine = index.as_query_engine(similarity_top_k=3)
    ask_question(query_engine, "What is Telangana known for?")

    # Way 2 — from nodes (production)
    print("\n[Way 2] Building from nodes (production approach)...")
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)
    nodes = [n for n in nodes if len(n.text.strip()) > 20]
    print(f"Nodes created: {len(nodes)}")

    index = VectorStoreIndex(nodes, show_progress=True)
    query_engine = index.as_query_engine(similarity_top_k=3)
    ask_question(query_engine, "What are the major industries of Telangana?")

    return index


# -------------------------------------------------------
# INDEX 2 — VectorStoreIndex (with Qdrant backend)
# Definition: Same as VectorStoreIndex but uses Qdrant
# as the persistent vector store backend.
# Best for: Production — vectors persist across runs.
# -------------------------------------------------------
def demo_vector_index_qdrant(documents):
    print("\n" + "=" * 55)
    print("INDEX 2 — VectorStoreIndex with Qdrant")
    print("Uses: Similarity search | Backend: Qdrant Docker")
    print("Best for: Production, persistent storage")
    print("=" * 55)

    try:
        collection_name = "telangana_day4"
        client = QdrantClient(url=QDRANT_URL)

        existing = [c.name for c in client.get_collections().collections]
        if collection_name not in existing:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE
                )
            )
            print(f"Created collection: {collection_name}")
        else:
            print(f"Collection exists: {collection_name}")

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name
        )
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        print("Index stored in Qdrant successfully")

        query_engine = index.as_query_engine(similarity_top_k=3)
        ask_question(query_engine, "What is the capital of Telangana?")

        return index

    except Exception as e:
        print(f"Qdrant not available: {e}")
        print("Skipping Qdrant demo — make sure Docker is running")
        return None


# -------------------------------------------------------
# INDEX 3 — SummaryIndex
# Definition: Stores all nodes in a sequential list.
# Reads through ALL nodes at query time — no similarity.
# Best for: Full document summarization only.
# Avoid for: Specific Q&A — reads everything every time.
# -------------------------------------------------------
def demo_summary_index(documents):
    print("\n" + "=" * 55)
    print("INDEX 3 — SummaryIndex")
    print("Uses: Sequential read | Embeddings: No")
    print("Best for: Full document summarization")
    print("Avoid for: Specific Q&A on large documents")
    print("=" * 55)

    index = SummaryIndex.from_documents(documents)

    # response_mode options:
    # compact       — fits context into fewer LLM calls (default)
    # tree_summarize — builds a tree of summaries (best quality)
    query_engine = index.as_query_engine(
        response_mode="tree_summarize"
    )
    ask_question(query_engine, "Give me a complete summary of this document")

# -------------------------------------------------------
# INDEX 4 — KeywordTableIndex
# Definition: Extracts keywords from nodes using LLM.
# Builds a keyword-to-node lookup table.
# Best for: Exact keyword-based search.
# Avoid for: Natural language queries.
# -------------------------------------------------------
def demo_keyword_index(documents):
    print("\n" + "=" * 55)
    print("INDEX 4 — KeywordTableIndex")
    print("Uses: Keyword matching | LLM during indexing: Yes")
    print("Best for: Exact keyword search")
    print("Avoid for: Natural language queries")
    print("=" * 55)

    index = KeywordTableIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    ask_question(query_engine, "Hyderabad economy technology")


# -------------------------------------------------------
# INDEX 5 — SimpleKeywordTableIndex
# Definition: Same as KeywordTableIndex but uses regex
# for keyword extraction — no LLM calls during indexing.
# Best for: Fast keyword indexing without LLM overhead.
# -------------------------------------------------------
def demo_simple_keyword_index(documents):
    print("\n" + "=" * 55)
    print("INDEX 5 — SimpleKeywordTableIndex")
    print("Uses: Regex keyword matching | LLM during indexing: No")
    print("Best for: Fast keyword indexing, no LLM overhead")
    print("=" * 55)

    from llama_index.core import SimpleKeywordTableIndex

    index = SimpleKeywordTableIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    ask_question(query_engine, "agriculture districts Telangana")


# -------------------------------------------------------
# INDEX 6 — TreeIndex
# Definition: Builds a hierarchical tree of summaries.
# Summarizes groups of nodes into parent nodes, then
# summarizes those further up until a root is reached.
# Best for: Broad topic exploration across large collections.
# Avoid for: Specific Q&A — overkill and slow.
# -------------------------------------------------------
def demo_tree_index(documents):
    print("\n" + "=" * 55)
    print("INDEX 6 — TreeIndex")
    print("Uses: Tree traversal | LLM during indexing: Yes")
    print("Best for: Broad topic exploration, large collections")
    print("Avoid for: Specific Q&A")
    print("=" * 55)

    index = TreeIndex.from_documents(documents)

    # child_branch_factor: how many children per parent node
    query_engine = index.as_query_engine(
        child_branch_factor=2
    )
    ask_question(query_engine, "What are the main themes of this document?")


# -------------------------------------------------------
# INDEX 7 — DocumentSummaryIndex
# Definition: Generates an LLM summary for each document
# and indexes that summary. Retrieval happens at document
# level first, then drills into the specific document.
# Best for: Multi-document setups with many large files.
# Avoid for: Single document Q&A — unnecessary overhead.
# -------------------------------------------------------
def demo_document_summary_index(documents):
    print("\n" + "=" * 55)
    print("INDEX 7 — DocumentSummaryIndex")
    print("Uses: Summary-based routing | LLM during indexing: Yes")
    print("Best for: Multi-document routing (50+ documents)")
    print("Avoid for: Single document Q&A")
    print("=" * 55)

    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    index = DocumentSummaryIndex.from_documents(
        documents,
        transformations=[splitter],
        show_progress=True,
        response_synthesizer_llm=Settings.llm
    )

    # Get the generated summary
    doc_id = documents[0].doc_id
    summary = index.get_document_summary(doc_id)
    print(f"\nGenerated document summary:")
    print(f"{summary[:400]}...")

    query_engine = index.as_query_engine()
    ask_question(query_engine, "What topics does this document cover?")


# -------------------------------------------------------
# INDEX 8 — Persist and Load
# Definition: Save index to disk once, load instantly
# on subsequent runs. Never re-embed unnecessarily.
# Best for: Every production pipeline.
# -------------------------------------------------------
def demo_persist_and_load(documents):
    print("\n" + "=" * 55)
    print("INDEX 8 — Persist and Load Pattern")
    print("Build once → Load forever")
    print("=" * 55)

    if os.path.exists(PERSIST_DIR):
        print("Existing index found — loading from disk...")
        storage_context = StorageContext.from_defaults(
            persist_dir=PERSIST_DIR
        )
        index = load_index_from_storage(storage_context)
        print("Loaded in seconds — no re-embedding needed")
    else:
        print("No existing index — building fresh...")
        index = VectorStoreIndex.from_documents(
            documents,
            show_progress=True
        )
        index.storage_context.persist(PERSIST_DIR)
        print(f"Index saved to {PERSIST_DIR}")

    query_engine = index.as_query_engine(similarity_top_k=3)
    ask_question(query_engine, "What is Telangana known for?")


# -------------------------------------------------------
# INDEX 9 — Insert, Delete, Refresh
# Definition: Modify an existing index without rebuilding.
# Best for: Production pipelines with changing data.
# -------------------------------------------------------
def demo_insert_delete(index):
    print("\n" + "=" * 55)
    print("INDEX 9 — Insert, Delete, Refresh")
    print("Modify existing index without rebuilding")
    print("=" * 55)

    # Insert a new document
    new_doc = Document(
        text="Telangana was formed on 2nd June 2014 as the 29th state of India after being carved out of Andhra Pradesh.",
        metadata={"source": "manual", "category": "telangana"}
    )
    index.insert(new_doc)
    print(f"Inserted new document: {new_doc.doc_id}")

    # Query after insert
    query_engine = index.as_query_engine(similarity_top_k=3)
    ask_question(query_engine, "When was Telangana formed?")

    # Delete the inserted document
    index.delete_ref_doc(new_doc.doc_id, delete_from_docstore=True)
    print(f"\nDeleted document: {new_doc.doc_id}")
    print("Index back to original state")


# -------------------------------------------------------
# Index Type Selector Menu
# -------------------------------------------------------
def index_selector_menu(documents):
    while True:
        print("\n" + "=" * 55)
        print("Select Index Type to Demo:")
        print("  1  - VectorStoreIndex (in-memory)")
        print("  2  - VectorStoreIndex (Qdrant backend)")
        print("  3  - SummaryIndex")
        print("  4  - KeywordTableIndex")
        print("  5  - SimpleKeywordTableIndex")
        print("  6  - TreeIndex")
        print("  7  - DocumentSummaryIndex")
        print("  8  - Persist and Load Pattern")
        print("  9  - Insert, Delete, Refresh")
        print("  10 - Run ALL demos")
        print("  0  - Exit")
        print("=" * 55)

        choice = input("Enter choice: ").strip()

        if choice == "1":
            demo_vector_index_memory(documents)

        elif choice == "2":
            demo_vector_index_qdrant(documents)

        elif choice == "3":
            demo_summary_index(documents)

        elif choice == "4":
            demo_keyword_index(documents)

        elif choice == "5":
            demo_simple_keyword_index(documents)

        elif choice == "6":
            demo_tree_index(documents)

        elif choice == "7":
            demo_document_summary_index(documents)

        elif choice == "8":
            demo_persist_and_load(documents)

        elif choice == "9":
            index = demo_vector_index_memory(documents)
            demo_insert_delete(index)

        elif choice == "10":
            index = demo_vector_index_memory(documents)
            demo_vector_index_qdrant(documents)
            demo_summary_index(documents)
            demo_keyword_index(documents)
            demo_simple_keyword_index(documents)
            demo_tree_index(documents)
            demo_document_summary_index(documents)
            demo_persist_and_load(documents)
            demo_insert_delete(index)
            print("\n" + "=" * 55)
            print("All demos complete!")
            print("=" * 55)

        elif choice == "0":
            print("\nExiting Day 4. Goodbye!")
            break

        else:
            print("Invalid choice. Enter 0-10.")


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    use_local = show_menu()
    configure(use_local=use_local)
    documents = load_documents()
    index_selector_menu(documents)


if __name__ == "__main__":
    main()