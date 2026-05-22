# day5_storage.py
# Day 5 — Storage & Persistence + Retrieval
# Supports: Single PDF, Multiple PDFs, Entire folder
import os
import sys
import json
import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterCondition
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from settings import configure
# -------------------------------------------------------
# Config
# -------------------------------------------------------
PERSIST_DIR = "storage/"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "day5_collection"
EMBEDDING_DIM = 384
TRACKER_FILE = "day5_tracker.json"
DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data")
)

# -------------------------------------------------------
# LLM Menu
# -------------------------------------------------------
def show_menu():
    print("\n" + "=" * 55)
    print("       LlamaIndex - Day 5")
    print("       Storage & Persistence + Retrieval")
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
        print("Invalid. Enter 1 or 2.")

# -------------------------------------------------------
# Tracker — tracks what files are indexed
# -------------------------------------------------------
def load_tracker() -> dict:
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, "r") as f:
            return json.load(f)
    return {}


def save_tracker(tracker: dict):
    with open(TRACKER_FILE, "w") as f:
        json.dump(tracker, f, indent=2)


# -------------------------------------------------------
# Document Loader
# -------------------------------------------------------
def load_documents_from_user() -> list:
    print("\n" + "=" * 55)
    print("Load Documents")
    print("=" * 55)
    print("Options:")
    print("  1 - Load specific file(s) from data folder")
    print("  2 - Load entire data folder")
    print("=" * 55)

    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice in ["1", "2"]:
            break
        print("Invalid. Enter 1 or 2.")

    documents = []

    if choice == "1":
        if os.path.exists(DATA_DIR):
            available = [
                f for f in os.listdir(DATA_DIR)
                if os.path.isfile(os.path.join(DATA_DIR, f))
            ]
            print(f"\nAvailable files in data folder:")
            for f in available:
                print(f"  - {f}")

        print("\nEnter filenames one by one.")
        print("Press Enter with empty input when done.")

        file_paths = []
        while True:
            filename = input("Filename: ").strip()
            if not filename:
                break
            full_path = os.path.join(DATA_DIR, filename)
            if os.path.exists(full_path):
                file_paths.append(full_path)
                print(f"  Added: {filename}")
            else:
                print(f"  Not found: {filename}")

        if not file_paths:
            print("No files selected.")
            return []

        print(f"\nLoading {len(file_paths)} file(s)...")
        for path in file_paths:
            docs = SimpleDirectoryReader(
                input_files=[path]
            ).load_data()
            for doc in docs:
                doc.metadata["source_file"] = os.path.basename(path)
                doc.metadata["category"] = os.path.splitext(
                    os.path.basename(path)
                )[0].lower().replace(" ", "_")
                doc.metadata["indexed_date"] = str(datetime.date.today())
                doc.excluded_llm_metadata_keys = [
                    "file_path", "creation_date"
                ]
            documents.extend(docs)
            print(f"  Loaded: {os.path.basename(path)} ({len(docs)} pages)")

    elif choice == "2":
        print(f"\nLoading all files from: {DATA_DIR}")
        all_docs = SimpleDirectoryReader(DATA_DIR, recursive=True).load_data()
        for doc in all_docs:
            doc.metadata["source_file"] = doc.metadata.get("file_name", "unknown")
            doc.metadata["category"] = os.path.splitext(
                doc.metadata.get("file_name", "unknown")
            )[0].lower().replace(" ", "_")
            doc.metadata["indexed_date"] = str(datetime.date.today())
            doc.excluded_llm_metadata_keys = ["file_path", "creation_date"]
        documents = all_docs
        print(f"Loaded {len(documents)} total pages from all files")

    print(f"\nTotal documents loaded: {len(documents)}")
    sources = {}
    for doc in documents:
        src = doc.metadata.get("source_file", "unknown")
        sources[src] = sources.get(src, 0) + 1
    print("Summary:")
    for src, count in sources.items():
        print(f"  {src} → {count} page(s)")

    return documents


# -------------------------------------------------------
# DEMO 1 — Basic Persistence (disk)
# -------------------------------------------------------
def demo_basic_persistence(documents):
    print("\n" + "=" * 55)
    print("DEMO 1 — Basic Persistence")
    print("Save index to disk — load instantly next run")
    print("=" * 55)

    if os.path.exists(PERSIST_DIR):
        print("Existing index found — loading from disk...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        print("Loaded instantly — no re-embedding needed")
    else:
        if not documents:
            print("No documents loaded. Load documents first.")
            return None

        print("No index found — building fresh...")
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        nodes = splitter.get_nodes_from_documents(documents)
        nodes = [n for n in nodes if len(n.text.strip()) > 20]
        print(f"Total nodes created: {len(nodes)}")

        index = VectorStoreIndex(nodes, show_progress=True)
        index.storage_context.persist(PERSIST_DIR)

        print(f"\nIndex saved to {PERSIST_DIR}")
        print("Files saved:")
        for f in os.listdir(PERSIST_DIR):
            size = os.path.getsize(os.path.join(PERSIST_DIR, f))
            print(f"  {f} ({size} bytes)")

    question = input("\nEnter a test question: ").strip()
    if question:
        query_engine = index.as_query_engine(similarity_top_k=3)
        response = query_engine.query(question)
        print(f"\nAnswer : {response}")
        for i, node in enumerate(response.source_nodes):
            score = f"{node.score:.4f}" if node.score else "N/A"
            print(f"\n  Source {i+1}:")
            print(f"    File  : {node.metadata.get('source_file')}")
            print(f"    Score : {score}")
            print(f"    Text  : {node.text[:150]}...")

    return index


# -------------------------------------------------------
# DEMO 2 — Qdrant Storage
#
# FIX: Previously called VectorStoreIndex.from_documents() every time,
# which re-embedded and re-uploaded all 178 nodes even when the
# collection already existed — causing duplicate vectors.
#
# NOW: Check the actual point count in Qdrant first.
#   - If points exist → skip ingestion, just load the index
#   - If empty        → embed and upload fresh
# This makes Demo 2 safe to run multiple times without duplicates.
# -------------------------------------------------------
def demo_qdrant_storage(documents):
    print("\n" + "=" * 55)
    print("DEMO 2 — Qdrant as Vector Store")
    print("Vectors persist in Qdrant across all runs")
    print("=" * 55)

    try:
        client = QdrantClient(url=QDRANT_URL)
        existing = [c.name for c in client.get_collections().collections]

        # Create collection if it doesn't exist
        if COLLECTION_NAME not in existing:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE
                )
            )
            print(f"Created Qdrant collection: {COLLECTION_NAME}")
        else:
            print(f"Qdrant collection exists: {COLLECTION_NAME}")

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME
        )
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )

        # FIX: Check how many points are already stored
        collection_info = client.get_collection(COLLECTION_NAME)
        point_count = collection_info.points_count

        if point_count > 0:
            # Collection already has vectors — just load the index, skip ingestion
            print(f"Collection already has {point_count} vectors — "
                  f"skipping re-embedding.")
            print("Loading index from Qdrant...")
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store
            )
        else:
            # Empty collection — embed and upload
            if not documents:
                print("No documents loaded and Qdrant is empty.")
                print("Load documents first (option 1 in main menu).")
                return None

            print(f"Empty collection — ingesting {len(documents)} documents...")
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                show_progress=True
            )
            print("Vectors stored in Qdrant successfully")

            # Show how many points were stored
            updated_info = client.get_collection(COLLECTION_NAME)
            print(f"Total vectors in Qdrant: {updated_info.points_count}")

        question = input("\nEnter a test question: ").strip()
        if question:
            query_engine = index.as_query_engine(similarity_top_k=3)
            response = query_engine.query(question)
            print(f"\nAnswer : {response}")
            for i, node in enumerate(response.source_nodes):
                score = f"{node.score:.4f}" if node.score else "N/A"
                print(f"\n  Source {i+1}:")
                print(f"    File  : {node.metadata.get('source_file')}")
                print(f"    Score : {score}")
                print(f"    Text  : {node.text[:150]}...")

        return index

    except Exception as e:
        print(f"Qdrant not available: {e}")
        print("Start Qdrant: docker run -p 6333:6333 qdrant/qdrant")
        return None


# -------------------------------------------------------
# DEMO 2 helper — Force re-index Qdrant
# Deletes the collection and re-ingests from scratch
# Useful when you want to update with new documents
# -------------------------------------------------------
def demo_qdrant_reindex(documents):
    print("\n" + "=" * 55)
    print("DEMO 2b — Force Re-index Qdrant")
    print("Deletes existing collection and re-ingests")
    print("=" * 55)

    if not documents:
        print("No documents loaded. Load documents first.")
        return None

    confirm = input(
        f"This will DELETE collection '{COLLECTION_NAME}' and re-ingest. "
        f"Confirm? (yes/no): "
    ).strip().lower()
    if confirm != "yes":
        print("Cancelled.")
        return None

    try:
        client = QdrantClient(url=QDRANT_URL)
        existing = [c.name for c in client.get_collections().collections]

        if COLLECTION_NAME in existing:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted collection: {COLLECTION_NAME}")

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE
            )
        )
        print(f"Created fresh collection: {COLLECTION_NAME}")

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME
        )
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        updated_info = client.get_collection(COLLECTION_NAME)
        print(f"Re-index complete. Vectors in Qdrant: {updated_info.points_count}")
        return index

    except Exception as e:
        print(f"Re-index failed: {e}")
        return None


# -------------------------------------------------------
# DEMO 3 — Explicit Retriever (no LLM)
# -------------------------------------------------------
def demo_explicit_retriever(index):
    print("\n" + "=" * 55)
    print("DEMO 3 — VectorIndexRetriever")
    print("Retrieve nodes WITHOUT calling LLM")
    print("Great for debugging what your pipeline retrieves")
    print("=" * 55)

    if not index:
        print("No index available. Run Demo 1 or 2 first.")
        return

    k = input("Enter Top-K value (default 3): ").strip()
    k = int(k) if k.isdigit() else 3

    retriever = VectorIndexRetriever(index=index, similarity_top_k=k)

    query = input("Enter query: ").strip()
    if not query:
        return

    print(f"\nRetrieving top {k} nodes (no LLM called)...")
    nodes = retriever.retrieve(query)

    # Deduplicate by node_id in case old duplicate vectors exist in Qdrant
    seen = set()
    unique_nodes = []
    for node in nodes:
        if node.node_id not in seen:
            seen.add(node.node_id)
            unique_nodes.append(node)

    print(f"\nRetrieved {len(unique_nodes)} unique nodes "
          f"(raw: {len(nodes)}):")
    for i, node in enumerate(unique_nodes):
        score = f"{node.score:.4f}" if node.score else "N/A"
        print(f"\n  Node {i+1}:")
        print(f"    Score    : {score}")
        print(f"    File     : {node.metadata.get('source_file', 'unknown')}")
        print(f"    Category : {node.metadata.get('category', 'unknown')}")
        print(f"    Date     : {node.metadata.get('indexed_date', 'unknown')}")
        print(f"    Preview  : {node.text[:200]}...")


# -------------------------------------------------------
# DEMO 4 — Top-K Comparison
# -------------------------------------------------------
def demo_topk_comparison(index):
    print("\n" + "=" * 55)
    print("DEMO 4 — Top-K Comparison")
    print("See how different Top-K values affect answers")
    print("=" * 55)

    if not index:
        print("No index available. Run Demo 1 or 2 first.")
        return

    query = input("Enter query to compare: ").strip()
    if not query:
        return

    for k in [1, 2, 3, 5]:
        print(f"\n--- similarity_top_k = {k} ---")
        query_engine = index.as_query_engine(similarity_top_k=k)
        response = query_engine.query(query)
        print(f"Nodes used : {len(response.source_nodes)}")
        print(f"Answer     : {str(response)[:300]}...")

        # Show unique sources only
        seen = set()
        for node in response.source_nodes:
            src = node.metadata.get('source_file', 'unknown')
            score = f"{node.score:.4f}" if node.score else "N/A"
            marker = " (duplicate)" if node.node_id in seen else ""
            print(f"  Source: {src} | Score: {score}{marker}")
            seen.add(node.node_id)


# -------------------------------------------------------
# DEMO 5 — Metadata Filtering
# -------------------------------------------------------
def demo_metadata_filtering(index, documents):
    print("\n" + "=" * 55)
    print("DEMO 5 — Metadata Filtering")
    print("Filter nodes by metadata before similarity search")
    print("=" * 55)

    if not index:
        print("No index available. Run Demo 1 or 2 first.")
        return

    print("\nAvailable source files in index:")
    sources = set()
    categories = set()
    for doc in documents:
        sources.add(doc.metadata.get("source_file", "unknown"))
        categories.add(doc.metadata.get("category", "unknown"))

    for src in sources:
        print(f"  source_file = {src}")
    print("\nAvailable categories:")
    for cat in categories:
        print(f"  category = {cat}")

    query = input("\nEnter query: ").strip()
    if not query:
        return

    # No filter
    print("\n[1] No filter — searching all documents:")
    qe = index.as_query_engine(similarity_top_k=3)
    response = qe.query(query)
    print(f"Answer  : {str(response)[:300]}...")
    print(f"Sources : {[n.metadata.get('source_file') for n in response.source_nodes]}")

    # Filter by source file
    if sources:
        source_file = input("\nEnter source_file to filter by: ").strip()
        if source_file:
            print(f"\n[2] Filter by source_file = {source_file}:")
            filters = MetadataFilters(
                filters=[MetadataFilter(key="source_file", value=source_file)]
            )
            qe = index.as_query_engine(similarity_top_k=3, filters=filters)
            response = qe.query(query)
            print(f"Answer  : {str(response)[:300]}...")
            print(f"Sources : {[n.metadata.get('source_file') for n in response.source_nodes]}")

    # Filter by category
    if categories:
        category = input("\nEnter category to filter by: ").strip()
        if category:
            print(f"\n[3] Filter by category = {category}:")
            filters = MetadataFilters(
                filters=[MetadataFilter(key="category", value=category)]
            )
            qe = index.as_query_engine(similarity_top_k=3, filters=filters)
            response = qe.query(query)
            print(f"Answer  : {str(response)[:300]}...")
            print(f"Sources : {[n.metadata.get('source_file') for n in response.source_nodes]}")

    # AND filter
    if len(sources) >= 1 and len(categories) >= 1:
        print("\n[4] AND filter — source_file AND category:")
        src = input("  source_file value: ").strip()
        cat = input("  category value: ").strip()
        if src and cat:
            filters = MetadataFilters(
                filters=[
                    MetadataFilter(key="source_file", value=src),
                    MetadataFilter(key="category", value=cat)
                ],
                condition=FilterCondition.AND
            )
            qe = index.as_query_engine(similarity_top_k=3, filters=filters)
            response = qe.query(query)
            print(f"Answer  : {str(response)[:300]}...")
            print(f"Sources : {[n.metadata.get('source_file') for n in response.source_nodes]}")


# -------------------------------------------------------
# DEMO 6 — Interactive Query
# -------------------------------------------------------
def demo_interactive_query(index, documents):
    print("\n" + "=" * 55)
    print("DEMO 6 — Interactive Query with Filters")
    print("Type 'exit' to go back")
    print("=" * 55)

    if not index:
        print("No index available. Run Demo 1 or 2 first.")
        return

    sources = list(set(
        doc.metadata.get("source_file", "unknown")
        for doc in documents
    ))
    print("\nIndexed files:")
    for i, src in enumerate(sources):
        print(f"  {i+1}. {src}")

    print("\nFilter options:")
    print("  1 - No filter (search all)")
    print("  2 - Filter by specific file")
    print("  3 - Filter by category")

    while True:
        opt = input("\nSelect filter (1/2/3): ").strip()
        if opt not in ["1", "2", "3"]:
            print("Invalid.")
            continue
        break

    filters = None

    if opt == "2":
        filename = input("Enter filename: ").strip()
        filters = MetadataFilters(
            filters=[MetadataFilter(key="source_file", value=filename)]
        )
        print(f"Filtering to: {filename}")

    elif opt == "3":
        category = input("Enter category: ").strip()
        filters = MetadataFilters(
            filters=[MetadataFilter(key="category", value=category)]
        )
        print(f"Filtering to category: {category}")

    query_engine = index.as_query_engine(
        similarity_top_k=3,
        filters=filters
    )

    print("\nAsk questions. Type 'exit' to stop.")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue

        response = query_engine.query(question)
        print(f"\nAnswer : {response}")
        print(f"\nSources ({len(response.source_nodes)}):")
        for i, node in enumerate(response.source_nodes):
            score = f"{node.score:.4f}" if node.score else "N/A"
            print(f"  {i+1}. {node.metadata.get('source_file')} | Score: {score}")
            print(f"     {node.text[:150]}...")


# -------------------------------------------------------
# Main Menu
# -------------------------------------------------------
def main_menu():
    documents = []
    index = None

    while True:
        print("\n" + "=" * 55)
        print("Day 5 — Storage & Persistence + Retrieval")
        print("=" * 55)
        print("  1 - Load documents")
        print("  2 - Demo 1: Basic Persistence (disk)")
        print("  3 - Demo 2: Qdrant Vector Store")
        print("  4 - Demo 2b: Force Re-index Qdrant (delete + re-ingest)")
        print("  5 - Demo 3: Explicit Retriever (no LLM)")
        print("  6 - Demo 4: Top-K Comparison")
        print("  7 - Demo 5: Metadata Filtering")
        print("  8 - Demo 6: Interactive Query")
        print("  9 - Show loaded documents")
        print("  0 - Exit")
        print("=" * 55)

        choice = input("Enter choice: ").strip()

        if choice == "1":
            documents = load_documents_from_user()
        elif choice == "2":
            index = demo_basic_persistence(documents)
        elif choice == "3":
            index = demo_qdrant_storage(documents)
        elif choice == "4":
            index = demo_qdrant_reindex(documents)
        elif choice == "5":
            demo_explicit_retriever(index)
        elif choice == "6":
            demo_topk_comparison(index)
        elif choice == "7":
            demo_metadata_filtering(index, documents)
        elif choice == "8":
            demo_interactive_query(index, documents)
        elif choice == "9":
            if not documents:
                print("No documents loaded yet.")
            else:
                print(f"\nLoaded documents ({len(documents)}):")
                sources = {}
                for doc in documents:
                    src = doc.metadata.get("source_file", "unknown")
                    sources[src] = sources.get(src, 0) + 1
                for src, count in sources.items():
                    print(f"  {src} → {count} page(s)")
        elif choice == "0":
            print("\nExiting Day 5. Goodbye!")
            break
        else:
            print("Invalid. Enter 0-9.")


def main():
    use_local = show_menu()
    configure(use_local=use_local)
    main_menu()


if __name__ == "__main__":
    main()