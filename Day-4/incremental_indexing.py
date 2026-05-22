# incremental_indexing.py
import os
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from settings import configure

# -------------------------------------------------------
# Config
# -------------------------------------------------------
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "my_documents"
EMBEDDING_DIM = 384
TRACKER_FILE = "indexed_files.json"  # tracks what is already indexed


# -------------------------------------------------------
# Tracker — keeps record of what is already indexed
# -------------------------------------------------------
def load_tracker() -> dict:
    """Load the tracker file. Returns empty dict if not exists."""
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, "r") as f:
            return json.load(f)
    return {}


def save_tracker(tracker: dict):
    """Save updated tracker to disk."""
    with open(TRACKER_FILE, "w") as f:
        json.dump(tracker, f, indent=2)
    print(f"Tracker saved: {len(tracker)} files tracked")


def is_already_indexed(filepath: str, tracker: dict) -> bool:
    """Check if a file is already in the index."""
    return filepath in tracker


def mark_as_indexed(filepath: str, doc_ids: list, tracker: dict):
    """Mark a file as indexed with its document IDs."""
    tracker[filepath] = {
        "doc_ids": doc_ids,
        "indexed_at": str(__import__("datetime").datetime.now())
    }


# -------------------------------------------------------
# Qdrant Setup
# -------------------------------------------------------
def setup_qdrant():
    client = QdrantClient(url=QDRANT_URL)

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE
            )
        )
        print(f"Created collection: {COLLECTION_NAME}")
    else:
        print(f"Collection exists: {COLLECTION_NAME}")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME
    )
    return vector_store


# -------------------------------------------------------
# Load existing index from Qdrant
# -------------------------------------------------------
def load_index(vector_store):
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context
    )
    return index


# -------------------------------------------------------
# Add new files to existing index
# -------------------------------------------------------
def add_documents(file_paths: list, index, tracker: dict):
    """
    Add only new files to the index.
    Skips files that are already indexed.
    """
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    new_files = 0
    skipped_files = 0

    for filepath in file_paths:
        # Normalize path
        filepath = os.path.abspath(filepath)

        # Check if already indexed
        if is_already_indexed(filepath, tracker):
            print(f"Already indexed — skipping: {os.path.basename(filepath)}")
            skipped_files += 1
            continue

        # Load the new file
        print(f"Indexing: {os.path.basename(filepath)}")
        try:
            documents = SimpleDirectoryReader(
                input_files=[filepath]
            ).load_data()

            # Add custom metadata
            for doc in documents:
                doc.metadata["source_file"] = os.path.basename(filepath)

            # Chunk into nodes
            nodes = splitter.get_nodes_from_documents(documents)

            # Validate nodes
            nodes = [n for n in nodes if len(n.text.strip()) > 20]

            # Insert into index
            for doc in documents:
                index.insert(doc)

            # Track doc_ids for future deletion if needed
            doc_ids = [doc.doc_id for doc in documents]
            mark_as_indexed(filepath, doc_ids, tracker)

            print(f"  Pages    : {len(documents)}")
            print(f"  Nodes    : {len(nodes)}")
            print(f"  Doc IDs  : {doc_ids}")
            new_files += 1

        except Exception as e:
            print(f"Failed to index {filepath}: {e}")

    print(f"\nSummary:")
    print(f"  New files indexed : {new_files}")
    print(f"  Skipped (already indexed) : {skipped_files}")
    print(f"  Total tracked files : {len(tracker)}")

    return index


# -------------------------------------------------------
# Remove a document from index
# -------------------------------------------------------
def remove_document(filepath: str, index, tracker: dict):
    """Remove a document from the index and tracker."""
    filepath = os.path.abspath(filepath)

    if not is_already_indexed(filepath, tracker):
        print(f"File not in index: {filepath}")
        return

    doc_ids = tracker[filepath]["doc_ids"]
    for doc_id in doc_ids:
        index.delete_ref_doc(doc_id, delete_from_docstore=True)
        print(f"Deleted doc_id: {doc_id}")

    del tracker[filepath]
    print(f"Removed from tracker: {os.path.basename(filepath)}")


# -------------------------------------------------------
# Query the index
# -------------------------------------------------------
def query_index(index, tracker: dict):
    print("\n" + "=" * 50)
    print("Query Your Index")
    print("Type 'exit' to go back to menu")
    print("=" * 50)

    # Ask if user wants to filter by document
    print("\nQuery options:")
    print("  1 - Query ALL documents")
    print("  2 - Query a SPECIFIC document")
    
    while True:
        opt = input("Enter option (1 or 2): ").strip()
        if opt in ["1", "2"]:
            break
        print("Invalid. Enter 1 or 2.")

    # Build filters if specific document chosen
    filters = None
    if opt == "2":
        print("\nAvailable documents:")
        tracked_files = list(tracker.keys())
        for i, f in enumerate(tracked_files):
            print(f"  {i+1}. {os.path.basename(f)}")
        
        filename = input("\nEnter filename to query: ").strip()
        
        from llama_index.core.vector_stores import (
            MetadataFilter,
            MetadataFilters
        )
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="source_file",
                    value=filename
                )
            ]
        )
        print(f"Filtering results to: {filename}")

    # Create query engine with or without filter
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        filters=filters
    )

    while True:
        question = input("\nEnter question: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            print("Please enter a question.")
            continue

        try:
            response = query_engine.query(question)
            print(f"\nAnswer : {response}")
            print(f"\nSource nodes used ({len(response.source_nodes)}):")
            for i, node in enumerate(response.source_nodes):
                print(f"\n  Source {i+1}:")
                print(f"    File  : {node.metadata.get('source_file', 'unknown')}")
                print(f"    Score : {node.score:.4f}")
                print(f"    Text  : {node.text[:150]}...")
        except Exception as e:
            print(f"Query failed: {e}")


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    print("\n" + "=" * 50)
    print("Incremental Indexing Pipeline")
    print("=" * 50)

    # Select LLM
    print("\nSelect LLM Provider:")
    print("  1 - Groq (cloud)")
    print("  2 - LM Studio (local)")
    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice == "1":
            configure(use_local=False)
            break
        elif choice == "2":
            configure(use_local=True)
            break

    # Load tracker
    tracker = load_tracker()
    print(f"\nCurrently tracked files: {len(tracker)}")
    for f in tracker:
        print(f"  - {os.path.basename(f)}")

    # Setup Qdrant and load index
    vector_store = setup_qdrant()
    index = load_index(vector_store)

    while True:
        print("\n" + "=" * 50)
        print("What do you want to do?")
        print("  1 - Add new documents")
        print("  2 - Remove a document")
        print("  3 - Query the index")
        print("  4 - Show tracked files")
        print("  5 - Exit")
        print("=" * 50)

        action = input("Enter choice: ").strip()

        if action == "1":
            print("\nEnter file paths (one per line, empty line to finish):")
            file_paths = []
            while True:
                path = input("File path: ").strip()
                if not path:
                    break
                if os.path.exists(path):
                    file_paths.append(path)
                else:
                    print(f"File not found: {path}")

            if file_paths:
                index = add_documents(file_paths, index, tracker)
                save_tracker(tracker)

        elif action == "2":
            print("\nEnter file path to remove:")
            path = input("File path: ").strip()
            remove_document(path, index, tracker)
            save_tracker(tracker)

        elif action == "3":
            query_index(index,tracker)

        elif action == "4":
            print(f"\nTracked files ({len(tracker)}):")
            for f, info in tracker.items():
                print(f"  - {os.path.basename(f)}")
                print(f"    Indexed at : {info['indexed_at']}")
                print(f"    Doc IDs    : {info['doc_ids']}")

        elif action == "5":
            print("Goodbye!")
            break

        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()