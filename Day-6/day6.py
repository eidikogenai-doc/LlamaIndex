# day6_queryengine.py
# Day 6 — Query Engine + Response Synthesis
# FIXES:
#   - Own storage folder (storage_day6/) — no collision with Day 5
#   - Rebuild properly deletes old index first
#   - All demos: exit/back/quit keywords checked BEFORE calling LLM
#   - Streaming demo rejects single-char accidental inputs (menu keys)
#   - Demo 8: Per-document query for proper multi-doc retrieval

import os
import sys
import shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    PromptTemplate,
    ChatPromptTemplate,
    Settings
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from settings import configure

DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data")
)

# FIX: Day 6 gets its own storage — no collision with Day 5
PERSIST_DIR = "storage_day6/"

# Shared exit keywords checked in every demo loop
EXIT_WORDS = ("back", "exit", "quit", "q")


# -------------------------------------------------------
# Menu
# -------------------------------------------------------
def show_menu():
    print("\n" + "=" * 55)
    print("       LlamaIndex - Day 6")
    print("       Query Engine + Response Synthesis")
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
# Load Documents
# -------------------------------------------------------
def load_documents():
    print("\n" + "=" * 55)
    print("Load Documents")
    print("=" * 55)
    print("Options:")
    print("  1 - Load specific file(s)")
    print("  2 - Load entire data folder")
    print("=" * 55)

    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice in ["1", "2"]:
            break
        print("Invalid.")

    documents = []

    if choice == "1":
        available = [
            f for f in os.listdir(DATA_DIR)
            if os.path.isfile(os.path.join(DATA_DIR, f))
        ]
        print("\nAvailable files:")
        for f in available:
            print(f"  - {f}")

        print("\nEnter filenames. Empty line to finish.")
        while True:
            filename = input("Filename: ").strip()
            if not filename:
                break
            full_path = os.path.join(DATA_DIR, filename)
            if os.path.exists(full_path):
                docs = SimpleDirectoryReader(
                    input_files=[full_path]
                ).load_data()
                for doc in docs:
                    doc.metadata["source_file"] = filename
                    doc.metadata["category"] = os.path.splitext(
                        filename
                    )[0].lower()
                documents.extend(docs)
                print(f"  Loaded: {filename} ({len(docs)} pages)")
            else:
                print(f"  Not found: {filename}")

    elif choice == "2":
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        for doc in documents:
            fname = doc.metadata.get("file_name", "unknown")
            doc.metadata["source_file"] = fname
            doc.metadata["category"] = os.path.splitext(fname)[0].lower()
        print(f"Loaded {len(documents)} pages from all files")

    print(f"\nTotal pages loaded: {len(documents)}")
    sources = {}
    for doc in documents:
        src = doc.metadata.get("source_file", "unknown")
        sources[src] = sources.get(src, 0) + 1
    for src, count in sources.items():
        print(f"  {src} → {count} page(s)")

    return documents


# -------------------------------------------------------
# Build or Load Index
# FIX: Rebuild properly deletes old index first
# -------------------------------------------------------
def build_or_load_index(documents):
    if os.path.exists(PERSIST_DIR):
        print(f"\nExisting Day 6 index found at {PERSIST_DIR}")
        print("  1 - Load existing index")
        print("  2 - Rebuild fresh index (deletes old)")
        choice = input("Enter choice (1 or 2): ").strip()

        if choice == "1":
            storage_context = StorageContext.from_defaults(
                persist_dir=PERSIST_DIR
            )
            index = load_index_from_storage(storage_context)
            print("Index loaded from disk")
            return index

        elif choice == "2":
            print(f"Deleting old index at {PERSIST_DIR}...")
            shutil.rmtree(PERSIST_DIR)
            print("Old index deleted.")

    if not documents:
        print("No documents loaded. Load documents first.")
        return None

    print("\nBuilding fresh index...")
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)
    nodes = [n for n in nodes if len(n.text.strip()) > 20]
    print(f"Nodes created: {len(nodes)}")

    sources = {}
    for node in nodes:
        src = node.metadata.get("source_file", "unknown")
        sources[src] = sources.get(src, 0) + 1
    print("Node distribution:")
    for src, count in sources.items():
        print(f"  {src} → {count} nodes")

    index = VectorStoreIndex(nodes, show_progress=True)
    index.storage_context.persist(PERSIST_DIR)
    print(f"Index saved to {PERSIST_DIR}")
    return index


# -------------------------------------------------------
# Helper — print response with source nodes
# -------------------------------------------------------
def print_response(response, show_sources=True):
    print(f"\nAnswer : {response}")
    if show_sources and response.source_nodes:
        print(f"\nSource nodes ({len(response.source_nodes)}):")
        for i, node in enumerate(response.source_nodes):
            score = f"{node.score:.4f}" if node.score else "N/A"
            print(f"\n  Node {i+1}:")
            print(f"    Score  : {score}")
            print(f"    File   : {node.metadata.get('source_file', 'unknown')}")
            print(f"    Page   : {node.metadata.get('page_label', 'unknown')}")
            print(f"    Text   : {node.text[:200]}...")


# -------------------------------------------------------
# Helper — shared question input loop
# Returns None when user wants to exit
# -------------------------------------------------------
def get_question(prompt="\nQuestion: "):
    question = input(prompt).strip()
    if question.lower() in EXIT_WORDS:
        return None
    return question


# -------------------------------------------------------
# DEMO 1 — Basic QueryEngine
# FIX: exit/back/quit all return to menu before LLM call
# -------------------------------------------------------
def demo_basic_query_engine(index):
    print("\n" + "=" * 55)
    print("DEMO 1 — Basic QueryEngine")
    print("Simplest way to query your index")
    print("=" * 55)

    if not index:
        print("No index. Build index first.")
        return

    query_engine = index.as_query_engine(similarity_top_k=3)

    print("\nType 'back' to return to menu.")
    while True:
        question = get_question()
        if question is None:
            break
        if not question:
            continue
        response = query_engine.query(question)
        print_response(response)


# -------------------------------------------------------
# DEMO 2 — RetrieverQueryEngine (explicit)
# FIX: exit/back/quit all return to menu before LLM call
# -------------------------------------------------------
def demo_retriever_query_engine(index):
    print("\n" + "=" * 55)
    print("DEMO 2 — RetrieverQueryEngine")
    print("Explicit control over retriever and synthesizer")
    print("=" * 55)

    if not index:
        print("No index. Build index first.")
        return

    k = input("Enter Top-K value (default 3): ").strip()
    k = int(k) if k.isdigit() else 3

    mode = input(
        "Enter response mode "
        "(compact/refine/tree_summarize, default compact): "
    ).strip() or "compact"

    retriever = VectorIndexRetriever(index=index, similarity_top_k=k)
    synthesizer = get_response_synthesizer(response_mode=mode)
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=synthesizer
    )

    print(f"\nQueryEngine built:")
    print(f"  Top-K          : {k}")
    print(f"  Response mode  : {mode}")
    print("\nType 'back' to return.")

    while True:
        question = get_question()
        if question is None:
            break
        if not question:
            continue
        response = query_engine.query(question)
        print_response(response)


# -------------------------------------------------------
# DEMO 3 — Response Mode Comparison
# -------------------------------------------------------
def demo_response_modes(index):
    print("\n" + "=" * 55)
    print("DEMO 3 — Response Mode Comparison")
    print("Same question, different synthesis modes")
    print("=" * 55)

    if not index:
        print("No index. Build index first.")
        return

    question = input("Enter question to compare: ").strip()
    if not question or question.lower() in EXIT_WORDS:
        return

    modes = [
        "compact",
        "refine",
        "tree_summarize",
        "simple_summarize",
        "accumulate",
        "no_text"
    ]

    for mode in modes:
        print(f"\n{'='*40}")
        print(f"Mode: {mode}")
        print(f"{'='*40}")
        try:
            query_engine = index.as_query_engine(
                similarity_top_k=3,
                response_mode=mode
            )
            response = query_engine.query(question)
            if mode == "no_text":
                print(f"Answer : [no LLM called]")
                print(f"Nodes retrieved: {len(response.source_nodes)}")
                for i, node in enumerate(response.source_nodes):
                    score = f"{node.score:.4f}" if node.score else "N/A"
                    print(
                        f"  Node {i+1} | Score: {score} "
                        f"| {node.text[:100]}..."
                    )
            else:
                print(f"Answer : {str(response)[:400]}...")
        except Exception as e:
            print(f"Error with mode {mode}: {e}")


# -------------------------------------------------------
# DEMO 4 — Custom PromptTemplate
# FIX: exit keywords checked BEFORE sending to LLM
# -------------------------------------------------------
def demo_custom_prompt(index):
    print("\n" + "=" * 55)
    print("DEMO 4 — Custom PromptTemplate")
    print("Control exactly what gets sent to the LLM")
    print("=" * 55)

    if not index:
        print("No index. Build index first.")
        return

    custom_qa_prompt = PromptTemplate(
        "You are a helpful document assistant.\n\n"
        "The following context is extracted from the document:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n\n"
        "Using ONLY the context above, answer the question below.\n"
        "If the answer is not in the context, respond with: "
        "'This information is not available in the document.'\n"
        "Be concise, accurate, and professional.\n\n"
        "Question: {query_str}\n"
        "Answer: "
    )

    default_engine = index.as_query_engine(similarity_top_k=3)
    custom_engine = index.as_query_engine(
        similarity_top_k=3,
        text_qa_template=custom_qa_prompt
    )

    print("\nType 'back' to return.")
    while True:
        # FIX: Exit check before ANYTHING else
        question = get_question()
        if question is None:
            break
        if not question:
            continue

        print("\n--- Default prompt ---")
        response1 = default_engine.query(question)
        print(f"Answer: {str(response1)[:300]}...")

        print("\n--- Custom prompt ---")
        response2 = custom_engine.query(question)
        print(f"Answer: {str(response2)[:300]}...")


# -------------------------------------------------------
# DEMO 5 — ChatPromptTemplate
# FIX: exit keywords checked before LLM call
# -------------------------------------------------------
def demo_chat_prompt(index):
    print("\n" + "=" * 55)
    print("DEMO 5 — ChatPromptTemplate")
    print("System + User message structure for chat models")
    print("=" * 55)

    if not index:
        print("No index. Build index first.")
        return

    chat_prompt = ChatPromptTemplate(
        message_templates=[
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=(
                    "You are an expert document analyst. "
                    "Answer questions strictly based on the provided "
                    "document context only. Do not use outside knowledge. "
                    "If the answer is not in the context, clearly state that."
                )
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=(
                    "Document context:\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n\n"
                    "Question: {query_str}\n"
                    "Answer: "
                )
            )
        ]
    )

    query_engine = index.as_query_engine(
        similarity_top_k=3,
        text_qa_template=chat_prompt
    )

    print("\nType 'back' to return.")
    while True:
        question = get_question()
        if question is None:
            break
        if not question:
            continue
        response = query_engine.query(question)
        print_response(response)


# -------------------------------------------------------
# DEMO 6 — Streaming Response
# FIX: Exit check before LLM call
# FIX: Reject single-char inputs (accidental menu keys like 9, A)
# -------------------------------------------------------
def demo_streaming(index):
    print("\n" + "=" * 55)
    print("DEMO 6 — Streaming Response")
    print("Tokens appear as they are generated")
    print("=" * 55)

    if not index:
        print("No index. Build index first.")
        return

    query_engine = index.as_query_engine(
        similarity_top_k=3,
        streaming=True
    )

    print("\nType 'back' to return to menu.")
    while True:
        question = get_question()

        # FIX: Exit check before anything else
        if question is None:
            break

        # FIX: Reject empty or single-character inputs
        # (catches accidental menu keypresses like 9, A, 0)
        if len(question) <= 1:
            print("  Please enter a proper question (more than 1 character).")
            continue

        print("\nAnswer (streaming): ", end="", flush=True)
        streaming_response = query_engine.query(question)
        for token in streaming_response.response_gen:
            print(token, end="", flush=True)
        print()

        print(f"\nSource nodes ({len(streaming_response.source_nodes)}):")
        for i, node in enumerate(streaming_response.source_nodes):
            score = f"{node.score:.4f}" if node.score else "N/A"
            print(
                f"  {i+1}. {node.metadata.get('source_file')} "
                f"| Score: {score}"
            )
            print(f"     {node.text[:150]}...")


# -------------------------------------------------------
# DEMO 7 — Source Node Inspection
# FIX: exit keywords checked before LLM call
# -------------------------------------------------------
def demo_source_inspection(index):
    print("\n" + "=" * 55)
    print("DEMO 7 — Source Node Inspection")
    print("Deep dive into what the LLM received")
    print("=" * 55)

    if not index:
        print("No index. Build index first.")
        return

    k = input("Enter Top-K (default 3): ").strip()
    k = int(k) if k.isdigit() else 3

    query_engine = index.as_query_engine(similarity_top_k=k)

    print("\nType 'back' to return.")
    while True:
        question = get_question()
        if question is None:
            break
        if not question:
            continue

        response = query_engine.query(question)

        print(f"\nFinal Answer:\n{response}")
        print(f"\n{'='*40}")
        print(f"Deep dive into source nodes ({len(response.source_nodes)}):")
        print(f"{'='*40}")

        for i, node in enumerate(response.source_nodes):
            score = f"{node.score:.4f}" if node.score else "N/A"
            print(f"\nNode {i+1}:")
            print(f"  Relevance Score : {score}")
            print(f"  Source File     : "
                  f"{node.metadata.get('source_file', 'unknown')}")
            print(f"  Page            : "
                  f"{node.metadata.get('page_label', 'unknown')}")
            print(f"  Category        : "
                  f"{node.metadata.get('category', 'unknown')}")
            print(f"  Node ID         : {node.node_id}")
            print(f"  Text length     : {len(node.text)} characters")
            print(f"  Full text:\n")
            print(f"    {node.text}")
            print(f"  {'─'*40}")


# -------------------------------------------------------
# DEMO 8 — Per-Document Query
#
# WHY THIS IS NEEDED:
#   When 2 documents have very different topics (Raspberry Pi vs Telangana),
#   a single similarity search always favours the larger document because
#   it has more nodes (171 vs 2). The smaller doc's nodes rarely make
#   it into top-K for unrelated queries.
#
# HOW IT WORKS:
#   Runs the same query separately against each source file using
#   MetadataFilters, then shows all answers together.
#   Every document is guaranteed a response.
#
# FIX: exit keywords checked before LLM call
# -------------------------------------------------------
def demo_per_document_query(index, documents):
    print("\n" + "=" * 55)
    print("DEMO 8 — Per-Document Query")
    print("Queries each document separately — fixes multi-doc retrieval")
    print("=" * 55)

    if not index:
        print("No index. Build index first.")
        return

    sources = list(set(
        doc.metadata.get("source_file", "unknown")
        for doc in documents
    ))

    if not sources:
        print("No source metadata found. Rebuild index with documents loaded.")
        return

    print("\nWill query these documents separately:")
    for i, src in enumerate(sources):
        print(f"  {i+1}. {src}")

    print("\nType 'back' to return.")
    while True:
        question = get_question()
        if question is None:
            break
        if not question:
            continue

        print(f"\nQuerying each document for: '{question}'")
        print("=" * 55)

        for src in sources:
            print(f"\n--- {src} ---")
            filters = MetadataFilters(
                filters=[MetadataFilter(key="source_file", value=src)]
            )
            qe = index.as_query_engine(
                similarity_top_k=2,
                filters=filters
            )
            try:
                response = qe.query(question)
                print(f"Answer : {str(response)[:300]}")
                if response.source_nodes:
                    best = response.source_nodes[0]
                    score = f"{best.score:.4f}" if best.score else "N/A"
                    print(f"Best node score : {score}")
                    print(f"Preview : {best.text[:150]}...")
            except Exception as e:
                print(f"Query failed for {src}: {e}")


# -------------------------------------------------------
# Main Menu
# -------------------------------------------------------
def main_menu(documents, index):
    while True:
        print("\n" + "=" * 55)
        print("Day 6 — Query Engine + Response Synthesis")
        print("=" * 55)
        print("  1 - Load documents")
        print("  2 - Build / Load index")
        print("  3 - Demo 1: Basic QueryEngine")
        print("  4 - Demo 2: RetrieverQueryEngine (explicit)")
        print("  5 - Demo 3: Response Mode Comparison")
        print("  6 - Demo 4: Custom PromptTemplate")
        print("  7 - Demo 5: ChatPromptTemplate")
        print("  8 - Demo 6: Streaming Response")
        print("  9 - Demo 7: Source Node Inspection")
        print("  A - Demo 8: Per-Document Query (multi-doc fix)")
        print("  0 - Exit")
        print("=" * 55)

        choice = input("Enter choice: ").strip().upper()

        if choice == "1":
            documents = load_documents()
        elif choice == "2":
            index = build_or_load_index(documents)
        elif choice == "3":
            demo_basic_query_engine(index)
        elif choice == "4":
            demo_retriever_query_engine(index)
        elif choice == "5":
            demo_response_modes(index)
        elif choice == "6":
            demo_custom_prompt(index)
        elif choice == "7":
            demo_chat_prompt(index)
        elif choice == "8":
            demo_streaming(index)
        elif choice == "9":
            demo_source_inspection(index)
        elif choice == "A":
            demo_per_document_query(index, documents)
        elif choice == "0":
            print("\nExiting Day 6. Goodbye!")
            break
        else:
            print("Invalid. Enter 0-9 or A.")

    return documents, index


def main():
    use_local = show_menu()
    configure(use_local=use_local)
    documents = []
    index = None
    main_menu(documents, index)


if __name__ == "__main__":
    main()