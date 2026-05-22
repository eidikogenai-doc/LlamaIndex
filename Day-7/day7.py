# day7_chatengine.py
# Day 7 — Chat Engine: Memory & Context

import os
import sys
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.node_parser import SentenceSplitter
from settings import configure

DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data")
)
PERSIST_DIR = "storage/"

# -------------------------------------------------------
# LLM Menu
# -------------------------------------------------------
def show_menu():
    print("\n" + "=" * 55)
    print("       LlamaIndex - Day 7")
    print("       Chat Engine: Memory & Context")
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

    available = [
        f for f in os.listdir(DATA_DIR)
        if os.path.isfile(os.path.join(DATA_DIR, f))
    ]
    print("Available files:")
    for f in available:
        print(f"  - {f}")

    print("\nOptions:")
    print("  1 - Load specific file(s)")
    print("  2 - Load entire data folder")

    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice in ["1", "2"]:
            break
        print("Invalid.")

    documents = []

    if choice == "1":
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
                    doc.metadata["category"] = (
                        os.path.splitext(filename)[0].lower()
                    )
                documents.extend(docs)
                print(f"  Loaded: {filename} ({len(docs)} pages)")
            else:
                print(f"  Not found: {filename}")

    elif choice == "2":
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        for doc in documents:
            doc.metadata["source_file"] = doc.metadata.get(
                "file_name", "unknown"
            )
        print(f"Loaded {len(documents)} pages")

    print(f"\nTotal pages loaded: {len(documents)}")
    return documents

# -------------------------------------------------------
# Build or Load Index
# -------------------------------------------------------
def build_or_load_index(documents):
    if os.path.exists(PERSIST_DIR):
        print("\nExisting index found.")
        print("  1 - Load existing")
        print("  2 - Rebuild fresh")
        choice = input("Enter choice (1 or 2): ").strip()

        if choice == "1":
            storage_context = StorageContext.from_defaults(
                persist_dir=PERSIST_DIR
            )
            index = load_index_from_storage(storage_context)
            print("Index loaded from disk")
            return index

    if not documents:
        print("No documents. Load documents first.")
        return None

    print("\nBuilding index...")
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)
    nodes = [n for n in nodes if len(n.text.strip()) > 20]
    print(f"Nodes created: {len(nodes)}")

    index = VectorStoreIndex(nodes, show_progress=True)
    index.storage_context.persist(PERSIST_DIR)
    print(f"Index saved to {PERSIST_DIR}")
    return index


# -------------------------------------------------------
# Helper — print chat response with sources
# -------------------------------------------------------
def print_chat_response(response, show_sources=True):
    print(f"\nAssistant: {response.response}")
    if show_sources and hasattr(response, "source_nodes") \
            and response.source_nodes:
        print(f"\nSources ({len(response.source_nodes)}):")
        for i, node in enumerate(response.source_nodes):
            score = f"{node.score:.4f}" if node.score else "N/A"
            print(f"  {i+1}. {node.metadata.get('source_file')} "
                  f"| Score: {score}")
            print(f"     {node.text[:120]}...")


# -------------------------------------------------------
# DEMO 1 — condense_question mode
# -------------------------------------------------------
def demo_condense_question(index):
    print("\n" + "=" * 55)
    print("DEMO 1 — condense_question mode")
    print("Condenses history + question into one query")
    print("=" * 55)

    if not index:
        print("No index. Build index first.")
        return

    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

    chat_engine = index.as_chat_engine(
        chat_mode="condense_question",
        memory=memory,
        verbose=True    # shows condensed question
    )

    print("\nChat started. Type 'reset' to clear history.")
    print("Type 'back' to return to menu.")
    print("Try asking a follow-up question using 'it' or 'its'.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "back":
            break
        if user_input.lower() == "reset":
            chat_engine.reset()
            print("Chat history cleared.\n")
            continue
        if not user_input:
            continue

        response = chat_engine.chat(user_input)
        print_chat_response(response)


# -------------------------------------------------------
# DEMO 2 — context mode
# -------------------------------------------------------
def demo_context_mode(index):
    print("\n" + "=" * 55)
    print("DEMO 2 — context mode")
    print("Retrieves fresh context for every message")
    print("=" * 55)

    if not index:
        print("No index. Build index first.")
        return

    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

    chat_engine = index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        verbose=True
    )

    print("\nChat started. Type 'reset' to clear.")
    print("Type 'back' to return.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "back":
            break
        if user_input.lower() == "reset":
            chat_engine.reset()
            print("Chat history cleared.\n")
            continue
        if not user_input:
            continue

        response = chat_engine.chat(user_input)
        print_chat_response(response)


# -------------------------------------------------------
# DEMO 3 — condense_plus_context mode (recommended)
# -------------------------------------------------------
def demo_condense_plus_context(index):
    print("\n" + "=" * 55)
    print("DEMO 3 — condense_plus_context mode")
    print("RECOMMENDED: Best mode for production RAG chatbot")
    print("Condenses history AND retrieves fresh context")
    print("=" * 55)

    if not index:
        print("No index. Build index first.")
        return

    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        memory=memory,
        system_prompt=(
            "You are a helpful document assistant. "
            "Answer questions only from the provided document context. "
            "If the answer is not in the document, clearly say so. "
            "Be concise and accurate."
        ),
        verbose=True,
        similarity_top_k=3
    )

    print("\nChat started. Type 'reset' to clear history.")
    print("Type 'history' to see conversation so far.")
    print("Type 'back' to return to menu.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "back":
            break

        if user_input.lower() == "reset":
            chat_engine.reset()
            print("Chat history cleared.\n")
            continue

        if user_input.lower() == "history":
            messages = chat_engine.chat_history
            if not messages:
                print("No history yet.\n")
            else:
                print(f"\nConversation history "
                      f"({len(messages)} messages):")
                for msg in messages:
                    role = msg.role.value.upper()
                    print(f"  [{role}]: {msg.content[:150]}...")
            continue

        if not user_input:
            continue

        response = chat_engine.chat(user_input)
        print_chat_response(response)


# -------------------------------------------------------
# DEMO 4 — simple mode (no retrieval)
# -------------------------------------------------------
def demo_simple_mode(index):
    print("\n" + "=" * 55)
    print("DEMO 4 — simple mode")
    print("Pure LLM chat — NO document retrieval")
    print("Compare answers with condense_plus_context")
    print("=" * 55)

    if not index:
        print("No index. Build index first.")
        return

    chat_engine = index.as_chat_engine(
        chat_mode="simple"
    )

    print("\nChat started (no document retrieval).")
    print("Type 'back' to return.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "back":
            break
        if not user_input:
            continue

        response = chat_engine.chat(user_input)
        print(f"\nAssistant: {response.response}")
        print("(Note: answer is from LLM training — not your document)\n")


# -------------------------------------------------------
# DEMO 5 — react mode
# -------------------------------------------------------
def demo_react_mode(index):
    print("\n" + "=" * 55)
    print("DEMO 5 — react mode")
    print("Agent-style: LLM decides whether to retrieve or not")
    print("=" * 55)

    if not index:
        print("No index. Build index first.")
        return

    chat_engine = index.as_chat_engine(
        chat_mode="react",
        verbose=True    # shows reasoning steps
    )

    print("\nChat started.")
    print("Watch verbose output — see when LLM decides to retrieve.")
    print("Type 'back' to return.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "back":
            break
        if not user_input:
            continue

        response = chat_engine.chat(user_input)
        print(f"\nAssistant: {response.response}\n")

# -------------------------------------------------------
# DEMO 6 — Memory Management
# -------------------------------------------------------
def demo_memory_management(index):
    print("\n" + "=" * 55)
    print("DEMO 6 — Memory Management")
    print("See how token_limit affects conversation history")
    print("=" * 55)

    if not index:
        print("No index. Build index first.")
        return

    token_limit = input(
        "Enter token_limit for memory (default 1500): "
    ).strip()
    token_limit = int(token_limit) if token_limit.isdigit() else 1500

    memory = ChatMemoryBuffer.from_defaults(
        token_limit=token_limit
    )

    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        memory=memory,
        verbose=False
    )
    print(f"\nMemory token limit: {token_limit}")
    print("Chat started. History will be truncated when limit reached.")
    print("Type 'history' to see stored messages.")
    print("Type 'reset' to clear.")
    print("Type 'back' to return.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "back":
            break

        if user_input.lower() == "reset":
            chat_engine.reset()
            print("History cleared.\n")
            continue

        if user_input.lower() == "history":
            messages = chat_engine.chat_history
            print(f"\nStored messages: {len(messages)}")
            for i, msg in enumerate(messages):
                role = msg.role.value.upper()
                print(f"  {i+1}. [{role}]: {msg.content[:100]}...")
            print()
            continue

        if not user_input:
            continue

        response = chat_engine.chat(user_input)
        print(f"\nAssistant: {response.response}")

        # Show memory usage after each message
        messages = chat_engine.chat_history
        print(f"\n[Memory: {len(messages)} messages stored "
              f"| token_limit: {token_limit}]\n")

# -------------------------------------------------------
# DEMO 7 — Streaming Chat
# -------------------------------------------------------
def demo_streaming_chat(index):
    print("\n" + "=" * 55)
    print("DEMO 7 — Streaming Chat")
    print("Tokens appear as they are generated")
    print("=" * 55)

    if not index:
        print("No index. Build index first.")
        return

    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        memory=memory,
        streaming=True
    )

    print("\nChat started (streaming mode).")
    print("Type 'back' to return.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "back":
            break
        if not user_input:
            continue

        print("\nAssistant: ", end="", flush=True)
        streaming_response = chat_engine.stream_chat(user_input)
        for token in streaming_response.response_gen:
            print(token, end="", flush=True)
        print("\n")

# -------------------------------------------------------
# DEMO 8 — Mode Comparison (same questions, all modes)
# -------------------------------------------------------
def demo_mode_comparison(index):
    print("\n" + "=" * 55)
    print("DEMO 8 — Mode Comparison")
    print("Same questions across all chat modes")
    print("=" * 55)

    if not index:
        print("No index. Build index first.")
        return

    questions = [
        input("Enter first question: ").strip(),
        input("Enter follow-up question (use 'it' or 'its'): ").strip()
    ]

    if not all(questions):
        print("Please enter both questions.")
        return

    modes = [
        "condense_question",
        "context",
        "condense_plus_context",
        "simple"
    ]

    for mode in modes:
        print(f"\n{'='*50}")
        print(f"Mode: {mode}")
        print(f"{'='*50}")

        try:
            memory = ChatMemoryBuffer.from_defaults(
                token_limit=3000
            )
            chat_engine = index.as_chat_engine(
                chat_mode=mode,
                memory=memory,
                verbose=False
            )

            for q in questions:
                print(f"\nYou: {q}")
                response = chat_engine.chat(q)
                print(f"Assistant: {response.response[:300]}...")

        except Exception as e:
            print(f"Mode {mode} failed: {e}")


# -------------------------------------------------------
# Main Menu
# -------------------------------------------------------
def main_menu(documents, index):
    while True:
        print("\n" + "=" * 55)
        print("Day 7 — Chat Engine: Memory & Context")
        print("=" * 55)
        print("  1  - Load documents")
        print("  2  - Build / Load index")
        print("  3  - Demo 1: condense_question mode")
        print("  4  - Demo 2: context mode")
        print("  5  - Demo 3: condense_plus_context (recommended)")
        print("  6  - Demo 4: simple mode (no retrieval)")
        print("  7  - Demo 5: react mode (agent-style)")
        print("  8  - Demo 6: Memory Management")
        print("  9  - Demo 7: Streaming Chat")
        print("  10 - Demo 8: Mode Comparison")
        print("  0  - Exit")
        print("=" * 55)

        choice = input("Enter choice: ").strip()

        if choice == "1":
            documents = load_documents()
        elif choice == "2":
            index = build_or_load_index(documents)
        elif choice == "3":
            demo_condense_question(index)
        elif choice == "4":
            demo_context_mode(index)
        elif choice == "5":
            demo_condense_plus_context(index)
        elif choice == "6":
            demo_simple_mode(index)
        elif choice == "7":
            demo_react_mode(index)
        elif choice == "8":
            demo_memory_management(index)
        elif choice == "9":
            demo_streaming_chat(index)
        elif choice == "10":
            demo_mode_comparison(index)
        elif choice == "0":
            print("\nExiting Day 7. Goodbye!")
            break
        else:
            print("Invalid. Enter 0-10.")

    return documents, index


def main():
    use_local = show_menu()
    configure(use_local=use_local)
    documents = []
    index = None
    main_menu(documents, index)


if __name__ == "__main__":
    main()