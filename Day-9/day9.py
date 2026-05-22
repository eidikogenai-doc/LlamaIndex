# day9_multidoc.py
# Day 9 — Multi-document & Metadata Filtering

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
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterCondition
)
from llama_index.core.query_engine import (
    RouterQueryEngine,
    SubQuestionQueryEngine,
    RetrieverQueryEngine
)
from llama_index.core.selectors import (
    LLMSingleSelector,
    LLMMultiSelector
)
from llama_index.core.tools import QueryEngineTool
from llama_index.core.indices.document_summary import (
    DocumentSummaryIndex
)
from llama_index.core import get_response_synthesizer
from settings import configure

DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data")
)
PERSIST_DIR = "storage/"


# -------------------------------------------------------
# Menu
# -------------------------------------------------------
def show_menu():
    print("\n" + "=" * 55)
    print("       LlamaIndex - Day 9")
    print("       Multi-document & Metadata Filtering")
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
# Show available files
# -------------------------------------------------------
def show_available_files():
    print("\nAvailable files in data folder:")
    available = [
        f for f in os.listdir(DATA_DIR)
        if os.path.isfile(os.path.join(DATA_DIR, f))
    ]
    for i, f in enumerate(available):
        print(f"  {i+1}. {f}")
    return available


# -------------------------------------------------------
# Load documents with metadata
# -------------------------------------------------------
def load_documents_with_metadata():
    print("\n" + "=" * 55)
    print("Load Documents with Metadata")
    print("=" * 55)
    print("You will load files and assign metadata to each.")
    print("This metadata is used for filtering later.\n")

    available = show_available_files()
    if not available:
        print("No files found in data folder.")
        return [], {}

    all_documents = []
    doc_registry = {}

    print("\nFor each file enter filename and metadata.")
    print("Empty filename to finish.\n")

    while True:
        filename = input("Filename: ").strip()
        if not filename:
            break

        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"  Not found: {filename}")
            continue

        # Collect metadata for this file
        print(f"  Setting metadata for: {filename}")
        doc_type = input(
            "  document_type (e.g. policy/claim/research/report): "
        ).strip() or "general"
        category = input(
            "  category (e.g. insurance/technology/finance): "
        ).strip() or "general"
        year = input(
            "  year (e.g. 2024): "
        ).strip() or "2024"
        department = input(
            "  department (e.g. legal/hr/it): "
        ).strip() or "general"

        # Load and tag
        docs = SimpleDirectoryReader(
            input_files=[filepath]
        ).load_data()

        for doc in docs:
            doc.metadata["source_file"] = filename
            doc.metadata["document_type"] = doc_type
            doc.metadata["category"] = category
            doc.metadata["year"] = year
            doc.metadata["department"] = department
            doc.excluded_llm_metadata_keys = [
                "file_path", "creation_date"
            ]
            doc.excluded_embed_metadata_keys = ["file_path"]

        all_documents.extend(docs)
        doc_registry[filename] = {
            "document_type": doc_type,
            "category": category,
            "year": year,
            "department": department,
            "page_count": len(docs)
        }

        print(f"  Loaded: {filename} — {len(docs)} pages")
        print(f"  Metadata: type={doc_type}, "
              f"category={category}, year={year}\n")

    print(f"\nTotal pages loaded: {len(all_documents)}")
    print(f"Total files: {len(doc_registry)}")
    return all_documents, doc_registry


# -------------------------------------------------------
# Build combined index from all documents
# -------------------------------------------------------
def build_combined_index(documents):
    if not documents:
        print("No documents to index.")
        return None

    if os.path.exists(PERSIST_DIR):
        print("\nExisting index found.")
        print("  1 - Load existing")
        print("  2 - Rebuild fresh")
        while True:
            choice = input("Enter choice (1 or 2): ").strip()
            if choice == "1":
                storage_context = StorageContext.from_defaults(
                    persist_dir=PERSIST_DIR
                )
                return load_index_from_storage(storage_context)
            elif choice == "2":
                break
            print("Invalid.")

    print("\nBuilding combined index...")
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)
    nodes = [n for n in nodes if len(n.text.strip()) > 20]
    print(f"Total nodes: {len(nodes)}")

    index = VectorStoreIndex(nodes, show_progress=True)
    index.storage_context.persist(PERSIST_DIR)
    print(f"Index saved to {PERSIST_DIR}")
    return index


# -------------------------------------------------------
# Show metadata summary
# -------------------------------------------------------
def show_metadata_summary(documents, doc_registry):
    print("\n" + "=" * 55)
    print("Metadata Summary")
    print("=" * 55)

    if not doc_registry:
        print("No documents loaded yet.")
        return

    for filename, meta in doc_registry.items():
        print(f"\n  File: {filename}")
        for key, val in meta.items():
            print(f"    {key}: {val}")


# -------------------------------------------------------
# DEMO 1 — No filter (baseline)
# -------------------------------------------------------
def demo_no_filter(index):
    print("\n" + "=" * 55)
    print("DEMO 1 — No Filter")
    print("Searches ALL documents — no restrictions")
    print("=" * 55)

    if not index:
        print("Build index first.")
        return

    query_engine = index.as_query_engine(similarity_top_k=3)

    print("\nType 'back' to return.")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() == "back":
            break
        if not question:
            continue

        response = query_engine.query(question)
        print(f"\nAnswer : {response}")
        print(f"\nSources ({len(response.source_nodes)}):")
        for i, node in enumerate(response.source_nodes):
            score = f"{node.score:.4f}" if node.score else "N/A"
            print(f"  {i+1}. File: "
                  f"{node.metadata.get('source_file')} "
                  f"| Type: {node.metadata.get('document_type')} "
                  f"| Score: {score}")
            print(f"     {node.text[:120]}...")


# -------------------------------------------------------
# DEMO 2 — Single Metadata Filter
# -------------------------------------------------------
def demo_single_filter(index, documents):
    print("\n" + "=" * 55)
    print("DEMO 2 — Single Metadata Filter")
    print("Filter by one metadata key")
    print("=" * 55)

    if not index:
        print("Build index first.")
        return

    # Show available metadata values
    print("\nAvailable metadata values in your index:")
    metadata_values = {}
    for doc in documents:
        for key in ["source_file", "document_type",
                    "category", "year", "department"]:
            if key not in metadata_values:
                metadata_values[key] = set()
            metadata_values[key].add(doc.metadata.get(key, "unknown"))

    for key, values in metadata_values.items():
        print(f"  {key}: {', '.join(sorted(values))}")

    print("\nEnter filter details:")
    key = input("Filter key (e.g. source_file): ").strip()
    value = input("Filter value: ").strip()

    if not key or not value:
        print("Key and value required.")
        return

    filters = MetadataFilters(
        filters=[MetadataFilter(key=key, value=value)]
    )

    query_engine = index.as_query_engine(
        similarity_top_k=3,
        filters=filters
    )

    print(f"\nFiltering: {key} = {value}")
    print("Type 'back' to return.")

    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() == "back":
            break
        if not question:
            continue

        response = query_engine.query(question)
        print(f"\nAnswer : {response}")
        print(f"\nSources ({len(response.source_nodes)}):")
        for i, node in enumerate(response.source_nodes):
            score = f"{node.score:.4f}" if node.score else "N/A"
            print(f"  {i+1}. File: "
                  f"{node.metadata.get('source_file')} "
                  f"| Score: {score}")


# -------------------------------------------------------
# DEMO 3 — AND Filter
# -------------------------------------------------------
def demo_and_filter(index, documents):
    print("\n" + "=" * 55)
    print("DEMO 3 — AND Filter")
    print("ALL conditions must match")
    print("=" * 55)

    if not index:
        print("Build index first.")
        return

    print("\nAvailable metadata:")
    for doc in documents[:1]:
        for key, val in doc.metadata.items():
            if key not in ["file_path", "creation_date",
                           "last_modified_date", "file_size",
                           "file_type"]:
                print(f"  {key}: {val}")

    print("\nEnter up to 3 filter conditions (AND logic).")
    print("Empty key to stop.\n")

    filter_list = []
    for i in range(3):
        key = input(f"Condition {i+1} key: ").strip()
        if not key:
            break
        value = input(f"Condition {i+1} value: ").strip()
        if not value:
            break
        filter_list.append(
            MetadataFilter(key=key, value=value)
        )
        print(f"  Added: {key} = {value}")

    if not filter_list:
        print("No filters entered.")
        return

    filters = MetadataFilters(
        filters=filter_list,
        condition=FilterCondition.AND
    )

    query_engine = index.as_query_engine(
        similarity_top_k=3,
        filters=filters
    )

    conditions = " AND ".join(
        [f"{f.key}={f.value}" for f in filter_list]
    )
    print(f"\nFiltering: {conditions}")
    print("Type 'back' to return.")

    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() == "back":
            break
        if not question:
            continue

        response = query_engine.query(question)
        print(f"\nAnswer : {response}")
        print(f"\nSources ({len(response.source_nodes)}):")
        for i, node in enumerate(response.source_nodes):
            score = f"{node.score:.4f}" if node.score else "N/A"
            print(f"  {i+1}. {node.metadata.get('source_file')} "
                  f"| Score: {score}")


# -------------------------------------------------------
# DEMO 4 — OR Filter
# -------------------------------------------------------
def demo_or_filter(index, documents):
    print("\n" + "=" * 55)
    print("DEMO 4 — OR Filter")
    print("ANY condition can match")
    print("=" * 55)

    if not index:
        print("Build index first.")
        return

    print("\nEnter filter conditions (OR logic).")
    print("Useful for: 'search in doc1 OR doc2'")
    print("Empty key to stop.\n")

    filter_list = []
    for i in range(5):
        key = input(f"Condition {i+1} key: ").strip()
        if not key:
            break
        value = input(f"Condition {i+1} value: ").strip()
        if not value:
            break
        filter_list.append(
            MetadataFilter(key=key, value=value)
        )
        print(f"  Added: {key} = {value}")

    if not filter_list:
        print("No filters entered.")
        return

    filters = MetadataFilters(
        filters=filter_list,
        condition=FilterCondition.OR
    )

    query_engine = index.as_query_engine(
        similarity_top_k=3,
        filters=filters
    )

    conditions = " OR ".join(
        [f"{f.key}={f.value}" for f in filter_list]
    )
    print(f"\nFiltering: {conditions}")
    print("Type 'back' to return.")

    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() == "back":
            break
        if not question:
            continue

        response = query_engine.query(question)
        print(f"\nAnswer : {response}")
        print(f"\nSources ({len(response.source_nodes)}):")
        for i, node in enumerate(response.source_nodes):
            score = f"{node.score:.4f}" if node.score else "N/A"
            print(f"  {i+1}. {node.metadata.get('source_file')} "
                  f"| Score: {score}")


# -------------------------------------------------------
# DEMO 5 — RouterQueryEngine
# Automatically routes to correct document
# -------------------------------------------------------
def demo_router(documents):
    print("\n" + "=" * 55)
    print("DEMO 5 — RouterQueryEngine")
    print("LLM automatically routes query to correct engine")
    print("=" * 55)

    if not documents:
        print("Load documents first.")
        return

    # Group documents by category
    category_docs = {}
    for doc in documents:
        cat = doc.metadata.get("category", "general")
        if cat not in category_docs:
            category_docs[cat] = []
        category_docs[cat].append(doc)

    if len(category_docs) < 2:
        print("Need at least 2 different categories for routing.")
        print(f"Found categories: {list(category_docs.keys())}")
        print("Load documents with different categories first.")
        return

    print(f"\nBuilding separate indexes for:")
    for cat, docs in category_docs.items():
        print(f"  {cat}: {len(docs)} pages")

    # Build index per category
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    tools = []

    for category, docs in category_docs.items():
        nodes = splitter.get_nodes_from_documents(docs)
        nodes = [n for n in nodes if len(n.text.strip()) > 20]
        cat_index = VectorStoreIndex(nodes, show_progress=False)
        cat_engine = cat_index.as_query_engine(
            similarity_top_k=3
        )

        # Get document types in this category
        types = set(
            d.metadata.get("document_type", "general")
            for d in docs
        )
        files = set(
            d.metadata.get("source_file", "unknown")
            for d in docs
        )

        tool = QueryEngineTool.from_defaults(
            query_engine=cat_engine,
            name=f"{category}_tool",
            description=(
                f"Use this for questions about {category}. "
                f"Contains: {', '.join(files)}. "
                f"Document types: {', '.join(types)}."
            )
        )
        tools.append(tool)
        print(f"  Built tool: {category}_tool")

    # Single selector — routes to ONE engine
    router_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=tools,
        verbose=True
    )

    print("\nRouter ready. LLM will decide which engine to use.")
    print("Type 'back' to return.")

    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() == "back":
            break
        if not question:
            continue

        response = router_engine.query(question)
        print(f"\nAnswer : {response}")


# -------------------------------------------------------
# DEMO 6 — SubQuestionQueryEngine
# Breaks complex questions into sub-questions
# -------------------------------------------------------
def demo_sub_question(documents):
    print("\n" + "=" * 55)
    print("DEMO 6 — SubQuestionQueryEngine")
    print("Breaks complex queries into sub-questions")
    print("Each sub-question answered by correct document")
    print("=" * 55)

    if not documents:
        print("Load documents first.")
        return

    # Group by source file
    file_docs = {}
    for doc in documents:
        src = doc.metadata.get("source_file", "unknown")
        if src not in file_docs:
            file_docs[src] = []
        file_docs[src].append(doc)

    if len(file_docs) < 2:
        print("Need at least 2 different files.")
        return

    print(f"\nBuilding tools for {len(file_docs)} documents:")
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    engines = {}

    for filename, docs in file_docs.items():
        nodes = splitter.get_nodes_from_documents(docs)
        nodes = [n for n in nodes if len(n.text.strip()) > 20]
        file_index = VectorStoreIndex(
            nodes, show_progress=False
        )
        engines[filename] = file_index.as_query_engine(
            similarity_top_k=3
        )
        print(f"  Built engine: {filename}")

    print("\nManual SubQuestion engine ready.")
    print("Each question is sent to ALL documents separately.")
    print("Then answers are combined by LLM.")
    print("Type 'back' to return.\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() == "back":
            break
        if not question:
            continue

        print(f"\nQuerying each document separately...")
        all_answers = []

        for filename, engine in engines.items():
            try:
                print(f"\n  Querying: {filename}")
                response = engine.query(question)
                answer = str(response).strip()
                print(f"  Answer: {answer[:250]}...")
                all_answers.append(
                    f"From [{filename}]:\n{answer}"
                )
            except Exception as e:
                print(f"  Failed: {e}")

        # Combine with LLM
        if all_answers:
            print(f"\n{'='*45}")
            print("Combined synthesis:")
            print(f"{'='*45}")
            combined = "\n\n".join(all_answers)
            prompt = (
                f"You have answers from multiple documents "
                f"for the question: '{question}'\n\n"
                f"{combined}\n\n"
                f"Synthesize a single comprehensive answer "
                f"combining insights from all documents:"
            )
            try:
                final = Settings.llm.complete(prompt)
                print(f"\n{final}")
            except Exception as e:
                print(f"Synthesis failed: {e}")
                print("See individual answers above.")

# -------------------------------------------------------
# DEMO 7 — DocumentSummaryIndex
# -------------------------------------------------------
def demo_document_summary_index(documents):
    print("\n" + "=" * 55)
    print("DEMO 7 — DocumentSummaryIndex")
    print("Generates LLM summary per document")
    print("Routes via summary then retrieves chunks")
    print("=" * 55)

    if not documents:
        print("Load documents first.")
        return

    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    print("\nBuilding DocumentSummaryIndex...")
    print("(LLM will generate a summary for each document)")

    try:
        index = DocumentSummaryIndex.from_documents(
            documents,
            transformations=[splitter],
            show_progress=True,
            response_synthesizer_llm=Settings.llm
        )

        # Show generated summaries
        print("\nGenerated document summaries:")
        for doc in documents[:2]:
            doc_id = doc.doc_id
            try:
                summary = index.get_document_summary(doc_id)
                src = doc.metadata.get("source_file", "unknown")
                print(f"\n  {src}:")
                print(f"  {summary[:300]}...")
            except Exception:
                pass

        query_engine = index.as_query_engine(
            response_mode="tree_summarize"
        )

        print("\nType 'back' to return.")
        while True:
            question = input("\nQuestion: ").strip()
            if question.lower() == "back":
                break
            if not question:
                continue

            response = query_engine.query(question)
            print(f"\nAnswer : {response}")

    except Exception as e:
        print(f"DocumentSummaryIndex failed: {e}")


# -------------------------------------------------------
# DEMO 8 — Separate Indexes per Document
# -------------------------------------------------------
def demo_separate_indexes(documents):
    print("\n" + "=" * 55)
    print("DEMO 8 — Separate Indexes per Document")
    print("Each file gets its own index")
    print("Query specific documents directly")
    print("=" * 55)

    if not documents:
        print("Load documents first.")
        return

    # Group by source file
    file_docs = {}
    for doc in documents:
        src = doc.metadata.get("source_file", "unknown")
        if src not in file_docs:
            file_docs[src] = []
        file_docs[src].append(doc)

    # Build separate index per file
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    indexes = {}

    print("\nBuilding separate indexes:")
    for filename, docs in file_docs.items():
        nodes = splitter.get_nodes_from_documents(docs)
        nodes = [n for n in nodes if len(n.text.strip()) > 20]
        idx = VectorStoreIndex(nodes, show_progress=False)
        indexes[filename] = idx
        print(f"  {filename} → {len(nodes)} nodes")

    print("\nAvailable indexes:")
    filenames = list(indexes.keys())
    for i, f in enumerate(filenames):
        print(f"  {i+1}. {f}")

    while True:
        print("\nOptions:")
        print("  Enter number to select document")
        print("  'all' to search all documents")
        print("  'back' to return")

        choice = input("\nSelect: ").strip()

        if choice.lower() == "back":
            break

        if choice.lower() == "all":
            # Search all indexes
            question = input("Question: ").strip()
            if not question:
                continue

            print(f"\nSearching all {len(indexes)} indexes...")
            for filename, idx in indexes.items():
                qe = idx.as_query_engine(similarity_top_k=2)
                response = qe.query(question)
                print(f"\n  [{filename}]")
                print(f"  {str(response)[:200]}...")

        elif choice.isdigit():
            idx_num = int(choice) - 1
            if 0 <= idx_num < len(filenames):
                selected_file = filenames[idx_num]
                selected_index = indexes[selected_file]
                qe = selected_index.as_query_engine(
                    similarity_top_k=3
                )

                print(f"\nQuerying: {selected_file}")
                print("Type 'back' to go back.")

                while True:
                    question = input("\nQuestion: ").strip()
                    if question.lower() == "back":
                        break
                    if not question:
                        continue

                    response = qe.query(question)
                    print(f"\nAnswer : {response}")
                    for i, node in enumerate(
                        response.source_nodes
                    ):
                        score = (
                            f"{node.score:.4f}"
                            if node.score else "N/A"
                        )
                        print(f"\n  Node {i+1} | Score: {score}")
                        print(f"  {node.text[:200]}...")
            else:
                print("Invalid number.")
        else:
            print("Invalid choice.")


# -------------------------------------------------------
# DEMO 9 — Filter Comparison
# Same question with different filters side by side
# -------------------------------------------------------
def demo_filter_comparison(index, documents):
    print("\n" + "=" * 55)
    print("DEMO 9 — Filter Comparison")
    print("Same question — no filter vs filtered")
    print("=" * 55)

    if not index:
        print("Build index first.")
        return

    question = input("Enter question: ").strip()
    if not question:
        return

    # Show available filters
    print("\nAvailable filter options:")
    metadata_values = {}
    for doc in documents:
        for key in ["source_file", "document_type", "category"]:
            if key not in metadata_values:
                metadata_values[key] = set()
            metadata_values[key].add(
                doc.metadata.get(key, "unknown")
            )

    for key, values in metadata_values.items():
        print(f"  {key}: {', '.join(sorted(values))}")

    key = input("\nFilter key to test: ").strip()
    value = input("Filter value to test: ").strip()

    if not key or not value:
        print("Key and value required.")
        return

    # No filter
    print(f"\n{'='*45}")
    print("No filter — searches all documents:")
    print(f"{'='*45}")
    qe_no_filter = index.as_query_engine(similarity_top_k=3)
    response1 = qe_no_filter.query(question)
    print(f"Answer : {str(response1)[:300]}...")
    print(f"Sources: {[n.metadata.get('source_file') for n in response1.source_nodes]}")

    # With filter
    print(f"\n{'='*45}")
    print(f"With filter — {key} = {value}:")
    print(f"{'='*45}")
    filters = MetadataFilters(
        filters=[MetadataFilter(key=key, value=value)]
    )
    qe_filtered = index.as_query_engine(
        similarity_top_k=3,
        filters=filters
    )
    response2 = qe_filtered.query(question)
    print(f"Answer : {str(response2)[:300]}...")
    print(f"Sources: {[n.metadata.get('source_file') for n in response2.source_nodes]}")


# -------------------------------------------------------
# Main Menu
# -------------------------------------------------------
def main_menu():
    documents = []
    doc_registry = {}
    index = None

    while True:
        print("\n" + "=" * 55)
        print("Day 9 — Multi-document & Metadata Filtering")
        print("=" * 55)
        print("  1  - Load documents with metadata")
        print("  2  - Build combined index")
        print("  3  - Show metadata summary")
        print("  4  - Demo 1: No filter (baseline)")
        print("  5  - Demo 2: Single metadata filter")
        print("  6  - Demo 3: AND filter")
        print("  7  - Demo 4: OR filter")
        print("  8  - Demo 5: RouterQueryEngine")
        print("  9  - Demo 6: SubQuestionQueryEngine")
        print("  10 - Demo 7: DocumentSummaryIndex")
        print("  11 - Demo 8: Separate indexes per document")
        print("  12 - Demo 9: Filter comparison")
        print("  0  - Exit")
        print("=" * 55)

        choice = input("Enter choice: ").strip()

        if choice == "1":
            documents, doc_registry = load_documents_with_metadata()

        elif choice == "2":
            index = build_combined_index(documents)

        elif choice == "3":
            show_metadata_summary(documents, doc_registry)

        elif choice == "4":
            demo_no_filter(index)

        elif choice == "5":
            demo_single_filter(index, documents)

        elif choice == "6":
            demo_and_filter(index, documents)

        elif choice == "7":
            demo_or_filter(index, documents)

        elif choice == "8":
            demo_router(documents)

        elif choice == "9":
            demo_sub_question(documents)

        elif choice == "10":
            demo_document_summary_index(documents)

        elif choice == "11":
            demo_separate_indexes(documents)

        elif choice == "12":
            demo_filter_comparison(index, documents)

        elif choice == "0":
            print("\nExiting Day 9. Goodbye!")
            break

        else:
            print("Invalid. Enter 0-12.")


def main():
    use_local = show_menu()
    configure(use_local=use_local)
    main_menu()


if __name__ == "__main__":
    main()