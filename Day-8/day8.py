# day8_advanced_retrieval.py
# Day 8 — Advanced Retrieval: Re-ranking & Hybrid Search

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
from llama_index.core.node_parser import (
    SentenceSplitter,
    SentenceWindowNodeParser,
    HierarchicalNodeParser
)
from llama_index.core.retrievers import (
    VectorIndexRetriever,
    QueryFusionRetriever,
    AutoMergingRetriever
)
from llama_index.core.postprocessor import (
    LLMRerank,
    SimilarityPostprocessor,
    MetadataReplacementPostProcessor
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer
from llama_index.postprocessor.sbert_rerank import (
    SentenceTransformerRerank
)
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.storage.docstore import SimpleDocumentStore
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
    print("       LlamaIndex - Day 8")
    print("       Advanced Retrieval")
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
        choice = input("Enter choice: ").strip()
        if choice in ["1", "2"]:
            break
        print("Invalid.")

    documents = []
    if choice == "1":
        print("Enter filenames. Empty line to finish.")
        while True:
            filename = input("Filename: ").strip()
            if not filename:
                break
            path = os.path.join(DATA_DIR, filename)
            if os.path.exists(path):
                docs = SimpleDirectoryReader(
                    input_files=[path]
                ).load_data()
                for doc in docs:
                    doc.metadata["source_file"] = filename
                documents.extend(docs)
                print(f"  Loaded: {filename} ({len(docs)} pages)")
            else:
                print(f"  Not found: {filename}")
    else:
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        for doc in documents:
            doc.metadata["source_file"] = doc.metadata.get(
                "file_name", "unknown"
            )
        print(f"Loaded {len(documents)} pages")

    return documents


# -------------------------------------------------------
# Build Standard Index
# -------------------------------------------------------
def build_standard_index(documents):
    if os.path.exists(PERSIST_DIR):
        print("Loading existing index from disk...")
        storage_context = StorageContext.from_defaults(
            persist_dir=PERSIST_DIR
        )
        return load_index_from_storage(storage_context)

    print("Building standard index...")
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)
    nodes = [n for n in nodes if len(n.text.strip()) > 20]
    print(f"Nodes created: {len(nodes)}")

    index = VectorStoreIndex(nodes, show_progress=True)
    index.storage_context.persist(PERSIST_DIR)
    print(f"Index saved to {PERSIST_DIR}")
    return index


# -------------------------------------------------------
# Helper — print response
# -------------------------------------------------------
def print_response(response):
    print(f"\nAnswer : {response}")
    if hasattr(response, "source_nodes") and response.source_nodes:
        print(f"\nSource nodes ({len(response.source_nodes)}):")
        for i, node in enumerate(response.source_nodes):
            score = f"{node.score:.4f}" if node.score else "N/A"
            print(f"\n  Node {i+1}:")
            print(f"    Score  : {score}")
            print(f"    File   : {node.metadata.get('source_file')}")
            print(f"    Text   : {node.text[:200]}...")


# -------------------------------------------------------
# DEMO 1 — Baseline (no re-ranking)
# Compare against re-ranked results
# -------------------------------------------------------
def demo_baseline(index):
    print("\n" + "=" * 55)
    print("DEMO 1 — Baseline Retrieval (no re-ranking)")
    print("Standard Top-K similarity search")
    print("=" * 55)

    query_engine = index.as_query_engine(
        similarity_top_k=3
    )

    print("\nType 'back' to return.")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() == "back":
            break
        if not question:
            continue
        response = query_engine.query(question)
        print_response(response)


# -------------------------------------------------------
# DEMO 2 — LLMRerank
# LLM re-scores retrieved nodes
# -------------------------------------------------------
def demo_llm_rerank(index):
    print("\n" + "=" * 55)
    print("DEMO 2 — LLMRerank")
    print("LLM re-scores Top-10 → returns Top-3")
    print("Slower but most accurate")
    print("=" * 55)

    reranker = LLMRerank(
        choice_batch_size=5,
        top_n=3
    )

    query_engine = index.as_query_engine(
        similarity_top_k=10,
        node_postprocessors=[reranker]
    )

    print("\nType 'back' to return.")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() == "back":
            break
        if not question:
            continue
        response = query_engine.query(question)
        print_response(response)


# -------------------------------------------------------
# DEMO 3 — SentenceTransformerRerank (recommended)
# Local cross-encoder re-ranker — free and fast
# -------------------------------------------------------
def demo_sbert_rerank(index):
    print("\n" + "=" * 55)
    print("DEMO 3 — SentenceTransformerRerank")
    print("Local cross-encoder — free, no API key")
    print("Retrieve Top-10 → re-rank → return Top-3")
    print("=" * 55)

    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-2-v2",
        top_n=3
    )

    query_engine = index.as_query_engine(
        similarity_top_k=10,
        node_postprocessors=[reranker]
    )

    print("\nType 'back' to return.")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() == "back":
            break
        if not question:
            continue
        response = query_engine.query(question)
        print_response(response)


# -------------------------------------------------------
# DEMO 4 — Score Threshold + Re-ranking combined
# -------------------------------------------------------
def demo_threshold_rerank(index):
    print("\n" + "=" * 55)
    print("DEMO 4 — Score Threshold + Re-ranking")
    print("Filter low scores THEN re-rank remaining")
    print("=" * 55)

    similarity_filter = SimilarityPostprocessor(
        similarity_cutoff=0.3
    )
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-2-v2",
        top_n=3
    )

    query_engine = index.as_query_engine(
        similarity_top_k=10,
        node_postprocessors=[similarity_filter, reranker]
    )

    print("\nType 'back' to return.")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() == "back":
            break
        if not question:
            continue
        response = query_engine.query(question)
        print_response(response)


# -------------------------------------------------------
# DEMO 5 — BM25 Retriever (keyword search)
# -------------------------------------------------------
def demo_bm25(index):
    print("\n" + "=" * 55)
    print("DEMO 5 — BM25Retriever")
    print("Pure keyword-based retrieval — no embeddings")
    print("Best for exact term matching")
    print("=" * 55)

    bm25_retriever = BM25Retriever.from_defaults(
        docstore=index.docstore,
        similarity_top_k=5
    )

    synthesizer = get_response_synthesizer(
        response_mode="compact"
    )
    query_engine = RetrieverQueryEngine(
        retriever=bm25_retriever,
        response_synthesizer=synthesizer
    )

    print("\nType 'back' to return.")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() == "back":
            break
        if not question:
            continue
        response = query_engine.query(question)
        print_response(response)


# -------------------------------------------------------
# DEMO 6 — Hybrid Search (Vector + BM25)
# QueryFusionRetriever combines both
# -------------------------------------------------------
def demo_hybrid_search(index):
    print("\n" + "=" * 55)
    print("DEMO 6 — Hybrid Search")
    print("Vector search + BM25 keyword search combined")
    print("Best of both worlds")
    print("=" * 55)

    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=5
    )
    bm25_retriever = BM25Retriever.from_defaults(
        docstore=index.docstore,
        similarity_top_k=5
    )

    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=5,
        num_queries=4,
        mode="reciprocal_rerank",
        use_async=False,
        verbose=True
    )

    synthesizer = get_response_synthesizer(
        response_mode="compact"
    )
    query_engine = RetrieverQueryEngine(
        retriever=hybrid_retriever,
        response_synthesizer=synthesizer
    )

    print("\nType 'back' to return.")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() == "back":
            break
        if not question:
            continue
        response = query_engine.query(question)
        print_response(response)


# -------------------------------------------------------
# DEMO 7 — Hybrid Search + Re-ranking (full pipeline)
# -------------------------------------------------------
def demo_hybrid_rerank(index):
    print("\n" + "=" * 55)
    print("DEMO 7 — Hybrid Search + Re-ranking")
    print("Production-grade retrieval pipeline")
    print("Vector + BM25 → Fused → Re-ranked")
    print("=" * 55)

    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10
    )
    bm25_retriever = BM25Retriever.from_defaults(
        docstore=index.docstore,
        similarity_top_k=10
    )

    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=10,
        num_queries=4,
        mode="reciprocal_rerank",
        use_async=False,
        verbose=False
    )

    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-2-v2",
        top_n=3
    )

    synthesizer = get_response_synthesizer(
        response_mode="compact"
    )
    query_engine = RetrieverQueryEngine(
        retriever=hybrid_retriever,
        response_synthesizer=synthesizer,
        node_postprocessors=[reranker]
    )

    print("\nType 'back' to return.")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() == "back":
            break
        if not question:
            continue
        response = query_engine.query(question)
        print_response(response)


# -------------------------------------------------------
# DEMO 8 — SentenceWindow Retrieval
# Precise sentence retrieval + context window to LLM
# -------------------------------------------------------
def demo_sentence_window(documents):
    print("\n" + "=" * 55)
    print("DEMO 8 — SentenceWindowNodeParser")
    print("Retrieve precise sentences")
    print("But send surrounding context to LLM")
    print("=" * 55)

    splitter = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_sentence"
    )
    nodes = splitter.get_nodes_from_documents(documents)
    nodes = [n for n in nodes if len(n.text.strip()) > 5]
    print(f"Sentence nodes created: {len(nodes)}")

    index = VectorStoreIndex(nodes, show_progress=True)

    postprocessor = MetadataReplacementPostProcessor(
        target_metadata_key="window"
    )
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-2-v2",
        top_n=3
    )

    query_engine = index.as_query_engine(
        similarity_top_k=10,
        node_postprocessors=[postprocessor, reranker]
    )

    print("\nType 'back' to return.")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() == "back":
            break
        if not question:
            continue
        response = query_engine.query(question)
        print_response(response)


# -------------------------------------------------------
# DEMO 9 — Retrieval Comparison
# Same question across all retrieval strategies
# -------------------------------------------------------
def demo_retrieval_comparison(index):
    print("\n" + "=" * 55)
    print("DEMO 9 — Retrieval Strategy Comparison")
    print("Same question across all strategies")
    print("=" * 55)

    question = input("Enter question to compare: ").strip()
    if not question:
        return

    strategies = {
        "Baseline (Top-3)": index.as_query_engine(
            similarity_top_k=3
        ),
        "Top-10 + SBERTRerank": index.as_query_engine(
            similarity_top_k=10,
            node_postprocessors=[
                SentenceTransformerRerank(
                    model="cross-encoder/ms-marco-MiniLM-L-2-v2",
                    top_n=3
                )
            ]
        ),
        "Top-10 + Score Filter": index.as_query_engine(
            similarity_top_k=10,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=0.3)
            ]
        ),
    }

    for name, qe in strategies.items():
        print(f"\n{'='*45}")
        print(f"Strategy: {name}")
        print(f"{'='*45}")
        try:
            response = qe.query(question)
            print(f"Answer : {str(response)[:400]}...")
            print(f"Nodes  : {len(response.source_nodes)}")
            for i, node in enumerate(response.source_nodes):
                score = f"{node.score:.4f}" if node.score else "N/A"
                print(f"  {i+1}. Score: {score} | "
                      f"{node.text[:100]}...")
        except Exception as e:
            print(f"Error: {e}")


# -------------------------------------------------------
# Main Menu
# -------------------------------------------------------
def main_menu(documents, index):
    while True:
        print("\n" + "=" * 55)
        print("Day 8 — Advanced Retrieval")
        print("=" * 55)
        print("  1  - Load documents")
        print("  2  - Build / Load index")
        print("  3  - Demo 1: Baseline (no re-ranking)")
        print("  4  - Demo 2: LLMRerank")
        print("  5  - Demo 3: SentenceTransformerRerank ✅")
        print("  6  - Demo 4: Score Threshold + Re-ranking")
        print("  7  - Demo 5: BM25 Keyword Retrieval")
        print("  8  - Demo 6: Hybrid Search")
        print("  9  - Demo 7: Hybrid + Re-ranking (production)")
        print("  10 - Demo 8: SentenceWindow Retrieval")
        print("  11 - Demo 9: Strategy Comparison")
        print("  0  - Exit")
        print("=" * 55)

        choice = input("Enter choice: ").strip()

        if choice == "1":
            documents = load_documents()
        elif choice == "2":
            index = build_standard_index(documents)
        elif choice == "3":
            demo_baseline(index)
        elif choice == "4":
            demo_llm_rerank(index)
        elif choice == "5":
            demo_sbert_rerank(index)
        elif choice == "6":
            demo_threshold_rerank(index)
        elif choice == "7":
            demo_bm25(index)
        elif choice == "8":
            demo_hybrid_search(index)
        elif choice == "9":
            demo_hybrid_rerank(index)
        elif choice == "10":
            if documents:
                demo_sentence_window(documents)
            else:
                print("Load documents first.")
        elif choice == "11":
            demo_retrieval_comparison(index)
        elif choice == "0":
            print("\nExiting Day 8. Goodbye!")
            break
        else:
            print("Invalid. Enter 0-11.")

    return documents, index


def main():
    use_local = show_menu()
    configure(use_local=use_local)
    documents = []
    index = None
    main_menu(documents, index)


if __name__ == "__main__":
    main()