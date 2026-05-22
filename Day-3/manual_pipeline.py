# manual_pipeline.py
# Manual Production Pipeline
# PDF: Telangana_Overview_Styled.pdf
# Vector Store: Qdrant (Docker)

import logging
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from settings import configure

# -------------------------------------------------------
# Logging Setup
# -------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------
# Config
# -------------------------------------------------------
PDF_PATH = "data/Telangana_Overview_Styled.pdf"
COLLECTION_NAME = "telangana_overview"
QDRANT_URL = "http://localhost:6333"
EMBEDDING_DIM = 384   # all-MiniLM-L6-v2 produces 384 dimensions

# -------------------------------------------------------
# System Prompt — restricts the model to Telangana topics only
# Change this to update what the assistant is allowed to answer
# -------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an assistant that answers questions ONLY about Telangana state in India.\n"
    "Use ONLY the context provided below to answer.\n"
    "If the question is not related to Telangana, respond with: "
    "'I can only answer questions about Telangana.'\n"
    "If the answer is not found in the context, respond with: "
    "'I don't have enough information to answer that.'\n"
    "Do NOT make up any information. Be concise and factual."
)

# -------------------------------------------------------
# Step 1 — Load
# -------------------------------------------------------
def load_documents(pdf_path: str) -> list:
    try:
        documents = SimpleDirectoryReader(
            input_files=[pdf_path]
        ).load_data()
        logger.info(f"Loaded {len(documents)} pages from {pdf_path}")
        return documents
    except Exception as e:
        logger.error(f"Failed to load documents: {e}")
        return []

# -------------------------------------------------------
# Step 2 — Enrich Metadata
# -------------------------------------------------------
def enrich_metadata(documents: list) -> list:
    for doc in documents:
        doc.metadata["category"] = "telangana"
        doc.metadata["project"] = "LlamaIndex Learning"
        doc.metadata["source"] = "Telangana_Overview_Styled.pdf"
        doc.excluded_llm_metadata_keys = ["file_path", "creation_date"]
        doc.excluded_embed_metadata_keys = ["file_path"]
    logger.info(f"Metadata enriched for {len(documents)} documents")
    return documents

# -------------------------------------------------------
# Step 3 — Chunk
# -------------------------------------------------------
def chunk_documents(documents: list) -> list:
    splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50
    )
    nodes = splitter.get_nodes_from_documents(documents)
    logger.info(f"Created {len(nodes)} nodes")
    return nodes

# -------------------------------------------------------
# Step 4 — Validate
# -------------------------------------------------------
def validate_nodes(nodes: list) -> list:
    valid = []
    for node in nodes:
        if not node.text.strip():
            logger.warning(f"Skipping empty node {node.node_id}")
            continue
        if len(node.text) < 20:
            logger.warning(f"Skipping short node: '{node.text}'")
            continue
        valid.append(node)
    logger.info(f"Valid nodes: {len(valid)}/{len(nodes)}")
    return valid

# -------------------------------------------------------
# Step 5 — Embed
# -------------------------------------------------------
def embed_nodes(nodes: list) -> list:
    logger.info(f"Embedding {len(nodes)} nodes...")
    for i, node in enumerate(nodes):
        try:
            node.embedding = Settings.embed_model.get_text_embedding(node.text)
            if (i + 1) % 10 == 0:
                logger.info(f"  Embedded {i+1}/{len(nodes)} nodes")
        except Exception as e:
            logger.error(f"Failed to embed node {node.node_id}: {e}")
    logger.info("Embedding complete")
    return nodes

# -------------------------------------------------------
# Step 6 — Connect to Qdrant
# -------------------------------------------------------
def setup_qdrant() -> QdrantVectorStore:
    try:
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
            logger.info(f"Created Qdrant collection: {COLLECTION_NAME}")
        else:
            logger.info(f"Collection already exists: {COLLECTION_NAME}")
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME
        )
        return vector_store
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        raise

# -------------------------------------------------------
# Step 7 — Build Index and Store in Qdrant
# -------------------------------------------------------
def build_and_store(nodes: list, vector_store: QdrantVectorStore):
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        show_progress=True
    )
    logger.info(f"Index built and stored in Qdrant: {COLLECTION_NAME}")
    return index

# -------------------------------------------------------
# Step 8 — Quick Test Query
# System prompt is applied here via text_qa_template so the
# model is restricted on EVERY query automatically
# -------------------------------------------------------
def test_query(index):
    print("\n" + "=" * 50)
    print("Pipeline Test Query")
    print("Model is restricted to Telangana topics only")
    print("=" * 50)

    # Option A — system_prompt (simple, clean)
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        system_prompt=SYSTEM_PROMPT      # <-- restriction applied here
    )

    # Option B — text_qa_template (uncomment to use instead of Option A)
    # Gives you full control over the exact prompt format sent to the LLM
    #
    # text_qa_template = PromptTemplate(
    #     f"{SYSTEM_PROMPT}\n"
    #     "---------------------\n"
    #     "Context:\n{context_str}\n"
    #     "---------------------\n"
    #     "Question: {query_str}\n"
    #     "Answer: "
    # )
    # query_engine = index.as_query_engine(
    #     similarity_top_k=3,
    #     text_qa_template=text_qa_template
    # )

    print("\nSystem prompt active:")
    print(f"  {SYSTEM_PROMPT[:120]}...")
    print("\nType 'exit' to quit.\n")

    while True:
        question = input("Enter question (or 'exit'): ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue
        try:
            response = query_engine.query(question)
            print(f"\nAnswer : {response}")
            print(f"\nSource nodes used ({len(response.source_nodes)}):")
            for i, node in enumerate(response.source_nodes):
                print(f"\n  Node {i+1}:")
                print(f"    Score   : {node.score:.4f}")
                print(f"    Preview : {node.text[:200]}...")
        except Exception as e:
            logger.error(f"Query failed: {e}")

# -------------------------------------------------------
# Main Pipeline
# -------------------------------------------------------
def run_pipeline():
    print("\n" + "=" * 50)
    print("Manual Production Pipeline")
    print("PDF: Telangana_Overview_Styled.pdf")
    print("Vector Store: Qdrant (Docker)")
    print("=" * 50)

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
        else:
            print("Invalid. Enter 1 or 2.")

    logger.info("=== Pipeline Start ===")

    documents = load_documents(PDF_PATH)
    if not documents:
        logger.error("No documents loaded. Exiting.")
        return

    documents = enrich_metadata(documents)
    nodes = chunk_documents(documents)
    nodes = validate_nodes(nodes)
    nodes = embed_nodes(nodes)
    vector_store = setup_qdrant()
    index = build_and_store(nodes, vector_store)

    logger.info("=== Pipeline Complete ===")

    test_query(index)

if __name__ == "__main__":
    run_pipeline()