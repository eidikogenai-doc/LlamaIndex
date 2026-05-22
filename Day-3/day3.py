# day3_embeddings.py
# Day 3 — Embeddings & Vector Basics + Data Ingestion
# Compatible with llama-index-core 0.14.x


import numpy as np
import sys
import os
import time
import requests

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))

from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.extractors import TitleExtractor, KeywordExtractor
from settings import configure

DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data")
)

# -------------------------------------------------------
# Menu
# -------------------------------------------------------
def show_menu():
    print("\n" + "=" * 50)
    print("       LlamaIndex - Day 3")
    print("       Embeddings & Data Ingestion")
    print("=" * 50)
    print("Select LLM Provider:")
    print("  1 - Groq (cloud)")
    print("  2 - LM Studio (local)")
    print("=" * 50)
    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice == "1":
            return False
        elif choice == "2":
            return True
        print("Invalid. Enter 1 or 2.")

# -------------------------------------------------------
# PART 1 — Embeddings + Cosine Similarity
# -------------------------------------------------------
def demo_embeddings():
    print("\n" + "=" * 50)
    print("PART 1 — Embeddings")
    print("See what a vector actually looks like")
    print("=" * 50)

    sentences = [
        "Insurance covers fire damage",
        "Policy protects against fire loss",
        "I like eating pizza",
        "The cat sat on the mat"
    ]

    print("Generating embeddings...")
    vectors = []
    for sentence in sentences:
        vector = Settings.embed_model.get_text_embedding(sentence)
        vectors.append(vector)
        print(f"\nSentence   : {sentence}")
        print(f"Dimensions : {len(vector)}")
        print(f"First 5    : {[round(v, 4) for v in vector[:5]]}")

    print("\n" + "=" * 50)
    print("Cosine Similarity")
    print("=" * 50)

    def cosine_similarity(v1, v2):
        v1 = np.array(v1)
        v2 = np.array(v2)
        return np.dot(v1, v2) / (
            np.linalg.norm(v1) * np.linalg.norm(v2)
        )

    base = sentences[0]
    base_vec = vectors[0]
    for sentence, vector in zip(sentences[1:], vectors[1:]):
        score = cosine_similarity(base_vec, vector)
        print(f"\n  '{base}'")
        print(f"  vs '{sentence}'")
        print(f"  Score: {score:.4f}")
        if score > 0.7:
            print("  → Very similar")
        elif score > 0.4:
            print("  → Somewhat related")
        else:
            print("  → Not related")


# -------------------------------------------------------
# PART 2 — SimpleDirectoryReader
# -------------------------------------------------------
def demo_simple_directory_reader():
    print("\n" + "=" * 50)
    print("PART 2 — SimpleDirectoryReader")
    print("Auto-detects all file types")
    print("=" * 50)

    if not os.path.exists(DATA_DIR):
        print(f"Data folder not found: {DATA_DIR}")
        return []

    # Load all files
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    print(f"All files loaded: {len(documents)} pages")

    # Load only PDFs
    pdf_docs = SimpleDirectoryReader(
        DATA_DIR,
        required_exts=[".pdf"]
    ).load_data()
    print(f"PDFs only: {len(pdf_docs)} pages")

    # Load a single specific file
    available = [
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".pdf")
    ]
    if available:
        single_path = os.path.join(DATA_DIR, available[0])
        single_docs = SimpleDirectoryReader(
            input_files=[single_path]
        ).load_data()
        print(f"Single file ({available[0]}): "
              f"{len(single_docs)} pages")

    # Inspect metadata
    for i, doc in enumerate(documents[:2]):
        print(f"\n--- Document {i+1} ---")
        print(f"  File    : {doc.metadata.get('file_name', 'unknown')}")
        print(f"  Page    : {doc.metadata.get('page_label', 'unknown')}")
        print(f"  Length  : {len(doc.text)} characters")
        print(f"  Preview : {doc.text[:200]}...")

    return documents


# -------------------------------------------------------
# PART 3 — PDF Loading
# -------------------------------------------------------
def demo_pdf_loading():
    print("\n" + "=" * 50)
    print("PART 3 — PDF Loading")
    print("Using SimpleDirectoryReader (works in all versions)")
    print("=" * 50)

    available = [
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".pdf")
    ]
    if not available:
        print(f"No PDFs found in {DATA_DIR}")
        return

    print(f"Available PDFs: {available}")
    filename = input("Enter PDF filename to load: ").strip()
    if not filename:
        filename = available[0]

    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    documents = SimpleDirectoryReader(
        input_files=[filepath]
    ).load_data()

    print(f"\nLoaded: {filename}")
    print(f"Pages: {len(documents)}")
    for i, doc in enumerate(documents[:3]):
        print(f"\n  Page {i+1}:")
        print(f"    Metadata : {doc.metadata}")
        print(f"    Length   : {len(doc.text)} characters")
        print(f"    Preview  : {doc.text[:250]}...")


# -------------------------------------------------------
# PART 4 — Web Loading
#
# WHY 403 HAPPENS:
#   Wikipedia and many sites block the default Python requests User-Agent.
#   Some also block known bot agents or require cookies/sessions.
#
# STRATEGY (tries in order, falls back gracefully):
#   1. Wikipedia REST API  — clean plain-text, no HTML, no scraping needed
#   2. requests + full browser headers + session (handles cookies)
#   3. urllib fallback     — different underlying HTTP stack
#   4. Informative failure with helpful message
# -------------------------------------------------------
def demo_web_loading():
    print("\n" + "=" * 50)
    print("PART 4 — Web Loading")
    print("Multi-strategy loader with automatic fallback")
    print("=" * 50)

    import re
    import urllib.request

    topic   = "Insurance"
    wiki_url = f"https://en.wikipedia.org/wiki/{topic}"

    # ----------------------------------------------------------
    # Strategy 1: Wikipedia REST API (best for Wikipedia pages)
    # Returns clean plain-text — no HTML stripping needed
    # ----------------------------------------------------------
    def try_wikipedia_api(topic):
        api_url = (
            f"https://en.wikipedia.org/api/rest_v1/page/summary/"
            f"{topic}"
        )
        headers = {
            "User-Agent": "LlamaIndexLearning/1.0 (educational; contact@example.com)",
            "Accept": "application/json"
        }
        resp = requests.get(api_url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("extract", "")
        if not text:
            raise ValueError("Empty extract from API")
        return text, api_url

    # ----------------------------------------------------------
    # Strategy 2: requests with full browser session + headers
    # ----------------------------------------------------------
    def try_requests_session(url):
        session = requests.Session()
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;"
                "q=0.9,image/webp,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        })
        # First visit the main page to get cookies, then fetch the target
        session.get("https://en.wikipedia.org", timeout=10)
        resp = session.get(url, timeout=10)
        resp.raise_for_status()
        text = re.sub(r'<[^>]+>', ' ', resp.text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text, url

    # ----------------------------------------------------------
    # Strategy 3: urllib (different HTTP stack than requests)
    # ----------------------------------------------------------
    def try_urllib(url):
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) "
                    "Gecko/20100101 Firefox/125.0"
                )
            }
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        text = re.sub(r'<[^>]+>', ' ', html)
        text = re.sub(r'\s+', ' ', text).strip()
        return text, url

    # ----------------------------------------------------------
    # Try each strategy in order
    # ----------------------------------------------------------
    strategies = [
        ("Wikipedia REST API (clean JSON)",  lambda: try_wikipedia_api(topic)),
        ("requests + session + cookies",     lambda: try_requests_session(wiki_url)),
        ("urllib fallback",                  lambda: try_urllib(wiki_url)),
    ]

    doc = None
    for name, strategy in strategies:
        print(f"\n  Trying: {name} ...")
        try:
            text, source = strategy()
            doc = Document(
                text=text[:5000],
                metadata={"source": source, "type": "web_page", "strategy": name}
            )
            print(f"  ✓ Success via: {name}")
            print(f"  Characters loaded : {len(doc.text)}")
            print(f"  Preview           : {doc.text[:300]}...")
            break
        except Exception as e:
            print(f"  ✗ Failed ({name}): {e}")

    if doc is None:
        print(
            "\n  All strategies failed. Possible reasons:\n"
            "  - No internet connection\n"
            "  - Corporate proxy / firewall blocking outbound HTTP\n"
            "  - Wikipedia is temporarily down\n"
            "  Try: ping en.wikipedia.org  in your terminal to check connectivity."
        )
        return

    # Try SimpleWebPageReader if installed (bonus)
    print("\n  Checking for SimpleWebPageReader...")
    try:
        from llama_index.readers.web import SimpleWebPageReader
        loader = SimpleWebPageReader(html_to_text=True)
        web_docs = loader.load_data(urls=[wiki_url])
        print(f"  SimpleWebPageReader also works: {len(web_docs)} doc(s)")
        print(f"  Preview: {web_docs[0].text[:200]}...")
    except ImportError:
        print("  SimpleWebPageReader not installed (optional).")
        print("  To install: pip install llama-index-readers-web")
    except Exception as e:
        print(f"  SimpleWebPageReader failed: {e}")


# -------------------------------------------------------
# PART 5 — Custom Metadata
# -------------------------------------------------------
def demo_custom_metadata(documents):
    print("\n" + "=" * 50)
    print("PART 5 — Custom Metadata")
    print("Add your own fields to every document")
    print("=" * 50)

    if not documents:
        print("No documents loaded. Run Part 2 first.")
        return

    for doc in documents:
        doc.metadata["project"] = "LlamaIndex Learning"
        doc.metadata["author"] = "Sumith"
        doc.metadata["category"] = "document"
        doc.metadata["indexed_by"] = "day3_script"
        doc.excluded_llm_metadata_keys = [
            "file_path", "creation_date", "last_modified_date"
        ]
        doc.excluded_embed_metadata_keys = [
            "file_path", "creation_date"
        ]

    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=20)
    nodes = splitter.get_nodes_from_documents(documents)

    print(f"Nodes created: {len(nodes)}")
    if nodes:
        print(f"\nSample node metadata:")
        for key, val in nodes[0].metadata.items():
            print(f"  {key}: {val}")
        print(f"\nSample node text preview:")
        print(f"  {nodes[0].text[:200]}...")


# -------------------------------------------------------
# PART 6 — IngestionPipeline
#
# FIX 1: Cap to MAX_PIPELINE_DOCS (default 3) to avoid Groq TPM rate limits.
#         TitleExtractor + KeywordExtractor each make one LLM call per document.
#         With 35 docs that's 70 calls at once — easily busts the free tier 6k TPM.
#
# FIX 2: num_workers=1 forces sequential (not parallel) LLM calls.
#
# FIX 3: Added per-doc sleep delay to further respect the rate limit window.
#
# If you want to process ALL documents, set MAX_PIPELINE_DOCS = None and
# switch to LM Studio (local) which has no rate limits.
# -------------------------------------------------------
MAX_PIPELINE_DOCS = 3      # Set to None to process all docs (local LLM recommended)
INTER_DOC_DELAY   = 1.0    # Seconds to wait between documents (increase if still hitting 429)


def demo_ingestion_pipeline(documents):
    print("\n" + "=" * 50)
    print("PART 6 — IngestionPipeline")
    print("Load + Chunk + Extract metadata in one step")
    print("=" * 50)

    if not documents:
        print("No documents loaded. Run Part 2 first.")
        return

    # FIX 1: Limit docs to avoid rate-limit hammering
    if MAX_PIPELINE_DOCS is not None and len(documents) > MAX_PIPELINE_DOCS:
        print(f"[Rate-limit guard] Processing first {MAX_PIPELINE_DOCS} of "
              f"{len(documents)} documents.")
        print(f"  → To process all docs, set MAX_PIPELINE_DOCS = None "
              f"and use LM Studio (local).")
        sample_docs = documents[:MAX_PIPELINE_DOCS]
    else:
        sample_docs = documents

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(
                chunk_size=512,
                chunk_overlap=20
            ),
            TitleExtractor(),
            KeywordExtractor(keywords=5),
        ]
    )

    print(f"Running ingestion pipeline on {len(sample_docs)} document(s)...")
    print(f"  num_workers=1  (sequential, avoids parallel rate-limit bursts)")

    all_nodes = []

    # FIX 2 & 3: Process one doc at a time with a small delay between each
    for i, doc in enumerate(sample_docs):
        print(f"\n  [{i+1}/{len(sample_docs)}] Processing: "
              f"{doc.metadata.get('file_name', 'doc')} "
              f"page {doc.metadata.get('page_label', '?')}")
        try:
            # num_workers=1 → sequential LLM calls, no parallel bursts
            nodes = pipeline.run(documents=[doc], num_workers=1)
            all_nodes.extend(nodes)
            print(f"    → {len(nodes)} node(s) produced")
        except Exception as e:
            print(f"    ✗ Failed: {e}")

        # FIX 3: Pause between documents to let the TPM window recover
        if i < len(sample_docs) - 1:
            time.sleep(INTER_DOC_DELAY)

    print(f"\nTotal nodes produced: {len(all_nodes)}")
    for i, node in enumerate(all_nodes[:3]):
        print(f"\n  Node {i+1}:")
        print(f"    Title    : {node.metadata.get('document_title', 'none')}")
        print(f"    Keywords : {node.metadata.get('excerpt_keywords', 'none')}")
        print(f"    Preview  : {node.text[:200]}...")


# -------------------------------------------------------
# PART 7 — Manual Document Creation
# -------------------------------------------------------
def demo_manual_document():
    print("\n" + "=" * 50)
    print("PART 7 — Manual Document Creation")
    print("Create Documents in code without loading files")
    print("=" * 50)

    docs = [
        Document(
            text=(
                "Insurance is a means of protection from financial loss. "
                "It is a form of risk management primarily used to hedge "
                "against the risk of a contingent or uncertain loss."
            ),
            metadata={
                "source": "manual",
                "topic": "insurance_basics",
                "author": "Sumith"
            }
        ),
        Document(
            text=(
                "A deductible is the amount you pay out of pocket "
                "before your insurance company pays a claim. "
                "Higher deductibles mean lower premiums."
            ),
            metadata={
                "source": "manual",
                "topic": "deductibles",
                "author": "Sumith"
            }
        ),
    ]

    print(f"Created {len(docs)} manual documents")
    for i, doc in enumerate(docs):
        print(f"\n  Document {i+1}:")
        print(f"    ID       : {doc.doc_id}")
        print(f"    Metadata : {doc.metadata}")
        print(f"    Preview  : {doc.text[:150]}...")

    splitter = SentenceSplitter(chunk_size=256, chunk_overlap=20)
    nodes = splitter.get_nodes_from_documents(docs)
    print(f"\nNodes from manual docs: {len(nodes)}")


# -------------------------------------------------------
# Demo selector menu
# -------------------------------------------------------
def demo_menu():
    documents = []

    while True:
        print("\n" + "=" * 50)
        print("Day 3 — Select Demo")
        print("=" * 50)
        print("  1 - Embeddings + Cosine Similarity")
        print("  2 - SimpleDirectoryReader")
        print("  3 - PDF Loading")
        print("  4 - Web Loading")
        print("  5 - Custom Metadata")
        print("  6 - IngestionPipeline")
        print("  7 - Manual Document Creation")
        print("  8 - Run ALL")
        print("  0 - Exit")
        print("=" * 50)

        choice = input("Enter choice: ").strip()

        if choice == "1":
            demo_embeddings()
        elif choice == "2":
            documents = demo_simple_directory_reader()
        elif choice == "3":
            demo_pdf_loading()
        elif choice == "4":
            demo_web_loading()
        elif choice == "5":
            demo_custom_metadata(documents)
        elif choice == "6":
            demo_ingestion_pipeline(documents)
        elif choice == "7":
            demo_manual_document()
        elif choice == "8":
            demo_embeddings()
            documents = demo_simple_directory_reader()
            demo_pdf_loading()
            demo_web_loading()
            demo_custom_metadata(documents)
            demo_ingestion_pipeline(documents)
            demo_manual_document()
            print("\n" + "=" * 50)
            print("Day 3 Complete!")
            print("=" * 50)
        elif choice == "0":
            print("Exiting Day 3.")
            break
        else:
            print("Invalid. Enter 0-8.")


def main():
    use_local = show_menu()
    configure(use_local=use_local)
    demo_menu()


if __name__ == "__main__":
    main()