from llama_index.core import SimpleDirectoryReader
from ParagraphSplitter import ParagraphSplitter
from llama_index.core.node_parser import (
    SentenceSplitter,
    TokenTextSplitter,
    SemanticSplitterNodeParser,
    SentenceWindowNodeParser,
    HierarchicalNodeParser,
    MarkdownNodeParser,
    JSONNodeParser,
    CodeSplitter,
    SimpleFileNodeParser
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# ─────────────────────────────────────────
# STEP 1 — Load your PDF
# ─────────────────────────────────────────
documents = SimpleDirectoryReader("data").load_data()
print(f"\nLoaded {len(documents)} document(s)")
print(f"Total text length: {sum(len(d.text) for d in documents)} characters")
# ─────────────────────────────────────────
# STEP 2 — Custom Splitter (Paragraph)
# ─────────────────────────────────────────
# class ParagraphSplitter(NodeParser):
#     def _parse_nodes(
#         self,
#         nodes: Sequence[BaseNode],
#         show_progress: bool = False
#     ) -> List[BaseNode]:
#         all_nodes = []
#         for node in nodes:
#             paragraphs = node.text.split("\n\n")
#             for para in paragraphs:
#                 para = para.strip()
#                 if not para:
#                     continue
#                 new_node = TextNode(
#                     text=para,
#                     metadata=node.metadata.copy()
#                 )
#                 all_nodes.append(new_node)
#         return all_nodes

# ─────────────────────────────────────────
# STEP 3 — Define all splitters
# ─────────────────────────────────────────
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
splitters = {
    "SentenceSplitter": SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50
    ),
    "TokenTextSplitter": TokenTextSplitter(
        chunk_size=512,
        chunk_overlap=50
    ),
    "SemanticSplitter": SemanticSplitterNodeParser(
        embed_model=embed_model,
        breakpoint_percentile_threshold=95,
        buffer_size=1
    ),
    "SentenceWindowNodeParser": SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text"
    ),
    "HierarchicalNodeParser": HierarchicalNodeParser.from_defaults(
        chunk_sizes=[2048, 512, 128]
    ),
    "SimpleFileNodeParser": SimpleFileNodeParser(),
    "ParagraphSplitter (Custom)": ParagraphSplitter(),
}
# ─────────────────────────────────────────
# STEP 4 — Run all splitters and compare
# ─────────────────────────────────────────
print("\n" + "="*60)
print("SPLITTER COMPARISON RESULTS")
print("="*60)
for name, splitter in splitters.items():
    try:
        nodes = splitter.get_nodes_from_documents(documents)
        # Calculate avg chunk size
        avg_size = sum(len(n.text) for n in nodes) / len(nodes) if nodes else 0
        print(f"\n--- {name} ---")
        print(f"  Total nodes produced : {len(nodes)}")
        print(f"  Avg chunk size (chars): {avg_size:.0f}")
        print(f"  Sample chunk preview  : {nodes[0].text[:120].strip()}...")
        # Show window metadata for SentenceWindow
        if name == "SentenceWindowNodeParser":
            print(f"  Window context sample : {nodes[0].metadata.get('window', '')[:150]}...")
        # Show hierarchy for Hierarchical
        if name == "HierarchicalNodeParser":
            print(f"  Relationship keys     : {list(nodes[0].relationships.keys())}")
        # Show custom splitter metadata
        if name == "ParagraphSplitter (Custom)":
            print(f"  Metadata on node      : {nodes[0].metadata}")
    except Exception as e:
        print(f"\n--- {name} ---")
        print(f"  Skipped: {e}")
# ─────────────────────────────────────────
# STEP 5 — Deep inspect one splitter
# ─────────────────────────────────────────
print("\n" + "="*60)
print("DEEP INSPECT — SentenceSplitter Node Details")
print("="*60)
main_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
main_nodes = main_splitter.get_nodes_from_documents(documents)
for i, node in enumerate(main_nodes[:3]):  # first 3 nodes only
    print(f"\nNode {i}")
    print(f"  node_id    : {node.node_id}")
    print(f"  text length: {len(node.text)} chars")
    print(f"  metadata   : {node.metadata}")
    print(f"  relations  : {list(node.relationships.keys())}")
    print(f"  text preview: {node.text[:150].strip()}")