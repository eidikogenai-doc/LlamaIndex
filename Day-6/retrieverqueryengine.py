from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever

index = 0
retriever= VectorIndexRetriever(
    similarity_top_k=3,
    index=index
)

synthezier=get_response_synthesizer(
    response_mode="compact"
)

query_engine=RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=synthezier
)