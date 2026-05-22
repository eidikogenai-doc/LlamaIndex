from llama_index.core.node_parser import NodeParser
from llama_index.core.schema import TextNode,BaseNode
from typing import List,Sequence
class ParagraphSplitter(NodeParser):
    """Splits documents on paragraph boundaries (double newlines)."""
    def _parse_nodes(
    self,
    nodes: Sequence[BaseNode],
    show_progress: bool = False,
) -> List[BaseNode]:
        all_nodes=[]

        for node in nodes:
            paragraphs=node.text.split("\n\n")
            for i,para in enumerate(paragraphs):
                para=para.strip()
                if not para:  # skip empty paragraphs
                    continue

                new_node = TextNode(
                    text=para,
                    metadata=node.metadata.copy(),  # inherit parent metadata
                )

                # Attach relationship back to source
                new_node.relationships = node.relationships.copy()

                all_nodes.append(new_node)

        return all_nodes