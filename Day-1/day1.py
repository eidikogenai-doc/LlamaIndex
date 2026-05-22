# Day 1 — Introduction & Architecture + LLMs & Settings Configuration
# What this script does:
#   1. Configures LLM and embeddings via Settings
#   2. Loads your insurance PDF as Documents
#   3. Builds a VectorStoreIndex (in-memory)
#   4. Creates a QueryEngine
#   5. Asks a question and prints the answer with source nodes
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader 
"""Here VectorStoreIndex is the in-built Vector DB available in LlamaINdex 
and SimpleDirectoryReader helps to find the files """
from settings import configure #---------Importing the settings configuration of the settings.py----------#
def get_user_choice(): #-------Making the choice between Groq and LM Studio--------------#
    print("Select LLM Provider:")
    print("  1 - Groq (cloud, fast)")
    print("  2 - LM Studio (local, offline)")
    """We are using while loop because to run the code continuously until the choice is made"""
    while True:
        choice=int(input("Enter your choice (1 or 2): ")) 
        if choice==1:
            return False
        elif choice==2:
            return True
        else:
            print("Invalid Choice Made Try Again")
#-----Getting the query/question from the user-------------#
def get_user_question():
    question=input("Enter the question: ").strip() #-----Strip is used to remove the spaces before and after the choices
    return question
#--------Main Function----------#
def main():
    #........Step-1: User should select the LLM Provider...........#
    use_local=get_user_choice()
    configure(use_local=use_local)
    #........Step-2: Load the documents.........#
    print("Loading Documents....")
    documents=SimpleDirectoryReader("data").load_data()
    print("Loaded Docuements......")
    # ........Inspect first 2 documents.......#
    for i, doc in enumerate(documents[:2]):
        print(f"\n--- Document {i+1} ---")
        print(f"  ID       : {doc.doc_id}")
        print(f"  Metadata : {doc.metadata}")
        print(f"  Preview  : {doc.text[:300]}...")
    # Step 3: Build index
    print("\nBuilding index.......")
    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True
    )
    print("Index built successfully")
    # Step 4: Create QueryEngine
    query_engine = index.as_query_engine(
        similarity_top_k=3
    )
    # Step 5: Query loop — keep asking until user types 'exit'
    print("\nType 'exit' to quit.")
    while True:
        question = get_user_question()
        if question.lower() == "exit":
            print("Exiting... Goodbye...!!!")
            break
        print("\nQuerying...")
        response = query_engine.query(question)
        print(f"\nAnswer : {response}")
        print(f"\nSource nodes used ({len(response.source_nodes)}):")
        for i, node in enumerate(response.source_nodes):
            print(f"\n  Node {i+1}:")
            print(f"    Score   : {node.score:.4f}")
            print(f"    Source  : {node.metadata.get('file_name', 'unknown')}")
            print(f"    Preview : {node.text[:200]}...")
if __name__ == "__main__":
    main()