##----------Loading all the required modules for the Setting up the model and embedding model---------##
import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
##----------Loading the environment file (.env) for the Groq API Key
load_dotenv()
##----------Defining the function for Llama Index Settings---------##
def configure(use_local=False):
    if use_local:
        llm=OpenAILike(
    # --- LM Studio (fully local, no API key needed) ---#
    # Make sure LM Studio is running and a model is loaded---#
    # Default server: http://localhost:1234/v1-----#
        model="mistralai/ministral-3-3b",
        api_base="http://localhost:1234/v1",
        api_key="Not Needed",
        is_chat_model=True,
        temperature=0.7,
        max_tokens=1024
        )
        llm_name="LM Studio / mistralai/ministral-3-3b (local)"
    else:
    #------Groq Cloud LLM Provider--------#
        llm=Groq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.3,
            max_tokens=1024
        )
        llm_name="Groq / llama-3.1-8b-instant"
    # --- Embeddings: HuggingFace (always local, no API key needed) ---
    # all-MiniLM-L6-v2:
    #   - 384 dimensional vectors
    #   - runs on CPU, no GPU needed
    #   - good quality for RAG use cases
    embed_model=HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
#----- Configuring the settings of the LLM's-------------#
    Settings.llm=llm
    Settings.embed_model=embed_model
    Settings.chunk_size=512        # tokens per chunk 
    Settings.chunk_overlap = 50    # overlap keeps context at chunk boundaries
#---------Checking the requirements by using the print statement-------------#
    print("=" * 45)
    print("LlamaIndex Settings Configured")
    print("=" * 45)
    print(f"  LLM          : {llm_name}")
    print(f"  Embeddings   : all-MiniLM-L6-v2 (local)")
    print(f"  Chunk size   : {Settings.chunk_size} tokens")
    print(f"  Chunk overlap: {Settings.chunk_overlap} tokens")
    print("=" * 45)
if __name__=="__main__":
    configure(use_local=False)