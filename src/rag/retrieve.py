import os
from llama_index.core import (
    VectorStoreIndex,
    Settings
)

from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()




def retrieve_from_vdb(query: str, collection_name:str):

    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Set the LLM
    Settings.llm = Ollama(model="llama3.2", request_timeout=600)

    # Set up Qdrant client
    qdrant_client = QdrantClient(
    os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"), 
    )

    # Set up vector store
    vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name)

    index = VectorStoreIndex.from_vector_store(vector_store)

    ### Trial from here 

    ### Approach 1 : 

    query_engine = index.as_query_engine(
    similarity_top_k=5
)
    
    response = query_engine.query(
    f"{query}")

    return response

    