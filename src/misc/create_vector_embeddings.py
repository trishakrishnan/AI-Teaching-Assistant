import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    Document
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

from chunking_utils import chunk_sentences_by_size, map_topics_keywords



def vectorise_transcript(transcript_path, collection_name):

    # Set up Qdrant client
    qdrant_client = QdrantClient(
    os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"), 
    )

    # Set up vector store
    vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name)


    # Set up storage context
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    parser = LlamaParse(
    result_type="text"
)

    # Define file extractor for .pdf using the parser
    file_extractor = {".txt": parser}

    # Load documents from the specified directory
    documents = SimpleDirectoryReader(input_dir=f"{transcript_path}", file_extractor=file_extractor).load_data()


    # Chunk documents 

    all_nodes = []

    for document in documents:
        text = document.text
        document_chunks = chunk_sentences_by_size(text)

        for chunk in document_chunks:
            topics,keywords = map_topics_keywords(chunk)
            all_nodes.append(Document(text=chunk, metadata={
                "topics": topics,
                "keywords": keywords
            }))

    # Set the embedding model
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Set the LLM
    Settings.llm = Ollama(model="llama3.2", request_timeout=600)

    # Build the index from the vectorized nodes
    index = VectorStoreIndex(all_nodes, use_metadata=True,storage_context=storage_context)

    success_message = "Transcript successfully vectorised"

    return success_message



print(vectorise_transcript("/Users/trishakrishnan/Documents/GitHub/AI-Teaching-Assistant/src/rag/","test_topic_chunks"))