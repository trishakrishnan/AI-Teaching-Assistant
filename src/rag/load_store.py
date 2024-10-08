import os
from dotenv import load_dotenv


from llama_index.core.ingestion import IngestionPipeline

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from llama_parse import LlamaParse
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    Document
)

from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor
)
#from llama_index.extractors.entity import EntityExtractor

load_dotenv()

def vectorise_transcript(transcript_path, collection_name):

    # Set the embedding model
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


    ### Define Extractors 

    extractors = [
    TitleExtractor(nodes=5, llm=Settings.llm),
    QuestionsAnsweredExtractor(questions=3, llm=Settings.llm),
    #EntityExtractor(prediction_threshold=0.5),
    SummaryExtractor(summaries=["prev", "self"], llm=Settings.llm),
    KeywordExtractor(keywords=10, llm=Settings.llm)
]



    # Set the embedding model
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Set the LLM
    Settings.llm = Ollama(model="llama3.2", request_timeout=600)

    # Build the index from the vectorized nodes
    index = VectorStoreIndex(documents, use_metadata=True,storage_context=storage_context, transformations = extractors)

    success_message = "Transcript successfully vectorised"

    return success_message

