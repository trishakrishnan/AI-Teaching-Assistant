from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

load_dotenv()

def access_collections():
    qdrant_client = QdrantClient(
        os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"), 
        )

    print(qdrant_client.get_collections())


def delete_collections(collection_name):

    qdrant_client = QdrantClient(
    os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"), 
    )

    qdrant_client.delete_collection(collection_name=f"{collection_name}")



"""

for i in ['trisha_Attention is all you need.pdf', 'test_topic_chunks', 'test2', 'test', 'qdrant_rag']:
    delete_collections(i)

access_collections() """