
from typing import List
from pydantic import BaseModel
from llama_index.llms.openai.utils import to_openai_tool
from llama_index.core.tools import FunctionTool


class ChunkMetadata(BaseModel):
    """A song with name and artist"""
    topics: List[str]
    keywords: List[str]


def genetate_metadata(topics: List[str], keywords: List[str]) -> ChunkMetadata:
    """Generates metadata for given text."""
    return ChunkMetadata(topics=topics, keywords=keywords)


