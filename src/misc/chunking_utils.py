

from src.rag.config import  METADATA_PROMPT
import re
import json
from rag.metadata_class import ChunkMetadata
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI



# Initialize the LLM 
llm=OpenAI(model="gpt-4o")


def clean_transcript(transcript):
    # Use regex to remove double quotes around words or phrases
    cleaned_transcript = transcript.replace('"', '')

    cleaned_transcript_newline = transcript.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')
    
    return cleaned_transcript_newline

def extract_json(output: str):
    # Attempt to locate the JSON part using regex
    json_match = re.search(r'(\[.*\])', output, re.DOTALL)
    if json_match:
        json_string = json_match.group(1)
        return json.loads(json_string)
    else:
        raise ValueError("JSON output not found in the LLM response")



def chunk_sentences_by_size(transcript):

    text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    section_nodes = text_splitter.split_text(transcript)

    return section_nodes



def map_topics_keywords(chunk):

    llm = OpenAI(model="gpt-4o-mini")

    output = (
    llm.as_structured_llm(ChunkMetadata)
    .complete(METADATA_PROMPT.format(transcript=chunk))
    .raw
)

    return output.topics,output.keywords

