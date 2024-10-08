from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import PunktSentenceTokenizer
from typing import List, Dict


from src.rag.config import TRANSCRIPT, TOPIC_CHUNKING_PROMPT


from custom_groq_llm import GroqLLM  # Use OpenAI or relevant LLM connector
from langchain.prompts import PromptTemplate
import re
import ast
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
import re
import json

from rag.metadata_class import ChunkOutput
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from itertools import groupby


# Initialize the LLM 
llm = GroqLLM()  


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



def chunking_with_llms(transcript):

    output_parser = PydanticOutputParser(pydantic_object = ChunkOutput)

    format_instructions = output_parser.get_format_instructions()

    llm = GroqLLM(system_message = "The assistant must generate only the JSON as per the requested format")  


    prompt = PromptTemplate(
    input_variables=["transcript"],
    template=TOPIC_CHUNKING_PROMPT,
    partial_variables = {"format_instructions": format_instructions}
)

    formatted_prompt = prompt.format_prompt(transcript=transcript)

    output = llm.invoke(formatted_prompt.to_string())

    json_match = re.search(r'(\[.*\])', output, re.DOTALL)

    json_string = json_match.group(1)

    print("JSON String:",json_string)
    json_output = extract_json(json_string)
    
    return  json_output
   


def chunk_transcript_with_lda(transcript, num_topics=5):
    """
    Splits a transcript into chunks based on high-level thematic topics using LDA for topic modeling.

    Parameters:
    - transcript (str): The full transcript text as a single string.
    - num_topics (int): Number of topics or clusters to find.

    Returns:
    - Dictionary where keys are thematic topic labels and values are the corresponding text content.
    """
    # Load the SpaCy language model
    nlp = spacy.load("en_core_web_sm")

    # Preprocess the transcript into sentences
    doc = nlp(transcript)
    sentences = [sent.text.strip() for sent in doc.sents]

    # Remove stopwords using NLTK
    stop_words = set(stopwords.words('english'))
    processed_sentences = [" ".join([word for word in sent.lower().split() if word not in stop_words])
                           for sent in sentences]

    # Use CountVectorizer to vectorize the sentences
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    sentence_vectors = vectorizer.fit_transform(processed_sentences)

    # Apply LDA to discover topics in the transcript
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(sentence_vectors)
    topic_labels = lda.transform(sentence_vectors).argmax(axis=1)

    # Group the sentences by their assigned topic labels
    theme_chunks = {}
    for topic_label, group in groupby(zip(topic_labels, sentences), lambda x: x[0]):
        grouped_sentences = [sentence for _, sentence in group]
        chunk_text = " ".join(grouped_sentences)

        # Extract the theme keyword for the topic
        topic_words = lda.components_[topic_label]
        top_word_indices = topic_words.argsort()[-1:]  # Get the most significant keyword index
        feature_names = vectorizer.get_feature_names_out()
        theme = feature_names[top_word_indices[0]]

        # Add the chunk to the dictionary with the theme as key
        theme_chunks[theme] = chunk_text

    return theme_chunks
\

print(chunk_transcript_with_lda(TRANSCRIPT))



""" Plan : 1) Semantic Segmentation
            2) Topic modelling based keyword chunking - 5 topics per semantic chunk
            3) Named entity tagging for each chunk to be stored as metadata 

"""