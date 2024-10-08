TRANSCRIPT = """
What exactly is generative AI? 
When new content is created
by artificial intelligence, 
it’s called generative AI. 
This could involve
generating texts and images, 
as well as videos, music, or voices. 
To do this, you describe in a chat
dialogue box what you want the AI to create. 
This description is called a "prompt".  
The generative AI tools 
provide answers to all sorts of questions, 
summarise complex information,
and generate diverse ideas quickly. 
Depending on how they're used,
they can create short stories, paintings, 
pieces of code,
or even musical compositions. 
The foundation for this creation
lies in large amounts of data 
that the AI system accesses
to identify patterns and similarities. 
The content produced by the AI is new. 
It's often impressive and challenging to
distinguish from things humans have made. 
Generative AI 
can also be misused: in so-called "deepfakes" 
AI is utilised to produce 
images or videos 
that seem real. 
AI-generated 
texts are also tough to recognise
as machine-made. 
Moreover, the 
AI can provide answers that sound correct
but are actually incorrect – 
this is called "hallucinating". 
The quality of what's created depends on
both the quality of the data used 
and the quality of the prompts given. 
To effectively utilise generative AI, 
we need to learn how to guide the tools
with meaningful prompts 
and use them thoughtfully. 
Generative AI holds immense 
potential and can help us in many ways – 
such as serving as a writing
or learning partner. 
However, the AI should do the hard work,
and humans should 
be responsible for the facts. 
"""




METADATA_PROMPT = """
Identify the topics of discussion ( upto 5) and keywords ( named entities, technical concepts, key phrases based on frequency) within the text given below - 

Input text : {transcript}


Generate an output in the following JSON format:
[
    {{
        "topics": [List of discussion topics (phrases) present in the text ],
        "keywords": [list of keywords in text]
    }},
]

Example - 
Input Text :
What exactly is generative AI? 
When new content is created
by artificial intelligence, 
it’s called generative AI. 
This could involve
generating texts and images, 
as well as videos, music, or voices. 
To do this, you describe in a chat
dialogue box what you want the AI to create. 
This description is called a "prompt".  
The generative AI tools 
provide answers to all sorts of questions, 
summarise complex information,
and generate diverse ideas quickly. 
Depending on how they're used,
they can create short stories, paintings, 
pieces of code,
or even musical compositions. 

Output : 
[
    {{
        "topics": ["Definition of Generative AI","The Concept of a Prompt","Capabilities of Generative AI Tools","Practical Applications of Generative AI"]
        "keywords": ["Generative AI","Artificial Intelligence","Prompt","Tools"]
    }},
]


IMPORTANT: Output only the JSON. Do not include any text before or after the JSON.
"""



METADATA_SYSTEM_MESSAGE = """You help identify discussion topics and keywords within the text provided to you. 

IMPORTANT: Output only the JSON. Do not include any text before or after the JSON.


"""