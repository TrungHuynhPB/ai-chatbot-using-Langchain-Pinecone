import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from pinecone import Pinecone

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "langchain-chatbot")

# -------------------------------
# Initialize Pinecone SDK
# -------------------------------
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
index = pinecone_client.Index(PINECONE_INDEX_NAME)

# -------------------------------
# Initialize embeddings
# -------------------------------
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    # With the `text-embedding-3` class
    # of models, you can specify the size
    # of the embeddings you want returned.
    # dimensions=1024
)
# -------------------------------
# Initialize Chat LLM for query refinement
# -------------------------------
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key=OPENAI_API_KEY
)

# -------------------------------
# Vector search / retrieval
# -------------------------------
def find_match(query: str, top_k: int = 2) -> str:
    """
    Perform a semantic search over Pinecone index and return top_k results as concatenated text.
    """
    # Embed the query
    query_embedding = embeddings.embed_query(query)

    # Search Pinecone
    result = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    # Concatenate retrieved texts
    matches = []
    for match in result.get("matches", []):
        text = match.get("metadata", {}).get("text", "")
        if text:
            matches.append(text)
    return "\n".join(matches)

# -------------------------------
# Query refinement
# -------------------------------
def query_refiner(conversation: str, query: str) -> str:
    """
    Refine the user query using the LLM to make it more suitable for semantic search.
    """
    prompt = f"""
Given the following conversation log and user query,
rewrite the query to be optimal for semantic search in a knowledge base.

Conversation log:
{conversation}

User query:
{query}

Refined query:
"""
    response = llm.invoke(prompt)
    return response.content.strip()

# -------------------------------
# Conversation history helper
# -------------------------------
def get_conversation_string() -> str:
    """
    Returns the conversation string from Streamlit session state in format:
    Human: ...
    Bot: ...
    """
    conversation_string = ""
    responses = st.session_state.get("responses", [])
    requests = st.session_state.get("requests", [])
    
    for i in range(len(responses) - 1):
        human_text = requests[i] if i < len(requests) else ""
        bot_text = responses[i + 1]
        conversation_string += f"Human: {human_text}\nBot: {bot_text}\n"
    
    return conversation_string
