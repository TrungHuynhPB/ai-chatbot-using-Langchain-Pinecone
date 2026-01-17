import os
import sys
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

# -------------------------------
# Load environment
# -------------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "langchain-chatbot")
DATA_DIR = "data"

# -------------------------------
# Get filename argument
# -------------------------------
if len(sys.argv) < 2:
    print("Usage: python -m indexing.py <filename>")
    sys.exit(1)

filename = sys.argv[1]
filepath = os.path.join(DATA_DIR, filename)

if not os.path.exists(filepath):
    print(f"File not found: {filepath}")
    sys.exit(1)

# -------------------------------
# Load single document
# -------------------------------
loader = DirectoryLoader(DATA_DIR, glob=filename)
documents = loader.load()

# -------------------------------
# Manual text chunking
# -------------------------------
def chunk_text(text, chunk_size=500, chunk_overlap=20):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

docs_chunks = []
for doc in documents:
    text_chunks = chunk_text(doc.page_content)
    for chunk in text_chunks:
        docs_chunks.append(chunk)

# -------------------------------
# Embeddings
# -------------------------------
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)
# -------------------------------
# Initialize Pinecone
# -------------------------------
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
index = pinecone_client.Index(PINECONE_INDEX_NAME)

# -------------------------------
# Upsert chunks into Pinecone
# -------------------------------
vectors = []
for i, chunk in enumerate(docs_chunks):
    vector = embeddings.embed_query(chunk)
    vectors.append(
        {
            "id": f"{filename}-{i}",
            "values": vector,
            "metadata": {"text": chunk, "source": filename},
        }
    )
    if len(vectors) >= 100:  # batch upsert every 100 vectors
        index.upsert(vectors=vectors)
        vectors = []

if vectors:
    index.upsert(vectors=vectors)

print(f"Indexed {len(docs_chunks)} chunks from '{filename}' into Pinecone '{PINECONE_INDEX_NAME}'.")
