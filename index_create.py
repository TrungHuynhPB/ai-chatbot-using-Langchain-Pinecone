from pinecone import Pinecone, ServerlessSpec
import os
import sys
from dotenv import load_dotenv
# -------------------------------
# Load environment
# -------------------------------   
load_dotenv()

# -------------------------------
# Get filename argument
# -------------------------------
if len(sys.argv) < 2:
    print("Usage: python -m index_create <index_name>")
    sys.exit(1)

index_name = sys.argv[1]

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", index_name)
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
DATA_DIR = "data"

# -------------------------------
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
dimension = 1536 # OpenAI embedding (text-embedding-3-small)
metric = "cosine" # "cosine","euclidean", "dotproduct", etc..

if not pc.has_index(index_name):
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        vector_type="dense",
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        deletion_protection="disabled"
    )
    print(f"Index '{index_name}' created successfully.")
else:
    print(f"Index '{index_name}' already exists.")
