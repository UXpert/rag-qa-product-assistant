import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "product-index"

# Check and delete existing index
if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)
    print(f"✅ Deleted existing index: {index_name}")
else:
    print("ℹ️ No existing index found. Creating a new one...")

# ✅ Create new index with 'spec' parameter
pc.create_index(
    name=index_name,
    dimension=384,  # Matches the embedding model
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Adjust region if needed
)

print(f"✅ Created new index: {index_name}")