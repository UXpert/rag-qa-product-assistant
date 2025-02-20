import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables from .env
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create an index (if it doesn't exist)
index_name = "product-assistant"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Embedding dimension for "all-MiniLM-L6-v2"
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"✅ Index '{index_name}' created.")
else:
    print(f"✅ Index '{index_name}' already exists.")

# Connect to the index
index = pc.Index(index_name)

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample product data
products = [
    {"id": "1", "description": "Noise-cancelling over-ear headphones with 30-hour battery life."},
    {"id": "2", "description": "Latest model smartphone with OLED display and dual cameras."},
    {"id": "3", "description": "15-inch laptop with 16GB RAM and 512GB SSD storage."},
]

# Create embeddings and upsert into Pinecone
for product in products:
    embedding = embedding_model.encode(product["description"]).tolist()
    index.upsert([(product["id"], embedding, {"description": product["description"]})])

print("✅ Embeddings created and stored in Pinecone.")