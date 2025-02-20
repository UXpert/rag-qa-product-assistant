import json
import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "product-index"

# Connect to the existing index
index = pc.Index(index_name)

# Load products from JSON
with open("products.json", "r") as file:
    products = json.load(file)

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings and metadata
vectors = [
    {
        "id": str(product["id"]),
        "values": embedding_model.encode(product["description"]).tolist(),
        "metadata": {
            "title": product["title"],
            "description": product["description"],
            "price": float(product["price"]),
            "image": product["image"],
            "category": product["category"],
            "rating_rate": float(product["rating"]["rate"]),
            "rating_count": int(product["rating"]["count"])
        }
    }
    for product in products
]

# Upload vectors to Pinecone
index.upsert(vectors=vectors)
print("✅ Embeddings created and uploaded successfully.")