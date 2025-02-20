import json
import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# 🚀 Load environment variables
load_dotenv()

# 🔑 Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "product-index"

# 📦 Load products
with open("products.json", "r") as file:
    products = json.load(file)

# 🧠 Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 📝 Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Embedding size for "all-MiniLM-L6-v2"
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# 🛠️ Generate embeddings with valid metadata
vectors = [
    {
        "id": str(product["id"]),
        "values": embedding_model.encode(product["description"]).tolist(),
        "metadata": {
            "title": product["title"],
            "description": product["description"],
            "price": product["price"],
            "image": product["image"],
            "category": product["category"],
            "rating_rate": float(product["rating"]["rate"]),    # ✅ Converted to float
            "rating_count": int(product["rating"]["count"])     # ✅ Converted to int
        }
    }
    for product in products
]

# 🚀 Upload embeddings to Pinecone
index.upsert(vectors=vectors)
print("✅ Embeddings created and uploaded successfully.")