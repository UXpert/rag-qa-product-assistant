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

# Check if index exists
if index_name not in [idx["name"] for idx in pc.list_indexes()]:
    print(f"❌ Index '{index_name}' not found. Please create it first.")
    exit()

index = pc.Index(index_name)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Category normalization map
CATEGORY_MAP = {
    "men's clothing": "clothing",
    "women's clothing": "clothing",
    "electronics": "electronics",
    "jewelery": "jewelry",  # Correct common typo
}

def normalize_category(category):
    """Convert categories to a standard format."""
    return CATEGORY_MAP.get(category.lower().strip(), "other")

def sanitize_metadata(product):
    """Ensure metadata values are valid (string, number, boolean, or list of strings)."""
    sanitized = {
        "title": product.get("title", ""),
        "description": product.get("description", ""),
        "price": float(product.get("price", 0.0)),
        "category": normalize_category(product.get("category", "")),
        "rating_rate": float(product.get("rating", {}).get("rate", 0.0)),
        "rating_count": int(product.get("rating", {}).get("count", 0)),
        "image": product.get("image", "")  # ✅ Added this line
    }
    return sanitized

# Load product data
with open("products.json", "r") as file:
    products = json.load(file)

# Create and upload embeddings
vectors = []
for product in products:
    product_id = str(product["id"])
    metadata = sanitize_metadata(product)
    text_to_embed = f"{metadata['title']} {metadata['description']} {metadata['category']}"
    embedding = model.encode(text_to_embed).tolist()

    vectors.append({
        "id": product_id,
        "values": embedding,
        "metadata": metadata
    })

# Upsert embeddings to Pinecone
print("🚀 Uploading embeddings to Pinecone...")
index.upsert(vectors=vectors)
print("✅ Embeddings uploaded successfully!")