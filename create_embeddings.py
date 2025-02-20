import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Initialize Chroma client with the new configuration
client = chromadb.PersistentClient(path="./chroma_db")

# Create a collection for products
collection = client.get_or_create_collection(name="products")

# Example product data (simulated for now)
products = [
    {"id": "1", "name": "Wireless Headphones", "description": "Noise-cancelling over-ear headphones with 30-hour battery life."},
    {"id": "2", "name": "Smartphone", "description": "Latest model smartphone with OLED display and dual cameras."},
    {"id": "3", "name": "Laptop", "description": "15-inch laptop with 16GB RAM and 512GB SSD storage."}
]

# Load the SentenceTransformer model to generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings and store them in ChromaDB
for product in products:
    embedding = model.encode(product['description']).tolist()
    collection.add(
        documents=[product['description']],
        embeddings=[embedding],
        ids=[product['id']]
    )

print("✅ Successfully added embeddings for products to the database.")