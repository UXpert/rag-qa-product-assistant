import faiss
import json
from sentence_transformers import SentenceTransformer
import numpy as np

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load product data from products.json
with open('products.json', 'r') as file:
    products = json.load(file)

# Extract product descriptions
product_descriptions = [product["description"] for product in products]

# Generate embeddings for product descriptions
embeddings = model.encode(product_descriptions)

# Convert embeddings to a numpy array (FAISS requirement)
embeddings = np.array(embeddings).astype('float32')

# Create a FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add embeddings to the index
index.add(embeddings)

# Save the FAISS index to disk
faiss.write_index(index, "faiss_index.index")

# Save product metadata for reference during retrieval
with open("product_metadata.json", "w") as f:
    json.dump(products, f)

print("✅ FAISS index and product metadata saved successfully!")