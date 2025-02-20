import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("product-index")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample query
query = "Show me products under $100"
query_embedding = model.encode(query).tolist()

# Query the index
response = index.query(vector=query_embedding, top_k=5, include_metadata=True)

# Print the response
print("✅ Retrieved Products:", response)