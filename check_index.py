import os
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "product-index"
index = pc.Index(index_name)

# Check how many vectors are in the index
stats = index.describe_index_stats()
print(f"✅ Index stats: {stats}")