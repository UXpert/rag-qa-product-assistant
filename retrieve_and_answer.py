import chromadb
from chromadb.config import Settings
import openai
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables from .env
load_dotenv()

# Initialize OpenAI client with your API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Correct client initialization for ChromaDB
client = chromadb.Client()
collection = client.get_collection(name="products")

# Check 'product_collection'
product_collection = client.get_collection(name="product_collection")
product_docs = product_collection.get()
print("📄 Documents in 'product_collection':", product_docs)

# Check 'products'
products_collection = client.get_collection(name="products")
products_docs = products_collection.get()
print("📄 Documents in 'products':", products_docs)

def retrieve_products(query: str, top_k: int = 3):
    """Retrieve the top k most relevant products from ChromaDB based on the query."""
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results["documents"][0] if results["documents"] else []

def generate_answer(user_query: str, retrieved_docs: list):
    """Generate a natural language answer using OpenAI and the retrieved documents."""
    if not retrieved_docs:
        return "Sorry, I couldn't find any relevant products."

    context = "\n".join(retrieved_docs)

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # Use "gpt-4" if available
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides information about products."},
            {"role": "user", "content": f"Context: {context}\nQuestion: {user_query}\nAnswer:"}
        ],
        temperature=0.5,
        max_tokens=150
    )

    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    # Example usage:
    query = input("🔎 Enter your product question: ")
    retrieved_docs = retrieve_products(query)
    answer = generate_answer(query, retrieved_docs)
    print("\n🤖 Answer:", answer)