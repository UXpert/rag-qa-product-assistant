import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
from dotenv import load_dotenv
import os

# Load environment variables from .env (for OpenAI API key)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the FAISS index
index = faiss.read_index("faiss_index.index")

# Load product metadata
with open("product_metadata.json", "r") as file:
    product_metadata = json.load(file)


def retrieve_products(query: str, top_k: int = 3):
    """Retrieve the top_k most relevant products using FAISS."""
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    retrieved_products = [product_metadata[i] for i in indices[0] if i != -1]
    return retrieved_products


def generate_answer(user_query: str, retrieved_docs: list):
    """Generate a natural language answer using OpenAI and the retrieved documents."""
    if not retrieved_docs:
        return "❌ Sorry, I couldn't find any relevant products."

    context = "\n".join([f"- {doc['description']}" for doc in retrieved_docs])

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # Use "gpt-4" if available
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides product information."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}\nAnswer:"},
        ],
        temperature=0.5,
        max_tokens=150,
    )

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    query = input("🔎 Enter your product question: ")
    retrieved_docs = retrieve_products(query)
    answer = generate_answer(query, retrieved_docs)
    print(f"\n🤖 Answer: {answer}")