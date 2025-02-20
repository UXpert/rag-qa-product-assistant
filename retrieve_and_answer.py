import os
from pinecone import Pinecone
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import openai

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize clients
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "product-assistant"
index = pc.Index(index_name)

openai.api_key = OPENAI_API_KEY
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_products(query: str, top_k: int = 3):
    query_embedding = embedding_model.encode(query).tolist()
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return results['matches'] if 'matches' in results else []

def generate_answer(user_query: str, retrieved_docs: list):
    """Generate a natural language answer using OpenAI and the retrieved documents."""
    if not retrieved_docs:
        return "Sorry, I couldn't find any relevant products."

    context = "\n".join(retrieved_docs)
    client = openai.OpenAI(api_key=OPENAI_API_KEY)  # New client initialization

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Change to "gpt-4" if you have access
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides information about products."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}\nAnswer:"}
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