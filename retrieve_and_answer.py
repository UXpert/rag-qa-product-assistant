import openai
from typing import List
from pinecone import Index
from sentence_transformers import SentenceTransformer

# Retrieve products from Pinecone
def retrieve_products(index: Index, query_embedding: List[float], top_k: int = 5):
    response = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return response['matches'] if 'matches' in response else []

# Re-rank retrieved products using OpenAI
def re_rank_products(query: str, products: List[dict]) -> List[dict]:
    prompt = "Rank the following products based on the query:\n"
    for product in products:
        prompt += f"Title: {product['metadata']['title']}, Description: {product['metadata']['description']}, Price: ${product['metadata']['price']}\n"

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )

    ranked_titles = [line.replace("Title: ", "").strip() for line in response.choices[0].message.content.split("\n") if line.startswith("Title:")]
    ranked_products = [product for title in ranked_titles for product in products if product['metadata']['title'] == title]

    return ranked_products