import streamlit as st
from retrieve_and_answer import retrieve_products, re_rank_products
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "product-index"
index = pc.Index(index_name)

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Streamlit UI setup
st.title("🛍️ Product Information Assistant")
st.write("Ask any question about our products (e.g., *Which product has the best battery life?*)")

query = st.text_input("Enter your product question:")

if query:
    query_embedding = embedding_model.encode(query).tolist()

    # Retrieve products from Pinecone
    retrieved_products = retrieve_products(index, query_embedding)

    if retrieved_products:
        # Re-rank the retrieved products
        ranked_products = re_rank_products(query, retrieved_products)

        st.subheader("📦 Retrieved Products:")
        for product in ranked_products:
            metadata = product["metadata"]
            st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 10px; background-color: #2C2C2C; padding: 10px; border-radius: 10px;">
                    <img src="{metadata['image']}" width="100" style="border-radius: 5px; margin-right: 15px;">
                    <div>
                        <strong>{metadata['title']}</strong><br>
                        💵 <span style="color: #32CD32;">${metadata['price']}</span><br>
                        ⭐ {metadata['rating']['rate']} ({metadata['rating']['count']} reviews)<br>
                        🗂️ Category: {metadata['category']}
                    </div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("🚫 No matching products found. Please try a different query.")