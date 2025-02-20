import streamlit as st
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
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

# Streamlit page configuration
st.set_page_config(page_title="🛍️ Product Information Assistant", layout="wide")

# Custom CSS for better UI
st.markdown("""
    <style>
        .block-container { padding-top: 2rem; }
        .product-card {
            display: flex;
            align-items: flex-start;
            background-color: #1E1E1E;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        .product-card img {
            width: 100px;
            height: auto;
            margin-right: 20px;
            border-radius: 8px;
        }
        .product-info {
            color: #ffffff;
        }
        .product-title {
            font-weight: bold;
            font-size: 1.1rem;
            margin-bottom: 5px;
        }
        .product-price {
            color: #4CAF50;
            font-size: 1rem;
            font-weight: bold;
            margin-bottom: 4px;
        }
        .product-rating {
            font-size: 0.9rem;
            color: #FFD700;
            margin-bottom: 4px;
        }
        .product-category {
            font-size: 0.9rem;
            color: #A0A0A0;
        }
    </style>
""", unsafe_allow_html=True)

# App title and description
st.title("🛍️ Product Information Assistant")
st.markdown("Ask any question about our products (e.g., *Which product has the best battery life?*)")

# Search input
query = st.text_input("Enter your product question:", value="", placeholder="Type your question and press Enter...")

# Add loading spinner when processing
if query:
    with st.spinner("🔎 Searching for the best products..."):
        # Encode the query and search in Pinecone
        query_embedding = embedding_model.encode(query).tolist()
        search_results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
        retrieved_products = [match['metadata'] for match in search_results['matches'] if match.get('metadata')]

    # Display results
    if retrieved_products:
        st.subheader("📦 Retrieved Products:")

        for metadata in retrieved_products:
            # Safely retrieve product details
            title = metadata.get("title", "No Title")
            price = metadata.get("price", "N/A")
            image = metadata.get("image", "")
            rating = metadata.get("rating", {})
            rate = rating.get("rate", "N/A")
            count = rating.get("count", "N/A")
            category = metadata.get("category", "Unknown")

            # Product display card
            st.markdown(
                f"""
                <div class="product-card">
                    <img src="{image}" alt="{title}">
                    <div class="product-info">
                        <div class="product-title">{title}</div>
                        <div class="product-price">💵 ${price}</div>
                        <div class="product-rating">⭐ {rate} ({count} reviews)</div>
                        <div class="product-category">📂 Category: {category}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.warning("🚫 No products found. Please try a different question.")