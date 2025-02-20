import streamlit as st
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# ------------------- Setup -------------------
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("product-index")

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# ------------------- Functions -------------------
def retrieve_products(query: str, top_k: int = 3):
    """Retrieve top k products from Pinecone based on query."""
    query_embedding = embedding_model.encode(query).tolist()
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return results['matches'] if 'matches' in results else []


# ------------------- Streamlit UI -------------------

# Page Configuration
st.set_page_config(page_title="🛍️ Product Information Assistant", layout="wide")

# CSS for custom styling
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 12px;
    }
    .search-container {
        display: flex;
        gap: 10px;
        align-items: center;
        margin-bottom: 20px;
    }
    .product-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .product-title {
        font-size: 20px;
        font-weight: bold;
        color: #333333;
    }
    .product-price {
        font-size: 16px;
        color: #27ae60;
        font-weight: 600;
    }
    .product-rating {
        font-size: 14px;
        color: #f39c12;
    }
    .product-description {
        font-size: 14px;
        color: #555555;
        margin-top: 8px;
    }
    hr {
        border: none;
        height: 1px;
        background-color: #e0e0e0;
        margin: 30px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🛍️ Product Information Assistant")
st.write("Ask any question about our products (e.g., *Which product has the best battery life?*)")

# ------------------- Search Bar -------------------
with st.form(key="search_form"):
    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_input("", placeholder="Enter your product question...")
    with col2:
        submit_button = st.form_submit_button("🔎 Search")

st.markdown("<hr>", unsafe_allow_html=True)


# ------------------- Search Handling -------------------
if submit_button and query:
    with st.spinner("🔍 Searching for relevant products..."):
        retrieved_docs = retrieve_products(query)

    if retrieved_docs:
        st.subheader("📦 Retrieved Products:")
        for doc in retrieved_docs:
            metadata = doc['metadata']

            # Product Card Layout
            with st.container():
                st.markdown('<div class="product-card">', unsafe_allow_html=True)

                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(metadata['image'], width=140)
                with col2:
                    st.markdown(f'<div class="product-title">{metadata["title"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="product-price">💵 ${metadata["price"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="product-rating">⭐ {metadata["rating"]["rate"]} ({metadata["rating"]["count"]} reviews)</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="product-description">{metadata["description"]}</div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

        st.success("✅ Products retrieved successfully!")
    else:
        st.warning("🚫 No products found for your query.")