import streamlit as st
import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Pinecone and index
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("product-index")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sidebar with demo description
st.sidebar.title("ℹ️ About this Demo")
st.sidebar.markdown("""
This **Product Information Assistant** uses **Retrieval-Augmented Generation (RAG)** to help you find products based on your queries.

### How it works:
1. Your query is converted into an embedding.
2. We retrieve relevant products from a **Pinecone** vector database.
3. Results display product info like price, category, and description.

👉 Try queries like:
- **Show me electronics**  
- **Show me clothing**  
- **Find products under $50**  
""")

# Main UI
st.title("🛍️ Product Information Assistant")
st.write("Ask any question about our products (e.g., *Show me electronics*)")

query = st.text_input("Enter your product question:", placeholder="Show me electronics")

if query:
    with st.spinner("🔎 Searching for products..."):
        # Generate embedding for the query
        query_embedding = model.encode(query).tolist()

        # Query Pinecone
        response = index.query(vector=query_embedding, top_k=5, include_metadata=True)
        matches = response.get("matches", [])

        # Diagnostic: Check for mismatched categories
        # for match in matches:
            # metadata = match["metadata"]
            # if metadata.get("category", "").lower() not in query.lower():
                # st.warning("⚠️ **Mismatched Category Detected:**")
                # st.json(metadata)

        # Display retrieved products
        if matches:
            st.subheader("📦 Retrieved Products:")
            for match in matches:
                metadata = match['metadata']
                image_html = f'<img src="{metadata.get("image", "")}" width="120" style="float:left; margin-right:16px; border-radius:8px;">' if metadata.get("image") else ''
                
                st.markdown(f"""
                    <div style="border:1px solid #4F4F4F; border-radius:8px; padding:16px; margin-bottom:16px; background-color:#2E2E2E; color:white;">
                        {image_html}
                        <h4>{metadata.get('title', 'No title')}</h4>
                        <p><strong>💵 Price:</strong> ${metadata.get('price', 'N/A')}<br>
                        <strong>⭐ Rating:</strong> {metadata.get('rating_rate', 'N/A')} ({metadata.get('rating_count', '0')} reviews)<br>
                        <strong>📂 Category:</strong> {metadata.get('category', 'N/A')}<br>
                        <strong>📝 Description:</strong> {metadata.get('description', 'No description available')}</p>
                        <div style="clear:both;"></div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ No matching products found. Please try a different query.")