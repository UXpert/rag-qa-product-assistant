import streamlit as st
from retrieve_and_answer import retrieve_products, generate_answer

# Page configuration
st.set_page_config(page_title="Product Info Assistant", page_icon="🛍️", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    .search-container {
        display: flex;
        gap: 10px;
        align-items: center;
        margin-bottom: 20px;
    }
    .search-input input {
        height: 50px;
        font-size: 16px;
        padding: 10px;
        border-radius: 8px;
        width: 100%;
        border: 1px solid #ccc;
    }
    .stButton>button {
        height: 50px;
        padding: 0 20px;
        font-size: 16px;
        border-radius: 8px;
        background-color: #4CAF50;
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .separator {
        margin: 30px 0;
        border-top: 2px solid #e0e0e0;
    }
    .answer-box {
        background-color: #d4edda;
        padding: 20px;
        border-radius: 10px;
        font-size: 16px;
        font-weight: 500;
        color: #155724;
        border: 1px solid #c3e6cb;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar info
with st.sidebar:
    st.info("💡 **About this tool**\nThis is a RAG-powered assistant that retrieves product information and provides human-like answers.")
    st.markdown("🔍 *Example questions:*\n- Which product has the best battery life?\n- Which product has the largest display?\n- Which product has the most storage?")

# Main title
st.markdown("<h1 style='text-align: center;'>🛍️ Product Information Assistant</h1>", unsafe_allow_html=True)

# 🔍 Search form for Enter key and button submission
with st.form("product_search_form"):
    st.markdown("##### Enter your product question:")
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input("", placeholder="Type your question here...", label_visibility="collapsed")
    with col2:
        submit = st.form_submit_button("🔍 Search")  # Handles both Enter and button click

# Separator between search and results
st.markdown("<div class='separator'></div>", unsafe_allow_html=True)

# Results section
if submit and query:
    with st.spinner("🔎 Searching for relevant products..."):
        retrieved_docs = retrieve_products(query)

    if retrieved_docs:
        st.markdown("## 📄 Retrieved Information:")
        for doc in retrieved_docs:
            st.write(f"- {doc}")

        with st.spinner("🤖 Generating answer..."):
            answer = generate_answer(query, retrieved_docs)

        if answer:
            st.markdown(f"<div class='answer-box'>📝 **Answer:**<br>{answer}</div>", unsafe_allow_html=True)
        else:
            st.error("⚠️ Could not generate an answer.")
    else:
        st.warning("⚠️ No relevant products found. Try rephrasing your question.")