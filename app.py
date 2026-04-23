import streamlit as st
from src.loader import load_pdf
from src.chunker import chunk_text
from src.embedder import create_vectorstore

st.set_page_config(page_title="Pharma Assistant")

st.title("Pharma Knowledge Assistant")
st.caption("Semantic Search on Pharma Documents")

if st.button("Create Knowledge Base"):
    text = load_pdf("data/sample.pdf")
    chunks = chunk_text(text)
    vectorstore = create_vectorstore(chunks)

    st.session_state["db"] = vectorstore
    st.success("Knowledge Base Created!")

query = st.text_input("Ask a question")

if query and "db" in st.session_state:
    docs = st.session_state["db"].similarity_search(query, k=3)

    st.subheader("Top Relevant Chunks")
    for i, doc in enumerate(docs):
        st.write(f"### Result {i+1}")
        st.write(doc.page_content)