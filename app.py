import streamlit as st
from src.loader import load_pdf
from src.chunker import chunk_text

st.set_page_config(page_title="Pharma Assistant")

st.title("Pharma Knowledge Assistant")
st.caption("Chunking pharma documents for RAG")

if st.button("Process Sample PDF"):
    text = load_pdf("data/sample.pdf")
    chunks = chunk_text(text)

    st.success(f"Created {len(chunks)} chunks")

    for i, chunk in enumerate(chunks):
        st.subheader(f"Chunk {i+1}")
        st.write(chunk)