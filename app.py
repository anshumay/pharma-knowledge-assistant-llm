import streamlit as st
from src.loader import load_pdf
from src.chunker import chunk_text
from src.embedder import create_vectorstore
from src.llm import generate_answer

st.set_page_config(page_title="Pharma Assistant")

st.title("Pharma Knowledge Assistant")
st.caption("Ask questions from pharma documents using RAG + OpenAI")

if st.button("Create Knowledge Base"):
    with st.spinner("Processing PDF and creating knowledge base..."):
        text = load_pdf("data/sample.pdf")
        chunks = chunk_text(text)
        vectorstore = create_vectorstore(chunks)
        st.session_state["db"] = vectorstore

    st.success("Knowledge Base Created!")

query = st.text_input("Ask a question")

if query and "db" in st.session_state:
    with st.spinner("Searching documents and generating answer..."):
        docs = st.session_state["db"].similarity_search(query, k=3)
        answer = generate_answer(query, docs)

    st.subheader("Answer")
    st.write(answer)

    with st.expander("Retrieved Chunks"):
        for i, doc in enumerate(docs):
            st.write(f"### Chunk {i+1}")
            st.write(doc.page_content)