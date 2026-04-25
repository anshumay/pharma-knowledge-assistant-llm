import streamlit as st
import os
from src.loader import load_pdf
from src.chunker import chunk_text
from src.embedder import create_vectorstore
from src.llm import generate_answer

st.set_page_config(page_title="Pharma Assistant")

st.title("Pharma Knowledge Assistant")
st.caption("Upload PDF, chat with memory, view citations")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    os.makedirs("temp", exist_ok=True)
    file_path = os.path.join("temp", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    if st.button("Create Knowledge Base"):
        with st.spinner("Processing PDF..."):
            pages = load_pdf(file_path)
            documents = chunk_text(pages)
            vectorstore = create_vectorstore(documents)
            st.session_state["db"] = vectorstore

        st.success("Knowledge Base Created!")

query = st.text_input("Ask a question")

if query and "db" in st.session_state:
    with st.spinner("Generating answer..."):
        docs = st.session_state["db"].max_marginal_relevance_search(query, k=3)

        answer = generate_answer(
            query,
            docs,
            st.session_state["chat_history"]
        )

        st.session_state["chat_history"].append({
            "user": query,
            "assistant": answer
        })

    st.subheader("Answer")
    st.write(answer)

    with st.expander("Sources"):
        for i, doc in enumerate(docs):
            page = doc.metadata.get("page", "Unknown")
            st.markdown(f"### Source {i+1} (Page {page})")
            st.write(doc.page_content)