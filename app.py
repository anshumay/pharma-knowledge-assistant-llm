import streamlit as st
import os
from src.loader import load_pdf
from src.chunker import chunk_text
from src.embedder import create_vectorstore
from src.llm import generate_answer

st.set_page_config(
    page_title="Pharma Knowledge Assistant",
    page_icon="💊",
    layout="wide"
)

# ---------- Session State ----------
defaults = {
    "chat_history": [],
    "db": None,
    "current_file": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ---------- Sidebar ----------
with st.sidebar:
    st.title("💊 Pharma Assistant")
    st.caption("AI-powered PDF Question Answering")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:
        st.info(f"Selected: {uploaded_file.name}")

        if st.button("📚 Create Knowledge Base", use_container_width=True):
            try:
                with st.spinner("Reading and indexing document..."):
                    os.makedirs("temp", exist_ok=True)
                    file_path = os.path.join("temp", uploaded_file.name)

                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.read())

                    pages = load_pdf(file_path)
                    documents = chunk_text(pages)
                    vectorstore = create_vectorstore(documents)

                    st.session_state["db"] = vectorstore
                    st.session_state["current_file"] = uploaded_file.name
                    st.session_state["chat_history"] = []

                st.success("Knowledge Base Ready!")

            except Exception as e:
                st.error(f"Error: {str(e)}")

    st.divider()

    if st.session_state["current_file"]:
        st.write(f"📄 Loaded File: **{st.session_state['current_file']}**")

    if st.button("🗑 Reset Chat", use_container_width=True):
        st.session_state["chat_history"] = []
        st.success("Chat cleared!")

    if st.button("♻ Reset All", use_container_width=True):
        st.session_state["chat_history"] = []
        st.session_state["db"] = None
        st.session_state["current_file"] = None
        st.success("Everything reset!")

# ---------- Main ----------
st.title("💬 Chat with Your Pharma Document")
st.caption("Upload a PDF from the sidebar and ask questions.")

# ---------- Render Chat ----------
for msg in st.session_state["chat_history"]:
    with st.chat_message("user"):
        st.write(msg["user"])

    with st.chat_message("assistant"):
        st.write(msg["assistant"])

# ---------- Input ----------
query = st.chat_input("Ask a question about the uploaded document...")

if query:
    if st.session_state["db"] is None:
        st.warning("Please upload and process a PDF first.")
    else:
        with st.chat_message("user"):
            st.write(query)

        try:
            with st.spinner("Thinking..."):
                docs = st.session_state["db"].max_marginal_relevance_search(
                    query, k=3
                )

                answer = generate_answer(
                    query,
                    docs,
                    st.session_state["chat_history"]
                )

                st.session_state["chat_history"].append({
                    "user": query,
                    "assistant": answer
                })

            with st.chat_message("assistant"):
                st.write(answer)

            with st.expander("📚 Sources"):
                for i, doc in enumerate(docs):
                    page = doc.metadata.get("page", "Unknown")
                    snippet = doc.page_content[:500]

                    st.markdown(f"**Source {i+1} | Page {page}**")
                    st.write(snippet + "...")

        except Exception as e:
            st.error(f"Something went wrong: {str(e)}")