from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def create_vectorstore(documents):
    # Better reusable embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    texts = [doc["text"] for doc in documents]
    metadatas = [{"page": doc["page"]} for doc in documents]

    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory="chroma_db",
        collection_name="pharma_docs"
    )

    return vectorstore