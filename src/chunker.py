from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    documents = []

    for page in pages:
        chunks = splitter.split_text(page["text"])

        for chunk in chunks:
            documents.append({
                "page": page["page"],
                "text": chunk
            })

    return documents