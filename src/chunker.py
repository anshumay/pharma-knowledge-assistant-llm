from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "; ",
            ", ",
            " "
        ]
    )

    documents = []

    for page in pages:
        page_num = page["page"]
        text = page["text"].strip()

        if not text:
            continue

        chunks = splitter.split_text(text)

        for chunk in chunks:
            documents.append({
                "page": page_num,
                "text": chunk.strip()
            })

    return documents