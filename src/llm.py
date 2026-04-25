import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_answer(query, docs, chat_history):
    context = "\n\n".join([doc.page_content for doc in docs])

    history_text = "\n".join(
        [f"User: {item['user']}\nAssistant: {item['assistant']}" for item in chat_history]
    )

    prompt = f"""
    You are a helpful pharma document assistant.

    Use the conversation history and retrieved context to answer the question.
    If the answer is not in the context, say you don't know.

    Conversation History:
    {history_text}

    Document Context:
    {context}

    Current Question:
    {query}

    Answer:
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content