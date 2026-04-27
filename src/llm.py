import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL_NAME = "gpt-4o-mini"


def generate_answer(query, docs, chat_history):
    # Retrieved context
    context = "\n\n".join(
        [f"[Source {i+1}]\n{doc.page_content}" for i, doc in enumerate(docs)]
    )

    # Keep only last 5 chat turns
    recent_history = chat_history[-5:]

    messages = [
        {
            "role": "system",
            "content": """
You are a helpful pharma document assistant.

Rules:
1. Answer ONLY using the provided document context when possible.
2. If the answer is not found, say: "I couldn't find that in the uploaded document."
3. Be clear, concise, and accurate.
4. For medical topics, remind users to consult a healthcare professional.
5. Use bullet points when useful.
6. If summarizing, provide structured output.
"""
        }
    ]

    # Add previous conversation
    for item in recent_history:
        messages.append({"role": "user", "content": item["user"]})
        messages.append({"role": "assistant", "content": item["assistant"]})

    # Current query
    messages.append({
        "role": "user",
        "content": f"""
Document Context:
{context}

Question:
{query}
"""
    })

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.2,
        max_tokens=700
    )

    return response.choices[0].message.content.strip()