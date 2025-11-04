from openai import OpenAI

def retriever(query: str):
    # Minimal example retriever
    return ["Harrison worked at Kensho"]

# OpenAI client call (no wrapping yet)
client = OpenAI()

def rag(question: str) -> str:
    docs = retriever(question)
    system_message = (
        "Answer the user's question using only the provided information below:\n"
        + "\n".join(docs)
    )

    # This call is not traced yet
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": question},
        ],
    )
    return resp.choices[0].message.content

if __name__ == "__main__":
    print(rag("Where did Harrison work?"))
