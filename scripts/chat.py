from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="local",
)

messages = [
    {
        "role": "system",
        "content": (
            "You are a concise Italian voice assistant. "
            "Answer in short, natural spoken sentences. "
            "Do not add emojis, asterisks, or any markdown symbols."
        ),
    }
]

while True:
    user_text = input("\nTu: ").strip()

    if user_text.lower() in {"exit", "quit"}:
        break

    if not user_text:
        continue

    messages.append({
        "role": "user",
        "content": user_text,
    })

    stream = client.chat.completions.create(
        model="local",
        messages=messages,
        temperature=0.7,
        max_tokens=500,
        stream=True,
    )

    print("\nModello: ", end="", flush=True)

    answer = ""

    for chunk in stream:
        content = chunk.choices[0].delta.content

        if content:
            print(content, end="", flush=True)
            answer += content

    print()

    messages.append({
        "role": "assistant",
        "content": answer,
    })