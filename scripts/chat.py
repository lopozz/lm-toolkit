from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="local",
)

messages = [
    {
        "role": "system",
        "content": """You create short fantasy episodes for a mobile game.

Each episode must follow this structure:

1. You — show the character in its normal state.
2. Need — give the character a clear reason to want the new ability.
3. Go — the character enters an unfamiliar situation.
4. Search — the character tries to gain or use the ability.
5. Find — the character discovers how the ability works.
6. Take — gaining the ability has a cost, risk, or difficulty.
7. Return — the character returns to safety or resolves the situation.
8. Change — the character clearly uses the new ability.

Rules:

* Use the character description provided by the user.
* write in a discorsive way"""
    }
]


messages = [
    {
        "role": "system",
        "content": """You create a space of abilities that a fantasy chracter could unlock in order to achieve a goal. The ability are dependent with each other. Some cannot be unlocked before others.

Rules:

* Use the character description provided by the user."""
    }
]

messages = [
    {
        "role": "system",
        "content": """You create a list specific goals for a fantasy chracter. The goal has to be specific and detailed for the character. Each goal is aimed at self-actualization and individuation.

Rules:

* Use the character description provided by the user."""
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
        temperature=1,
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