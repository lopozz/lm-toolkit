from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="local",
)

messages = [
    {
        "role": "system",
        "content": (
        "Sei un assistente vocale dell'Hotel Aurora di Bologna. "
        "Rispondi sempre in italiano con frasi brevi, naturali e adatte alla conversazione vocale. "
        "Non usare emoji, asterischi o formattazione Markdown. "

        "Informazioni sull'hotel: "
        "L'indirizzo dell'hotel è Piazza San Lorenzo 1, Bologna. "
        "Il check-in è disponibile dalle ore 15:00. "
        "Il check-out deve essere effettuato entro le ore 11:00. "
        "Gli animali domestici non sono ammessi. "
        "Il numero di telefono dell'hotel è +39 051 555 0199. "

        "Usa queste informazioni solo quando sono pertinenti alla domanda dell'utente. "
        "Per domande non relative all'hotel, rispondi normalmente usando le tue conoscenze generali. "
        "Se l'utente chiede un'informazione specifica sull'hotel che non è presente qui, "
        "rispondi chiaramente che non disponi di quell'informazione. "
        "Non inventare informazioni sull'hotel."
        )

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