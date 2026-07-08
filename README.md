# 🤖 lm-toolkit
Handy scripts to run, evaluate and train Language Models.

## 🛠 Installation & Setup
Install the Python Library
```
uv venv
source .venv/bin/activate
uv pip install -e .
```

## 🌐 Run Servers

### llama.cpp

```bash
curl -LsSf https://llama.app/install.sh | sh
```

Check the installation:

```bash
llama-server --version
```


Install the Hugging Face CLI:

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
```

Download the GGUF file:

```bash
mkdir -p models/gemma3

hf download \
  unsloth/gemma-3-1b-it-GGUF \
  gemma-3-1b-it-Q2_K.gguf \
  --local-dir models/gemma3
```

```bash
llama-server \
  -m models/gemma3/gemma-3-1b-it-Q2_K.gguf \
  -c 8192 \
  --host 127.0.0.1 \
  --port 8080
```

For GPU offloading:

```bash
llama serve \
  -m models/gemma-3-1b-it-Q2_K.gguf \
  -c 4192 \
  -ngl 99 \
  --host 127.0.0.1 \
  --port 8080
```

Open the chat UI:

```text
http://127.0.0.1:8080
```

Test the API

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [
      {
        "role": "system",
        "content": "Sei un assistente utile e preciso. Rispondi in italiano."
      },
      {
        "role": "user",
        "content": "Ciao, raccontami brevemente cosa sai fare."
      }
    ],
    "temperature": 0.7,
    "max_tokens": 500
  }'
```

The server exposes an OpenAI-compatible API at:

```text
http://127.0.0.1:8080/v1
```

Remove it with:
```
rm -rf ~/.llama-app
rm -f ~/.local/bin/llama
rm -f ~/.local/bin/llama-server
rm -f ~/.local/bin/llama-cli
```