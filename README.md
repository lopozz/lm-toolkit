# 🤖 lm-toolkit

Handy scripts to run, evaluate, and train Language Models.

## 🛠 Installation & Setup

Clone the repository together with its Git submodules:

```bash
git clone --recurse-submodules https://github.com/lopozz/lm-toolkit.git
cd lm-toolkit
```

If the repository was already cloned without submodules, initialize them with:

```bash
git submodule update --init --recursive
```

### Install the Python Library

Create and activate the virtual environment:

```bash
uv venv
source .venv/bin/activate
```

Install the package:

```bash
uv pip install -e .
```

### Install the Hugging Face CLI

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
```

## 🦙 Build llama.cpp

`llama.cpp` is included in this repository as a Git submodule under:

For complete build instructions, including CUDA, Metal, Vulkan, HIP, SYCL, and other backends, see the official llama.cpp build documentation:

https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md


## 🌐 Run a llama.cpp Server

### Download a GGUF Model

For example, download Gemma 3 1B:

```bash
mkdir -p models/gemma3

hf download \
  unsloth/gemma-3-1b-it-GGUF \
  gemma-3-1b-it-Q2_K.gguf \
  --local-dir models/gemma3
```

### Start the Server

```bash
llama.cpp/build/bin/llama-server \
  -m models/gemma3/gemma-3-1b-it-Q2_K.gguf \
  -c 8192 \
  --host 127.0.0.1 \
  --port 8080
```

For GPU offloading:

```bash
llama.cpp/build/bin/llama-server \
  -m models/gemma3/gemma-3-1b-it-Q2_K.gguf \
  -c 8192 \
  -ngl 99 \
  --host 127.0.0.1 \
  --port 8080
```

Open the chat UI:

```text
http://127.0.0.1:8080
```

### Test the API

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


## 💧 Liquid AI Audio Models

Support for:

```text
LiquidAI/LFM2.5-Audio-1.5B-GGUF
```

is currently being developed in the following llama.cpp pull request:

https://github.com/ggml-org/llama.cpp/pull/18641

The implementation required by the Liquid Audio model is not part of the default llama.cpp branch used by this repository. To experiment with the model, switch the llama.cpp submodule to the code from PR `#18641` before building.

### Switch the llama.cpp Submodule to the Liquid Audio PR

From the repository root:

```bash
cd llama.cpp
```

Fetch the PR into a local branch:

```bash
git fetch origin pull/18641/head:lfm2.5-audio
```

Switch to it:

```bash
git switch lfm2.5-audio
```

Return to the project root:

```bash
cd ../..
```

Then configure and build llama.cpp as described in the [Build llama.cpp](#-build-llamacpp) section.


> **Note:** switching the branch inside the submodule changes the local llama.cpp checkout. Running `git submodule update --init --recursive` from the parent repository will restore the submodule to the commit recorded by `lm-toolkit`.

### Return to the Repository-Pinned llama.cpp Version

To return to the llama.cpp revision tracked by `lm-toolkit`:

```bash
git submodule update --init --recursive --force
```

You can inspect the currently checked-out submodule revision with:

```bash
git submodule status
```
