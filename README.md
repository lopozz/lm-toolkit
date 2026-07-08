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

### Switch the llama.cpp Submodule to the Liquid Audio Branch

Support for `LiquidAI/LFM2.5-Audio-1.5B-GGUF` is currently being developed in:

https://github.com/ggml-org/llama.cpp/pull/18641

The PR is based on the following branch:

```text
tdakhran:tarek/feat/os-lfm2.5-audio-1.5b-upstream
```

Because this support is not yet available on the default llama.cpp branch, switch the `llama.cpp` submodule to the PR branch before building.

From the repository root:

```bash
cd llama.cpp

git remote add tdakhran https://github.com/tdakhran/llama.cpp.git
git fetch tdakhran

git switch --track \
  tdakhran/tarek/feat/os-lfm2.5-audio-1.5b-upstream
```

Return to the project root:

```bash
cd ..
```

> **Note:** switching branches inside the submodule changes the local llama.cpp checkout. Running `git submodule update --init --recursive` from the parent repository restores the commit recorded by `lm-toolkit`.

### Build the Liquid Audio Targets

The Liquid Audio implementation is still a draft pull request, so not all llama.cpp targets are guaranteed to build.

After switching the `llama.cpp` submodule to the Liquid Audio branch, configure the build:

```bash
cmake \
  -S llama.cpp \
  -B llama.cpp/build \
  -DCMAKE_BUILD_TYPE=Release
```

Then build only the targets required for Liquid Audio, as suggested in the PR discussion:

https://github.com/ggml-org/llama.cpp/pull/18641#issuecomment-4016884412

```bash
cmake \
  --build llama.cpp/build \
  --target llama-server \
  --target llama-liquid-audio-server \
  --target llama-liquid-audio-cli \
  -j
```

The resulting binaries are available under:

```text
llama.cpp/build/bin/
```

In particular:

```text
llama-server
llama-liquid-audio-server
llama-liquid-audio-cli
```

#### LFM2.5-Audio-1.5B-GGUF usage instructions

Download the Q4 model components:

```bash
mkdir -p models/LFM2.5-Audio-1.5B-GGUF

hf download \
  LiquidAI/LFM2.5-Audio-1.5B-GGUF \
  LFM2.5-Audio-1.5B-Q4_0.gguf \
  mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf \
  vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf \
  tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf \
  --local-dir models/LFM2.5-Audio-1.5B-GGUF
```

Set the model directory:

```bash
export CKPT="$PWD/models/LFM2.5-Audio-1.5B-GGUF"
```

The **TTS mode** generates audio from a text prompt.

The system prompt selects the TTS mode and voice, while `-p` contains the text to be spoken.

For example:

```bash
llama.cpp/build/bin/llama-liquid-audio-cli \
  -m "$CKPT/LFM2.5-Audio-1.5B-Q4_0.gguf" \
  -mm "$CKPT/mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf" \
  -mv "$CKPT/vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf" \
  --tts-speaker-file "$CKPT/tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf" \
  -sys "Perform TTS. Use the US male voice." \
  -p "Can you tell me how to get to the hotel?" \
  --output input.wav
```

This creates:

```text
input.wav
```

The available voice-specific TTS system prompts include:

```text
Perform TTS. Use the US male voice.
Perform TTS. Use the UK male voice.
Perform TTS. Use the US female voice.
Perform TTS. Use the UK female voice.
```

The **interleaved mode** accepts audio input and can also be conditioned with additional text through `-p`.

For example, the following command listens to `input.wav` while providing additional textual information about the hotel location:

```bash
llama.cpp/build/bin/llama-liquid-audio-cli \
  -m "$CKPT/LFM2.5-Audio-1.5B-Q4_0.gguf" \
  -mm "$CKPT/mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf" \
  -mv "$CKPT/vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf" \
  --tts-speaker-file "$CKPT/tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf" \
  -sys "Respond with interleaved text and audio." \
  -p "Answer that the hotel is located at Piazza dei Fiori 3." \
  --audio input.wav \
  --output response.wav
```



### Return to the Repository-Pinned llama.cpp Version

To return to the llama.cpp revision tracked by `lm-toolkit`:

```bash
git submodule update --init --recursive --force
```

You can inspect the currently checked-out submodule revision with:

```bash
git submodule status
```
