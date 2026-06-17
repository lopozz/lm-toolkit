# Tool Calling Evaluation

This directory contains a small evaluator for OpenAI-compatible chat completion
servers. It sends a YAML-defined tool schema and test prompts, then checks
whether the model emits the expected `message.tool_calls`.

The evaluator uses the same client path for vLLM, llama.cpp, and any other
server that implements `/v1/chat/completions`.

## Run the Evaluator

```bash
python3 scripts/tool_calls/evaluate_tool_call.py \
  --base-url http://localhost:8000/v1 \
  --api-key EMPTY \
  --model mistralai/Ministral-3-3B-Instruct-2512 \
  --config configs/tools/student_courses.yaml \
  --temperature 0 \
  --debug
```


## vLLM Server

vLLM needs tool calling enabled at server startup. For Mistral/Ministral-family
models, use the Mistral tool parser:

```bash
vllm serve mistralai/Ministral-3-3B-Instruct-2512 \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-auto-tool-choice \
  --tool-call-parser mistral
```

Then run:

```bash
python3 scripts/tool_calls/evaluate_tool_call.py \
  --base-url http://localhost:8000/v1 \
  --api-key EMPTY \
  --model mistralai/Ministral-3-3B-Instruct-2512 \
  --config configs/tools/student_courses.yaml
```

For non-Mistral models, choose the parser recommended by vLLM for that model
family. For example, vLLM documents `llama3_json` for Llama 3.1-style tool
calling.

Reference: https://docs.vllm.ai/en/stable/features/tool_calling/

## llama.cpp Server

llama.cpp tool calling requires `llama-server` with Jinja chat-template support
enabled. The model also needs a tool-aware chat template or a compatible generic
template.

Example with a local GGUF file:

```bash
./llama.cpp/build/bin/llama-server \
  --model /path/to/model.gguf \
  --alias local-llama \
  --host 127.0.0.1 \
  --port 8080 \
  --jinja \
  --no-webui
```

Then run:

```bash
python3 scripts/tool_calls/evaluate_tool_call.py \
  --base-url http://localhost:8080/v1 \
  --api-key EMPTY \
  --model local-llama \
  --config configs/tools/student_courses.yaml
```

If the model does not have a tool-use template in its metadata, pass an explicit
template:

```bash
./llama.cpp/build/bin/llama-server \
  --model /path/to/model.gguf \
  --alias local-llama \
  --host 127.0.0.1 \
  --port 8080 \
  --jinja \
  --chat-template chatml \
  --no-webui
```

For better tool-calling reliability, prefer models/templates listed by llama.cpp
as native tool-call formats, such as Llama 3.x, Hermes, Qwen 2.5, Mistral Nemo,
Functionary, FireFunction, or Command R.

Reference: https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md

## Common Checks

Check that the server exposes models:

```bash
curl http://localhost:8000/v1/models
curl http://localhost:8080/v1/models
```

If every positive case fails with `NO CALL`, check the server launch flags first.
Sending `tools=[...]` and `tool_choice="auto"` from the client is not enough if
the server was not started with tool-calling support.

If arguments fail but calls are detected, rerun with `--debug` and inspect
`actual_arguments`.
