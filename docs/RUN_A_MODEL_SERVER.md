# Run Your Own Model Server

OnIt talks to any OpenAI-compatible `/v1` endpoint. This guide covers standing one
up on your own hardware — [vLLM](https://github.com/vllm-project/vllm) and
[SGLang](https://github.com/sgl-project/sglang) on NVIDIA GPUs,
[MLX](https://github.com/ml-explore/mlx-lm) on Apple silicon, and
[Ollama](https://ollama.com) anywhere — and pointing OnIt at the result.

Nothing here is required to try OnIt: a hosted endpoint (Ollama cloud, OpenRouter)
works out of the box and needs no GPU. See the
[Quick Start](../README.md#quick-start). Once a server is running, the OnIt side of
it — key storage, context sizing, several endpoints with failover — is in
[MODEL_SERVING.md](MODEL_SERVING.md).

## What the model has to do

**Tool calling is not optional.** OnIt does its work through MCP tools: it reads
files, runs shell commands, and searches the web by emitting tool calls. A model
without tool support will hold a conversation and accomplish nothing. Pick an
instruct model whose chat template implements tool calling — the Qwen3 instruct
line, Llama 3.x, Mistral, GLM, and gpt-oss all do — and make sure the *server* is
started with tool-call parsing switched on. Both vLLM and SGLang need an explicit
flag for it; without it, tool calls arrive as prose in the assistant message and
the agent never acts.

Two other things pay off across a whole session:

- **Prefix caching.** Every request opens with the same bytes — the tool schemas,
  then the agent's standing rules, roughly 4k tokens. An agent turn re-sends the
  entire conversation to add one tool result, so a warm prefix cache is paid back
  on every turn of every task. It is on by default in current vLLM
  (`--enable-prefix-caching`) and SGLang (RadixAttention), and both flags below
  pin it deliberately. Serving without it makes prefill, not decode, the thing you
  wait on.
- **A large context window.** Agent turns grow fast — tool results, file contents,
  error output. Serve as much context as the hardware allows, then tell OnIt the
  same number (`serving.max_context_tokens`) so its compaction budget matches
  reality.

## vLLM (NVIDIA GPUs)

```bash
pip install vllm
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --port 8000 \
  --max-model-len 262144 \
  --tensor-parallel-size 4 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  --chat-template-content-format string \
  --enable-prefix-caching
```

→ host `http://localhost:8000/v1`

| Flag | Why |
|------|-----|
| `--enable-auto-tool-choice` | Lets the model decide when to call a tool. Without it vLLM never emits `tool_calls`. |
| `--tool-call-parser hermes` | Turns the model's tool syntax into OpenAI `tool_calls`. Match it to the model family — `hermes` for Qwen3, `llama3_json` for Llama 3.x, `mistral` for Mistral. `vllm serve --help` lists what your build carries. |
| `--reasoning-parser qwen3` | Separates thinking from the answer, so `onit --think` shows reasoning instead of leaking it into the reply. Omit for a non-reasoning model. |
| `--chat-template-content-format string` | Sends message content as a plain string. Some templates render the structured form badly and drop tool results. |
| `--max-model-len` | The context window. Lower it if the KV cache doesn't fit — vLLM refuses to start rather than silently truncate. |
| `--tensor-parallel-size` | Number of GPUs to shard across. Must divide the model's attention-head count. |

Restrict access with one or more API keys — **space-separated, not
comma-separated** (`--api-key` is parsed with `nargs="+"`, so a comma-joined
string becomes one literal key):

```bash
vllm serve ... --api-key key1 key2 key3
```

Quantized weights (AWQ, GPTQ, FP8) cut GPU memory roughly in half and are loaded
by naming the quantized repo — `vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507-AWQ`;
vLLM infers the method from the checkpoint.

## SGLang (NVIDIA GPUs)

[SGLang](https://github.com/sgl-project/sglang) is the other production server
worth knowing. Its RadixAttention prefix cache is on by default and it tends to
hold throughput better under the many-turn, shared-prefix traffic an agent
generates.

```bash
pip install "sglang[all]"
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server \
  --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --host 0.0.0.0 --port 30000 \
  --context-length 262144 \
  --tp 4 \
  --tool-call-parser qwen25 \
  --reasoning-parser qwen3
```

→ host `http://localhost:30000/v1`

The flags map onto vLLM's nearly one for one: `--tp` is `--tensor-parallel-size`,
`--context-length` is `--max-model-len`, and `--tool-call-parser` plays the role of
vLLM's parser *plus* `--enable-auto-tool-choice` — SGLang needs no separate switch
to allow tool choice, but it emits `tool_calls` only when a parser is named. Parser
names differ from vLLM's (`qwen25` covers the Qwen3 instruct models, alongside
`llama3`, `mistral`, `deepseekv3`, `pythonic`); run
`python -m sglang.launch_server --help` to see the set your build ships.

Add `--api-key <key>` to require one, and `--mem-fraction-static 0.85` to hand more
GPU memory to the KV cache when a long context won't fit. Prefix caching needs no
flag; `--disable-radix-cache` would turn it off, which you do not want here.

## MLX (Apple silicon)

[MLX LM](https://github.com/ml-explore/mlx-lm) runs quantized models on the Apple
silicon GPU and ships an OpenAI-compatible server:

```bash
pip install mlx-lm
mlx_lm.server --model mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit --port 8080
```

→ host `http://localhost:8080/v1`

No API key, no parser flags — tool calling comes from the model's chat template, so
pick a build whose template implements it (the 4-bit `mlx-community` releases of
Qwen3, Llama 3.x, and Mistral do). A rough sizing rule for 4-bit weights: a 30B
model wants ~18 GB of unified memory before the KV cache, so 32 GB is a
comfortable floor and 64 GB is roomy.

Two MLX-specific quirks:

- `/v1/models` lists **every** MLX model in your Hugging Face cache, not just the
  one loaded at startup. Naming another one in a request loads it on demand — and
  it is why you should always pass `--model` on the OnIt side rather than let
  auto-detection pick the first cache entry.
- The server does not report `max_model_len`, so OnIt cannot discover the context
  window and falls back to an assumed 262144. Set `max_context_tokens` yourself
  (see [below](#tell-onit-about-it)), or long sessions compact against the wrong
  budget.

The first request after startup waits on weights being read from disk.

## Ollama (anywhere)

The least setup of the local options, and the only one that runs on CPU without
ceremony:

```bash
OLLAMA_CONTEXT_LENGTH=131072 ollama serve   # the 4096 default truncates agent turns
ollama pull qwen3:30b
```

→ host `http://localhost:11434/v1` — the **`/v1` suffix is required**. The bare
root serves Ollama's native API and every request 404s.

Raising `OLLAMA_CONTEXT_LENGTH` matters more here than anywhere else: reached over
the OpenAI path, Ollama uses its own default window (4096 tokens in current
releases), which cuts long agent turns off mid-stream. Set it on the server and
mirror the number into `max_context_tokens`.

Other OpenAI-compatible servers — LM Studio, llama.cpp's `llama-server`,
mlx-omni-server — work the same way: give OnIt the `/v1` base URL, name the model,
and set `max_context_tokens` if the server doesn't publish `max_model_len`.

## Check the server before pointing OnIt at it

Two curls separate a server problem from an OnIt problem. First, that it answers
and what it calls the model:

```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```

Then — the one that actually matters — that it emits a tool call:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "messages": [{"role": "user", "content": "What is the weather in Manila?"}],
    "tools": [{"type": "function", "function": {
      "name": "get_weather",
      "description": "Get the weather for a city",
      "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    }}]
  }' | python3 -m json.tool
```

The response must contain a `tool_calls` array. If the model instead *describes*
calling the tool in `content`, the parser flags are wrong or missing — fix that
before going further, because OnIt will do nothing useful until it is fixed.

Add `-H "Authorization: Bearer <key>"` to both if you started the server with an
API key.

## Tell OnIt about it

Per run:

```bash
onit --host http://localhost:8000/v1 --model Qwen/Qwen3-30B-A3B-Instruct-2507
```

Saved, with the key prompted for and stored in the OS keychain rather than
`config.yaml`:

```bash
onit setup
```

Or by hand in `~/.onit/config.yaml`:

```yaml
serving:
  host: http://localhost:8000/v1
  model: Qwen/Qwen3-30B-A3B-Instruct-2507
  max_context_tokens: 262144
```

**Name the model explicitly** on any server that lists more than one — MLX,
Ollama, a hosted endpoint. Auto-detection takes the first entry from `/v1/models`,
which is rarely the one you meant. A single-model vLLM or SGLang server can be
left to auto-detect.

Several servers at once, with priority tiers and failover, are covered in
[MODEL_SERVING.md](MODEL_SERVING.md#multiple-model-endpoints) — including the
`\host add` command that spreads a running session across a second GPU box.

## When it doesn't work

| Symptom | Cause |
|---|---|
| Every request 404s | Missing `/v1` on the host URL (Ollama and llama.cpp especially). |
| Agent talks but never runs a tool | Tool parsing off: no `--enable-auto-tool-choice`/`--tool-call-parser` (vLLM), no `--tool-call-parser` (SGLang), or a model whose template has no tool support. |
| Wrong tool-call parser for the family | Tool calls come back malformed or half-parsed. Match the parser to the model, not to the server's default. |
| Replies cut off mid-tool-call | Server context window too small — raise `--max-model-len` / `--context-length` / `OLLAMA_CONTEXT_LENGTH`. |
| Sessions compact far too early or overflow late | OnIt's `max_context_tokens` disagrees with the server's real window. Set it to match. |
| 401 from the endpoint | Key missing or stored against a different URL. `onit setup --show` lists what is set; `\key` sets one inside a session. |
| Server won't start, OOM on load | Context window or model too large for the GPU. Lower `--max-model-len`, raise `--tensor-parallel-size`, or use a quantized checkpoint. |
| First task after startup hangs for a minute | Weights loading (MLX and Ollama load lazily). |

Run `onit --show-logs` to see the tool traffic while you debug.
