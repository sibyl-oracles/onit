# Model Serving

How to run the LLM behind OnIt: private vLLM, local Ollama, local MLX on Apple
silicon, OpenRouter, Ollama cloud — and how to combine several of them.

For the five-minute version, see the [Quick Start](../README.md#quick-start).

## Private vLLM

Serve models locally with [vLLM](https://github.com/vllm-project/vllm):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --max-model-len 262144 --port 8000 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --reasoning-parser qwen3 --tensor-parallel-size 4 \
  --chat-template-content-format string \
  --enable-prefix-caching
```

`--enable-prefix-caching` is on by default in current vLLM and is pinned here
because OnIt is built around it. Every request opens with the same bytes — the
tool schemas, then the agent's standing rules — roughly 4k tokens that a warm
prefix cache skips prefilling entirely. An agent turn re-sends the whole
conversation to add one tool result, so that saving is paid back on every turn
of every task, not once per session. Serving without it makes prefill, not
decode, the thing you wait on.

```bash
onit --host http://localhost:8000/v1
```

To restrict the vLLM server to authorized clients, start it with one or more
API keys — **space-separated, not comma-separated** (`--api-key` is parsed
with `nargs="+"`; a comma-joined string becomes one literal key):

```bash
vllm serve ... --api-key key1 key2 key3
```

Then give OnIt one of those keys (the others are for your other clients):

```bash
onit setup   # enter it at the "vLLM API key" prompt (stored in the OS keychain)
```

Or set the environment variable:

```bash
export VLLM_API_KEY=key1
```

Resolution order: `serving.host_key` in the config YAML > `VLLM_API_KEY` env var > keychain. Without `--api-key` on the vLLM side, no key is needed and OnIt connects as before.

## Local Ollama

[Ollama](https://ollama.com) runs models on your own machine and exposes an
OpenAI-compatible API on port 11434. Start the server and pull a model that
supports tool calling — OnIt drives every task through MCP tools, so a model
without tool support will talk but not act:

```bash
ollama serve                       # or the menu-bar app
ollama pull qwen3:30b              # any tool-capable model
```

Point OnIt at it. The **`/v1` suffix is required** — that is the
OpenAI-compatible path; the bare `http://localhost:11434` root serves Ollama's
native API and every request 404s:

```bash
onit --host http://localhost:11434/v1 --model qwen3:30b
```

No API key is needed. Pass `--model` explicitly: auto-detection takes the first
entry from `/v1/models`, which for a local Ollama is whichever model you pulled
most recently, not the one you meant.

**Size the context window on the server.** Local Ollama is reached over the
OpenAI path, so the automatic `num_ctx` sizing OnIt does for Ollama *cloud*
doesn't apply — Ollama falls back to its own default (4096 tokens in current
releases, 2048 in older ones), which truncates long agent turns mid-stream. Raise it when starting the server, and tell OnIt the
same number so context compaction accounts for it:

```bash
OLLAMA_CONTEXT_LENGTH=131072 ollama serve
```

```yaml
serving:
  host: http://localhost:11434/v1
  model: qwen3:30b
  max_context_tokens: 131072
```

> **Mixing local Ollama with other endpoints:** any host on port 11434 counts as
> an Ollama endpoint, so it stays out of rotation while a vLLM or OpenRouter
> endpoint is healthy (see `serving.ollama_fallback_only` in
> [CONFIGURATION.md](CONFIGURATION.md)). That is usually what
> you want — the local box is the backstop. To load-balance across it equally,
> pass `--no-ollama-fallback-only`, or give it an explicit `priority` in an
> `endpoints` list.

## Local MLX (Apple silicon)

[MLX LM](https://github.com/ml-explore/mlx-lm) runs quantized models on the
Apple silicon GPU and ships an OpenAI-compatible server:

```bash
pip install mlx-lm
mlx_lm.server --model mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit --port 8080
```

```bash
onit --host http://localhost:8080/v1 --model mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit
```

No API key is needed. `--model` on the server side is the model loaded at
startup; `/v1/models` also lists every MLX model in your Hugging Face cache, and
naming one of those in a request loads it on demand. Name the model on the OnIt
side too — auto-detection would pick the first cache entry.

The MLX server does not report `max_model_len`, so OnIt cannot discover the
context window. Set it yourself, or long sessions compact against the wrong
budget:

```yaml
serving:
  host: http://localhost:8080/v1
  model: mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit
  max_context_tokens: 262144
```

Pick an instruct model whose chat template implements tool calling — the 4-bit
`mlx-community` builds of Qwen3, Llama 3.x, and Mistral do. `mlx_lm.server`
loads a model on first request, so the first task after startup waits on weights
being read from disk.

Any other OpenAI-compatible local server (LM Studio, llama.cpp's
`llama-server`, mlx-omni-server) works the same way: give OnIt the `/v1` base
URL, name the model, and set `max_context_tokens` if the server doesn't publish
`max_model_len`.

## OpenRouter.ai

[OpenRouter](https://openrouter.ai/) gives access to models from OpenAI, Google, Meta, Anthropic, and others through a single API.

```bash
onit --host https://openrouter.ai/api/v1
```

Browse available models at [openrouter.ai/models](https://openrouter.ai/models).

## Ollama Cloud

[Ollama cloud](https://ollama.com) hosts models accessed via the native [Ollama Python SDK](https://github.com/ollama/ollama-python). Store your API key once:

```bash
onit setup   # enter your Ollama API key when prompted
```

Or set the environment variable:

```bash
export OLLAMA_API_KEY=your-ollama-key
```

Then point OnIt at the Ollama cloud host and specify a model:

```bash
onit --host https://api.ollama.com --model glm-5.1:cloud
onit --host https://api.ollama.com --model gemma4:31b-cloud
onit --host https://api.ollama.com --model llama4:scout-cloud
```

Enable thinking mode (if supported by the model):

```bash
onit --think --host https://api.ollama.com --model glm-5.1:cloud
```

Model is auto-detected from the endpoint if `--model` is omitted. You can also set the host permanently in your config:

```yaml
serving:
  host: https://api.ollama.com
  model: glm-5.1:cloud
```

> **Note:** Ollama cloud uses the `ollama_api_key` keyring entry (the same key used for the web search tool).


## Multiple model endpoints

`serving.host` / `serving.host2` cover one or two servers. For any number of
them — or to say explicitly which should be tried first — use a
`serving.endpoints` list instead. It replaces `host`/`host2` entirely when
present:

```yaml
serving:
  endpoints:
    - name: gpu-a                        # optional label for logs
      host: http://10.0.0.1:8000/v1
      priority: 1
    - name: gpu-b
      host: http://10.0.0.2:8000/v1
      priority: 1                        # same tier as gpu-a → load balanced
    - name: ollama
      host: https://ollama.com
      model: glm-5.1:cloud               # blank = auto-detect from endpoint
      host_key: sk-...                   # optional; provider key used if omitted
      priority: 2                        # only while every tier-1 host is down
  load_balancer: least_busy              # sticky, round_robin, random, least_busy
```

**How priority works.** Lower is preferred. Requests go to the lowest-numbered
tier that still has a healthy endpoint, and `load_balancer` distributes *within*
that tier — so equal numbers share traffic, and a higher number is held in
reserve. A failing endpoint cools down for 60s; when that empties a tier, the
next one takes over, and traffic returns as soon as the preferred tier recovers.
Omit `priority` on every entry and all endpoints share a single tier.

Explicit priorities override `ollama_fallback_only`, so ranking an Ollama
endpoint first is honored rather than silently demoted.

Entries may be bare URL strings (`- http://10.0.0.1:8000/v1`) when you need
nothing but the host. Entries without a `host`, and duplicates of a host already
listed, are skipped with a warning.

**Editing endpoints.** `onit setup` opens a small editor for this list — you
don't have to write the YAML by hand:

```
   #  PRIO  HOST                     MODEL          NAME
   1  1     http://10.0.0.1:8000/v1  auto-detect    gpu-a
   2  1     http://10.0.0.2:8000/v1  auto-detect    gpu-b
   3  2     https://ollama.com       glm-5.1:cloud  ollama
  Commands: [a]dd  [e]dit N  [d]elete N  [p]riority N  [Enter] done
  endpoints>
```

Rows are listed best-first, but the number identifies the endpoint and doesn't
move when you re-rank. The wizard writes back whichever shape fits: a plain one-
or two-server config with no priorities stays as `serving.host` / `serving.host2`,
and it promotes to an `endpoints` list as soon as you add a third server, set a
priority, or name an endpoint.

## Sampling parameters

Sampling parameters (`temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `repetition_penalty`) are set in `configs/default.yaml` under `serving:`. They are not exposed as CLI flags to keep the command line clean.

**Recommended parameters for Qwen3.5:**

| Mode | Use case | `temperature` | `top_p` | `top_k` | `presence_penalty` |
|------|----------|:---:|:---:|:---:|:---:|
| Thinking (`think: true`) | General | `1.0` | `0.95` | `20` | `1.5` |
| Thinking (`think: true`) | Precise coding | `0.6` | `0.95` | `20` | `0.0` |
| Instruct (no think) | General | `0.7` | `0.8` | `20` | `1.5` |
| Instruct (no think) | Reasoning | `1.0` | `1.0` | `40` | `2.0` |

Set `repetition_penalty: 1.0` in all cases.

