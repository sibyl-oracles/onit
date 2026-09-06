# Configuration

`onit setup` covers the common cases; this is the full reference for
`~/.onit/config.yaml`, environment variables, and project config files.

## A minimal config

Everything a local Ollama setup needs, hand-written into `~/.onit/config.yaml`:

```yaml
serving:
  host: http://localhost:11434/v1
  model: qwen3:30b
  max_context_tokens: 131072   # set it when the server doesn't report its own
                               # (unreported and unset falls back to 262144)
  think: true                  # reasoning mode; off by default
  max_tokens: 131072

theme: white          # or "dark"
timeout: 600
data_path: ~          # working directory (default: ~/sandbox)
```

The rest of this page is the full set of keys.

## Where settings come from

`onit setup` is the recommended way to configure OnIt. It stores:

- **Settings** in `~/.onit/config.yaml` (LLM endpoint, theme, ports, timeout)
- **Secrets** in your OS keychain (API keys, bot tokens)

You can also use environment variables or a project-level YAML config:

```bash
# Environment variables
export ONIT_HOST=https://openrouter.ai/api/v1
export OPENROUTER_API_KEY=sk-or-v1-...

# Or a custom config file
onit --config configs/default.yaml
```

Priority order: CLI flags > environment variables > `~/.onit/config.yaml` > project config file.

## Example config (`configs/default.yaml`)

```yaml
serving:
  host: https://openrouter.ai/api/v1
  # Keys are per endpoint. 'onit setup' asks for one alongside each endpoint
  # URL and stores it in the OS keychain, which keeps it out of this file —
  # prefer that. Written here it is read first, ahead of the stored one:
  # host_key: sk-or-v1-your-key-here   # ('api_key' in an endpoints list)
  # model: auto-detected from endpoint. Set explicitly for OpenRouter:
  # model: google/gemini-2.5-pro
  think: true
  max_tokens: 131072  # max output tokens per response (default; clamped to
                      # whatever is left of the context window per request)
  # Both budgets are also CLI flags, which accept a k/M suffix:
  #   onit --max-tokens 1M --max-context-tokens 1M
  # Sampling parameters (all optional — sensible defaults apply):
  # temperature: 1.0
  # top_p: 0.95
  # top_k: 20
  # presence_penalty: 1.5
  # repetition_penalty: 1.0
  # Optional second model server (any mix of vLLM / local Ollama or MLX /
  # OpenRouter / Ollama cloud).
  # By default new sessions are spread across hosts round-robin, then each
  # session's inference sticks to its host and fails over to the other
  # only on timeout/error (the failed host cools down for 60s):
  # host2: http://localhost:8001/v1
  # host2_key: sk-...              # optional; 'onit setup' stores host2's key
                                 # under its URL instead
  # model2: auto-detected from host2 unless set
  # load_balancer: sticky          # or: round_robin, random, least_busy
  # Ollama endpoints (cloud or local) are fallback-only: while any
  # vLLM/OpenRouter endpoint is healthy they stay out of rotation. Set false
  # (or pass --no-ollama-fallback-only) to load-balance across them equally:
  # ollama_fallback_only: true
  # For more than two servers, or to rank them explicitly, use an endpoints
  # list instead of host/host2 — see MODEL_SERVING.md.

verbose: false
timeout: 600

# Text UI only: fold each intermediate AI turn (the narration a model streams
# before a tool call) into a one-line dim step marker, so the terminal shows
# your question, the steps, then the final answer. Default: false (folding on).
# Set true to see every intermediate turn in full, as before.
# show_intermediate: false

web_port: 9000
a2a_port: 9001

theme: white         # or "dark"
topic: ~             # default topic context, e.g. "machine learning"
template_path: ~     # custom prompt template YAML
data_path: ~         # working directory for file operations (default: ~/sandbox)

mcp:
  # fixed_ports: false   # true pins the ports below instead of finding free ones
  servers:
    - name: PromptsMCPServer
      url: http://127.0.0.1:18200/sse
      enabled: true
    - name: ToolsLocalMCPServer   # per-user, over stdio — no port
      transport: stdio
      module: tasks.tools
      profile: local
      enabled: true
    - name: ToolsNetMCPServer
      url: http://127.0.0.1:18201/sse
      enabled: true
```

The ports above are a starting point, not fixed addresses: each OnIt process
finds free ports at or above 18200 on startup. That is what lets several people
run OnIt on one machine at the same time — before, the second to start found
those ports taken, assumed the servers were already its own, and ran its tools
in the first user's account and sandbox. Set `mcp.fixed_ports: true` for a
single-user host, or when something outside OnIt has to reach the servers.

`ToolsLocalMCPServer` has no port at all. Every tool that touches the session
working directory lives there, and OnIt starts it as a subprocess of its own
and talks to it over a pipe — so it runs as you, exits with you, and no other
account on the machine can reach it.


## Fact-checking the answer

An answer is written from whatever is left in the model's context by the time it
writes — tool results from ten turns ago, already decayed to a summary, next to
whatever the weights remember. That is where a figure drifts by a digit: the
search result was right and the sentence quoting it is not.

So OnIt checks the answer after it is written — in two stages, because being
careful and being quick want opposite things.

**The stage you wait for** is bounded at two seconds and usually costs nothing
at all. Every figure in the answer that appears verbatim in a document the run
read, or on a source trusted about that subject, is cleared by string
comparison before any model is involved — that is where a drifting digit shows
up, and it does not need an LLM to see. Whatever is left gets one small check
against the gathered evidence, with no lookups behind it. Where the evidence
contradicts the answer, the finding is flagged in a line underneath:

```
The 2019 filing puts revenue at 4.2M …

Correction after fact-check: revenue was 3.1M
```

Flagged rather than rewritten, because rewriting means generating the whole
answer a second time, at the same speed it was written the first time. That is
what used to double a turn.

**The stage that runs behind you** starts once the answer is yours and has no
clock on it. It can make read-only lookups for claims the run gathered no
evidence about (search, file reads — never a write or a shell command), and it
does rewrite the answer when something is wrong. If it finds something, the
answer is corrected where it stands and a line says what changed — in the
browser, in place; in the terminal, at the top of your next turn, since a
terminal that writes under a half-typed line is worse than one that waits. Ask
anything else and the check is cancelled outright: a correction to an answer
you have moved past is not worth the interruption. It only runs where there is
somewhere to show it, so one-shot callers (A2A) never start one.

Measured on Qwen3.6-27B, against 5–15s to write the answer itself:

| | you wait |
|---|---:|
| every figure came from a document you gave it | **0.00s** — no call at all |
| clean check against gathered evidence | **0.2–1.5s** |
| two wrong figures found and flagged | **0.8–1.6s** |
| endpoint slow or busy | **2.0s** ceiling, then the draft stands |

The check runs with the model's chain of thought switched off: comparing a
sentence against a source is recognition, not deliberation. The same verdicts
that take 0.19s that way take **15–21s** with a hybrid model left to reason its
way through them, so a server whose chat template has no such switch is detected
— by a genuine refusal of the parameter, never by a timeout or a bad gateway —
and given room to think instead.

Answers with nothing checkable in them ("I've saved the file, let me know if
you'd like it formatted differently") skip the check, and so do runs that
gathered no evidence to check against. A check that fails, times out, or comes
back unreadable leaves the answer exactly as written — it can correct an
answer, never lose one. A claim the evidence simply does not cover is left
alone rather than doubted, unless the background stage can look it up.

```yaml
serving:
  verify_answers: true       # false hands back the answer unchecked
  verify_timeout_s: 2        # the ceiling on what you wait for
  verify_background: true    # keep checking behind the answer
  verify_max_tool_turns: 2   # lookups the background stage may make
  verify_trusted_domains:    # added to the built-in list
      - "docs.internal.example.com"
```

The per-run log line reports it alongside the rest of the timing:
`… | fact-check 2.4s (1 claim(s) corrected)`.

