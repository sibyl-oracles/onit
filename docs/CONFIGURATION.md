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
                               # (unreported and unset falls back to 1000000)
  think: true                  # reasoning mode; on by default
  max_tokens: 32768

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

The environment variables OnIt reads are listed in full at the bottom of this
page.

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
  think: true          # reasoning before answering; --no-think turns it off
  # Per-UI overrides. A web run inherits everything above, exactly as the
  # terminal does; anything named here applies to the browser only.
  # web:
  #   think: false       # reasoning off in the browser alone
  #   temperature: 0.7   # and different sampling, if you want it
  max_tokens: 32768   # max output tokens per response (shipped default;
                      # clamped to whatever is left of the context window
                      # per request). 32768 fits any single answer and keeps
                      # the context-compaction threshold healthy (0.70 on a
                      # 128k window, 0.825 on a 256k one); raise it only for
                      # genuinely huge single responses.
  # Both budgets are also CLI flags, which accept a k/M suffix:
  #   onit --max-tokens 1M --max-context-tokens 1M
  # Sampling parameters (all optional — sensible defaults apply):
  # temperature: 1.0
  # top_p: 0.95
  # top_k: 20
  # presence_penalty: 1.5
  # repetition_penalty: 1.0
  # Additional model servers (any mix of vLLM / local Ollama or MLX /
  # OpenRouter / Ollama cloud) go in an endpoints list — see MODEL_SERVING.md.
  # load_balancer: sticky          # or: round_robin, random, least_busy
  # Ollama endpoints (cloud or local) are fallback-only: while any
  # vLLM/OpenRouter endpoint is healthy they stay out of rotation. Set false
  # to load-balance across them equally:
  # ollama_fallback_only: true

verbose: false
timeout: 600

# Text UI only: fold each intermediate AI turn (the narration a model streams
# before a tool call) into a one-line dim step marker, so the terminal shows
# your question, the steps, then the final answer. Default: false (folding on).
# Set true to see every intermediate turn in full, as before.
# show_intermediate: false

# Stream tokens to the UI as they are generated (default: true).
# stream: true

web_port: 9000

theme: white         # or "dark"
topic: ~             # default topic context, e.g. "machine learning"
template_path: ~     # custom prompt template YAML
data_path: ~         # working directory for file operations (default: ~/sandbox)

mcp:
  servers:
    - name: PromptsMCPServer
      transport: stdio
      module: onit.mcp.prompts.prompts
      enabled: true
    - name: ToolsLocalMCPServer   # per-user: bash, files, document search
      transport: stdio
      module: tasks.tools
      profile: local
      enabled: true
    - name: ToolsNetMCPServer     # stateless: web search, weather
      transport: stdio
      module: tasks.tools
      profile: net
      enabled: true
```

Every default server is a subprocess of the OnIt that started it, spoken to
over a pipe. There is no port to find, to collide with, or to expose: each
server runs as you, exits with you, and no other account on the machine can
reach it — which is also what lets several people run OnIt on one machine at
the same time. `profile` picks which half of the toolset a tools server
registers: `local` holds every tool that touches the session working
directory, `net` the stateless remote calls.

An MCP server that already runs somewhere else — not started by OnIt — is
listed with `external: true`. OnIt neither starts it nor re-ports it; tools are
discovered from the URL as written:

```yaml
mcp:
  servers:
    - name: MyRemoteServer
      url: http://gpu-box:8080/sse
      external: true
      enabled: true
```


## The agent loop's guard rails

Ceilings on the ways a run can fail to terminate. Every key is optional and
lives under `serving:`; the default shown is what applies when the key is
absent.

```yaml
serving:
  # Total turns (one model call each) before the loop gives up and returns
  # whatever it has. Disabled by default: a long task legitimately runs
  # hundreds of turns, and a fixed cap stops that work mid-step. Set a
  # positive number to bound runs by turn count.
  max_chat_iterations: -1
  # Identical tool calls (same name and byte-identical arguments) before
  # bailing out. A model alternating between calls that differ by a page
  # number or a timestamp never trips it.
  max_repeated_tool_calls: 30
  # Retries for a transient API error, with exponential backoff.
  max_api_retries: 3
  # Nudges to a model that announces a plan but calls no tool, and pushes
  # past a content-free acknowledgment, before accepting either.
  max_planning_continuations: 2
  max_ack_continuations: 2
  # Resumes of a final answer that stopped before it was finished — the
  # output token budget ran out, or the host dropped the tail of the answer
  # and still reported a clean stop. Each resume grants another max_tokens,
  # so a long answer is bounded at ~(N+1)*max_tokens.
  max_final_continuations: 3
  # Repeated-call stops (the same tool call with the same arguments returning
  # the same result) converted into a strategy-change prompt instead of
  # ending the run. The first bail asks the model to change course and the
  # task continues; a model that re-loops after this many recoveries ends
  # the run on the guard's own stop message.
  max_repeat_recoveries: 2
  # Times tools are handed back to a model that answered a harness nudge
  # ("write the final answer now, do not call any more tools") by saying the
  # task is unfinished instead of answering. The first one gets its tools
  # back; a model that argues again after that has said its piece, and the
  # reply is returned as-is.
  max_nudge_decline_recoveries: 1
```

## Harness tools

The harness can hand the model a few tools of its own — the ones that let a
long run manage its own context:

```yaml
serving:
  # note_write / note_read (a note that survives context summarization),
  # context_status, and result_read / result_grep (reading the rest of a
  # stored tool result instead of running the tool again). Offered only to a
  # run that has tools of its own; set false to withdraw them and their
  # prompt block.
  harness_tools: true
  # A tool result larger than ~8,000 characters is written whole to
  # <data_path>/.onit/results/ and enters the conversation as its first
  # 6,000 characters plus a handle; an aged result is cut to ~1,200
  # characters, because the rest stays reachable. Set false to go back to a
  # hard cut at 16,000 characters with the middle discarded permanently.
  # Needs a data_path; rides with harness_tools.
  result_store: true
```

## Code as action

```yaml
serving:
  # The model may answer by writing Python instead of JSON tool calls.
  # OFF BY DEFAULT, and read this before turning it on: the code runs in a
  # child process with the same privileges as OnIt — no AST allowlist and no
  # path jail, unlike the bash tool. Enable it where the deployment is
  # already isolated (`onit --container`), and be deliberate about enabling
  # it on a web deployment other people can reach. Small models also write
  # worse Python than they write JSON tool calls, so this can cost accuracy
  # on 8B-class models even as it cuts turns: benchmark your model first.
  code_execution: false
  # Seconds one run_code block may take, tool calls made from inside it
  # included. Past it the interpreter is killed and restarted, which loses
  # the variables — the model is told so.
  code_timeout: 120
```


## Thinking, and turning it off

Reasoning is on by default, in the terminal and in the browser. The models
worth running locally reason before they answer, and a reasoning model told not
to behaves like a smaller one — the deliberation is the capability, not an
extra. It costs latency, and `--no-think` (or `serving.think: false`) is how
you trade that back:

```bash
onit --no-think                          # one run, no reasoning
```

```yaml
serving:
  think: false        # every run, no reasoning
```

The two UIs no longer default differently. There used to be a web-only set of
values — reasoning off, and the instruct-mode sampling to match — on the theory
that a browser turn is one iteration with no tool loop behind it, so the
reasoning *is* the wait. With reasoning on by default everywhere that trade has
no second side, and the instruct sampling would be the wrong half of a pair left
on a model that thinks. A browser run now inherits `serving:` exactly as the
terminal does.

`serving.web` remains as a **per-UI override**, which is a different thing from
a default. Anything named there applies to the web run only, so one config can
still serve the two UIs differently:

```yaml
serving:
  think: true          # both UIs reason…
  temperature: 0.6     # …at thinking-mode sampling
  web:
    think: false       # …except the browser, which does not
```

A key you name under `serving.web` is a stated preference and is applied as
written; keys you leave out are inherited from `serving:` unchanged. The block
is consumed while the config is resolved and never reaches the model server.

An explicit `--think` / `--no-think` on the command line wins over both, for one
run: `onit --no-think serve web` starts a browser session that does not reason,
without editing anything.

`think_tool_turns` is a separate, narrower switch, and it applies to both UIs:
with it `false` (the default), a thinking model reasons on the opening turn —
where the approach is decided — and not before each tool call after it. A loop
of N tool calls otherwise pays for N full reasoning passes.


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
somewhere to show it, so one-shot callers never start one.

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


## Teaching the agent about itself (`learn:`)

What the agent is allowed to change about itself is one ladder rather than a
switch per feature, so the question has one answer that can be read off the
config file, printed in a benchmark row, and raised one rung at a time (see
docs/SELF_IMPROVEMENT.md):

| level | name | what it does |
|---|---|---|
| 0 | `off` | nothing is recorded, nothing is injected |
| 1 | `observe` | trajectories are recorded; nothing is injected (default) |
| 2 | `adapt` | recalled episodes and the playbook are injected |
| 3 | `extend` | validated skills are served as tools |
| 4 | `evolve` | scaffold variants may be proposed (never auto-merged) |

`off` through `extend` are implemented. A config asking for a rung that does
not exist yet gets the *highest* implemented rung rather than the lowest —
degrading a request to `observe` would silently turn "remember this" into "do
nothing", the opposite of what was asked.

```yaml
learn:
  autonomy: observe
  path: "~/.onit/learned"   # trajectories, episodes, playbook, skills
  # Store a digest of each tool call's arguments rather than the arguments.
  # Parameter *names* are kept either way; it is the values that carry file
  # paths, search queries and the occasional pasted credential.
  redact_tool_args: true
```

At `observe` every request is byte-for-byte what it would have been with this
off; the only difference is a JSONL line per task under `path` recording what
the run did — which tools ran, how they went, tokens per turn, retries.

Environment overrides exist for benchmark baselines and one-off reproductions
that pin a run to a known level without editing the file: `ONIT_LEARN` (the
level), `ONIT_LEARN_PATH`, `ONIT_LEARN_REDACT`, and `ONIT_LEARN_AUTONOMY` —
the level as the spawned MCP servers see it, since a stdio server has no
config file of its own.


## Speaking the answer aloud (TTS)

A `speak` tool renders text as spoken audio with Kokoro-82M, run locally as
ONNX inference — no GPU required, no cloud call, no key. The web UI's speaker
toggle and the macOS app's read-aloud fetch the same engine's output from the
harness (`/api/tts`), so one model directory serves all three.

- The engine ships in a `tts` extra, not the base install:
  `pip install -e '.[tts]'`
- The model and voice files download to `~/.onit/models/kokoro` on first use
  and are reused from disk after that, so the tool works offline from the
  second call on. `ONIT_KOKORO_DIR` points the cache somewhere else.
- `ONIT_DISABLE_TTS=1` withdraws the tool.


## Voice mode (`voice:`)

Full-duplex speech-to-speech over a realtime WebSocket to an NVIDIA
NemotronLabs VoiceChat container. That model hears, thinks and speaks end to
end — OnIt adds no ASR, TTS or VAD. What OnIt adds is the tool registry: the
voice model is given a three-tool facade whose centrepiece, `ask_onit`, runs
the full agent loop. See docs/VOICE.md. The container wants a GPU with ~66 GB
free and takes minutes to load, so the microphone button stays hidden until
`/api/voice/health` reports ready.

```yaml
voice:
  enabled: false
  # Port 9100, not the container's own default of 9000 — that collides with
  # web_port when both run on one host.
  url: "ws://localhost:9100/v1/realtime"
  sample_rate: 24000        # PCM16 mono, both directions; the server
                            # resamples to 16 kHz internally
  # Spoken the moment the model commits to a tool call, so a 20-second agent
  # run is not 20 seconds of silence.
  # on_hold_message: "Give me a moment while I look that up."
  max_spoken_chars: 600     # past this the spoken form is cut at a sentence
                            # boundary; the full markdown still reaches the
                            # transcript, links and generated files included
  tool_timeout: 90          # seconds before a runaway ask_onit is cancelled
                            # and the call moves on
  barge_in: true            # OnIt measures the microphone itself and cancels
                            # the task through the same safety queue the stop
                            # button uses
  # barge_in_rms: 900       # PCM16 level that counts as speech
  # barge_in_frames: 5      # consecutive ~80 ms frames above it (~400 ms)
  max_unprompted_turns: 3   # end the call after this many model-only turns
```


## The web UI

```yaml
web: false             # serve the browser UI instead of the terminal
web_port: 9000
web_title: "OnIt Chat"
```

Google OAuth2 signs people in (docs/WEB_AUTHENTICATION.md has the setup
steps): create an OAuth 2.0 Web-application client in Google Cloud Console,
add `http://localhost:9000/auth/callback` — and your server's callback URL —
as authorized redirect URIs, then either run `onit setup`, which stores the
pair in the OS keychain, or write them here; a value in the config file takes
precedence over the stored one.

```yaml
web_google_client_id:
web_google_client_secret:
# Optional. Not set, any verified Google account gets in. Exact emails and
# domain wildcards both work.
# web_allowed_emails:
#   - "*@example.com"
#   - "user1@allusers.com"
```


## The remaining top-level keys

```yaml
session_path: "~/.onit/sessions"  # where session transcripts live
template_path: ~                  # custom prompt template YAML (examples in
                                  # onit/mcp/prompts/prompt_templates)

messages:
  safety_warning: "Press 'Enter' key to stop all tasks."

# Inputs a text-UI session ends on. Replaces the default list:
# \goodbye, \bye, \quit, \exit
# stop_command: ["\\quit"]

# Documents a research answer may open after local_search. Each one is a
# tool round trip whose result then rides along in every later prompt, so
# this is the dial between recall and how long an answer takes to arrive.
max_documents: 6

# Prior task/response pairs replayed into every request. They are re-sent on
# each turn, so a long tail costs prompt tokens on all of them: the most
# recent few are replayed whole, older answers cut to their opening.
history_turns: 6

# Build the agent instruction by calling the prompt function directly
# instead of reaching PromptsMCPServer over the network. Set false if your
# PromptsMCPServer runs a customized prompt rather than the one shipped here.
prompt_in_process: true

# Seconds before a request times out; -1 for no overall timeout. Even with
# -1, connection setup and streaming stalls (no chunk received for 300s) are
# bounded, so a wedged server cannot hang the chat loop forever.
timeout: -1
```


## Environment variable reference

Everything above can also be set — or overridden — from the environment.
`onit setup` writes the API keys to the OS keychain, which is the better place
for them; these are the variables it would otherwise fill.

| variable | what it does |
|---|---|
| `ONIT_HOST` | Serving host, same as `--host`. |
| `ONIT_MODEL` | Model for the learn loop's own LLM calls, when `serving.model` is unset. |
| `OPENROUTER_API_KEY` | OpenRouter endpoints. |
| `VLLM_API_KEY` | vLLM endpoints; also a fallback key for the embedding endpoint. |
| `OLLAMA_API_KEY` | Ollama cloud and the web-search backend. Without it (and without a keychain entry) web search is disabled for the session. |
| `TAVILY_API_KEY` | Optional first-tier web-search provider; without it search still works via Ollama / DuckDuckGo. |
| `OPENWEATHERMAP_API_KEY` | Weather tool (`OPENWEATHER_API_KEY` also read). Without it the weather tool is unavailable. |
| `GITHUB_TOKEN` | GitHub repository tools. |
| `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` | Web UI OAuth, read from the keychain when the config keys are unset. |
| `ONIT_DATA_PATH` | Session working directory, propagated to the MCP servers. Set by OnIt from `data_path` (default `~/sandbox`). |
| `ONIT_DOCUMENTS_PATH` | Corpus directory `local_search` indexes by default. |
| `ONIT_EMBEDDING_HOST`, `ONIT_EMBEDDING_MODEL` | OpenAI-compatible embedding endpoint; set both to make `local_search` hybrid (BM25 + dense). |
| `ONIT_EMBEDDING_API_KEY` | That endpoint's key (`VLLM_API_KEY` / `OPENROUTER_API_KEY` also tried). |
| `ONIT_WARM_INDEX` | `1` builds the search index at startup instead of on the first question. |
| `ONIT_SEARCH_REFRESH_INTERVAL` | Minimum seconds between index refreshes (default 30). |
| `ONIT_DISABLE_TTS` | `1` withdraws the `speak` tool. |
| `ONIT_DISABLE_WEB_SEARCH`, `ONIT_DISABLE_WEATHER`, `ONIT_DISABLE_LOCAL_SEARCH` | `1` removes that tool. Web search and weather are also disabled automatically when their key is missing. |
| `ONIT_KOKORO_DIR` | Where the TTS model files are cached. |
| `ONIT_LEARN`, `ONIT_LEARN_PATH`, `ONIT_LEARN_REDACT` | `learn:` overrides. |
| `ONIT_LEARN_AUTONOMY` | The autonomy level as the spawned MCP servers see it. |
| `ONIT_EVAL_PATH` | Where the learn loop writes evaluation output (default `<repo>/benchmarks/eval`). |
| `ONIT_UNRESTRICTED` | Set by `--unrestricted`: the bash tool runs with the container's relaxed policy on the host. Catastrophic operations stay blocked either way. |
| `ONIT_CONTAINER` | `1` inside the onit container (the Dockerfile sets it); hardens the bash tool. |
| `ONIT_ALLOW_PACKAGE_INSTALL` | `1` lets the bash tool install packages — version-pinned installs only, and ignored inside the container, where installs are sealed. |
| `ONIT_COMMAND_ALLOWLIST` | `1` enforces the command allowlist everywhere, host mode included. |
| `ONIT_AUTO_APPROVE` | `1` auto-approves tool calls that would otherwise ask. |
| `ONIT_ASK_APPROVAL` | `0` turns approval questions off where they would otherwise be asked. |
| `ONIT_APPROVAL_CHANNEL` | `1` in the UIs that can show an approval question. |
| `ONIT_MAX_UPLOAD_MB` | Web upload cap in MB (default 25). |
| `ONIT_PUBLIC_URL` | Base URL as the browser sees it, for OAuth redirect URIs; otherwise taken from the request. |
| `ONIT_DOMAIN` | Domain shown in the web UI's branding; falls back to `ONIT_PUBLIC_URL`, then the Host header. |
| `ONIT_SECURITY_CONTACT` | Disclosure contact published at `/.well-known/security.txt`; unset, the endpoint stays 404. |
| `ONIT_GA_MEASUREMENT_ID` | Google Analytics 4 ID for the web UI (`web_ga_measurement_id` in the config wins). |
| `ONIT_SETTINGS` | Path to the bash tool's settings file. |

A few more (`ONIT_STDIO_SPAWNED`, `ONIT_WEB_UI`, `ONIT_CONTAIN_THRESHOLD`,
`ONIT_CONTAIN_WINDOW`, `ONIT_TARGET_ENV_BIN`, `ONIT_FILE_SERVER_URL`) are set
by OnIt for its own subprocesses and are not meant to be set by hand.
