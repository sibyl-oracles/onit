# Agent Harness Capabilities — Gap Analysis and Build Plan

**Status**: Proposal / RFC. Nothing in Phases 1–6 is implemented.
**Date**: August 2026
**Source**: [Six Agent Harness Capabilities for Higher Model Performance](https://developer.nvidia.com/blog/six-agent-harness-capabilities-for-higher-model-performance/) (NVIDIA, NOOA framework)

---

## 1. Executive summary

NVIDIA's NOOA post names six interface properties it credits for higher agent scores:
typed I/O, pass by reference, code as action, programmable loop engineering, explicit
object state, and model-callable harness APIs. This document scores OnIt against each,
and turns the gaps into six independently shippable phases.

**Read the framing critically.** NOOA assumes the harness and its tools share one Python
process, so "typed objects" can be passed by reference in memory. OnIt's tools are MCP
servers reached over SSE, sometimes containerized — a deliberate isolation boundary, and
one worth keeping. Every phase below is deliverable without giving it up: Phase 4 buys
pass-by-reference with **handles** instead of shared object memory, and Phase 5 buys
code-as-action with a **sandboxed interpreter** instead of a shared heap.

### Scorecard

| # | Capability | OnIt today | Gap | Phase |
|---|---|---|---|---|
| 1 | Typed input/output | Schemas discovered, then partly unused; arg validation is regex repair | **Large** | 1 |
| 2 | Model-callable harness APIs | None — context management is invisible to the model | **Large** | 2 |
| 3 | Programmable loop engineering | Plain Python loop ✅, but policy constants hardcoded | Small | 3 |
| 4 | Pass by reference | Everything serializes into context; truncation is lossy | **Large** | 4 |
| 5 | Code as action | `bash` only; one model round trip per step | **Large** | 5 |
| 6 | Explicit object state | Run state is locals inside `chat()`, dies on return | Medium | 6 |

### Build order

Ordered by (payoff ÷ effort), not by the article's numbering:

```
Phase 1  Typed I/O            ~1 day     fixes a live bug
Phase 2  Harness APIs         ~1 day     small, high leverage
Phase 3  Loop policy config   ~half day  fixes an unbounded loop
Phase 4  Result store         ~3 days    largest token win
Phase 5  Code as action       ~1-2 weeks largest latency win
Phase 6  Explicit run state   ~3 days    unlocks resume + testing
```

Phases 1–3 are independent of each other and of everything else. Phase 5 depends on
Phase 4 (handles are what keep interpreter output out of the context). Phase 6 is
independent but pairs naturally with `docs/SELF_IMPROVEMENT.md` Layer 0, which already
records trajectories and wants the same state object.

---

## 2. Where OnIt already scores well

Worth stating before the gap list, because these should not be regressed:

- **The loop is plain Python.** [`chat()`](../src/model/serving/chat.py) is a readable
  `while True:` — not a graph DSL, not a framework callback maze. This is exactly what
  the article's capability #4 asks for, and it is why Phase 3 is half a day rather than
  a rewrite.
- **Per-turn telemetry exists.** [`TurnMetrics`](../src/model/serving/chat.py#L98)
  records prefill/decode/tool time and token counts turn by turn, into a caller-owned
  sink. The "caller owns the dict, harness fills it in place" pattern is the right one —
  **Phase 6 should copy it verbatim.**
- **Read-only tool batching.**
  [`_execute_tools_in_parallel`](../src/model/serving/chat.py#L1975) runs a batch
  concurrently when every call is in `_READ_ONLY_TOOLS`, with buffers drained in call
  order to satisfy the tool_call_id ordering rule.
- **MCP connection pooling.** [`_PooledClient`](../src/type/tools.py#L106) keeps one
  session per (URL, event loop), removing a connect + handshake from every tool call.
- **Recovery heuristics.** Planning-response, acknowledgment, meta-commentary and no-op
  detection are more than most harnesses have. Phase 3 makes them configurable; it must
  not delete them.

---

## 3. Phase 1 — Typed input/output

**Effort**: ~1 day · **Depends on**: nothing · **Priority**: first (contains a live bug)

### 3.1 The bug

[`_build_returns`](../src/lib/tools.py#L71) extracts each MCP tool's `outputSchema` and
stores it at [tools.py:182](../src/lib/tools.py#L182) as `function.returns` inside the
`ToolItem`. Nothing ever reads it.

Meanwhile [`ToolRegistry.get_tool_items()`](../src/type/tools.py#L374) hands the whole
dict — `returns` included — straight to `tools=` in the API payload at
[chat.py:2525](../src/model/serving/chat.py#L2525). `returns` is **not a valid field** in
the OpenAI tool schema. Today it is dead weight on every request; on any provider that
validates the tool schema strictly it is a rejection.

### 3.2 The gap

Input validation is repair-based rather than schema-based:

| Site | What it does | What it misses |
|---|---|---|
| [`_parse_tool_arguments`](../src/model/serving/chat.py#L1955) | regex quote-swap + trailing-comma strip on malformed JSON | anything that parses but is wrong |
| [`ToolHandler.__call__`](../src/type/tools.py#L243) | unwraps `{"query": {"query": ...}}` nesting | only for `string`-typed params |
| [`blank_required_args`](../src/type/tools.py#L400) | catches missing/blank **required strings** | wrong types, bad enums, out-of-range numbers |

A wrong-typed argument reaches the MCP server and returns a stack trace, which the model
then has to interpret. The existing blank-args error at
[chat.py:890](../src/model/serving/chat.py#L890) shows the right shape for the fix — it
names the offending parameters and tells the model not to repeat the call unchanged.

### 3.3 Steps

1. **Strip `returns` from the wire.** Give `ToolRegistry` a second accessor — keep
   `get_tool_items()` for internal use, add `get_api_tool_items()` that projects each
   entry down to `{type, function: {name, description, parameters}}`. Point
   [chat.py:2222](../src/model/serving/chat.py#L2222) at the new one.
2. **Hand-roll a schema validator** in `src/lib/schema.py`:
   `validate_arguments(schema: dict, args: dict) -> list[str]`, returning human-readable
   problems (`"depth: expected integer, got 'three'"`). Support `type`, `required`,
   `enum`, `minimum`/`maximum`, `anyOf`. That covers what MCP servers actually declare.
   **Do not add a dependency** — `pip install jsonschema` is blocked by the user's deny
   rules.
3. **Call it before dispatch** in [`_execute_tool`](../src/model/serving/chat.py#L795),
   folding it into the existing `_blank_args` branch so there is one refusal path, one
   error format, and one `_log(False, 0)` site.
4. **Coerce the safe cases** rather than refusing them: `"3"` → `3` for an `integer`
   param, `"true"` → `True` for a `boolean`. Small models get these wrong constantly and
   a refusal costs a whole round trip. Log coercions at debug level.
5. *(Optional, defer)* Validate tool **output** against `returns` and attach a warning to
   the tool message when it does not conform. Only worth it once a tool actually declares
   a non-trivial `outputSchema` — check what the servers in `src/mcp/servers/tasks/`
   really emit before building this.

### 3.4 Acceptance

- New `src/test/test_schema.py`: type mismatch, missing required, bad enum, out-of-range,
  `anyOf` accepted, coercion of `"3"`→`3`.
- Extend `src/test/test_tool_registry.py`: `get_api_tool_items()` output has exactly the
  keys `type` and `function`, and `function` has exactly `name`/`description`/`parameters`.
- Extend `src/test/test_chat.py`: a bad-typed call is refused before dispatch — assert the
  MCP handler was never awaited and the tool message names the parameter.

---

## 4. Phase 2 — Model-callable harness APIs

**Effort**: ~1 day · **Depends on**: nothing · **Priority**: second (small, high leverage)

### 4.1 The gap

Context management is entirely invisible to the model:

- Compaction fires on a threshold at
  [chat.py:2396–2406](../src/model/serving/chat.py#L2396-L2406) — the model is not told
  it is about to happen, and cannot prepare for it.
- [`_decay_old_tool_results`](../src/model/serving/chat.py#L733) trims older results every
  iteration, silently.
- `set_context_usage` computes the exact fill percentage — and forwards it **only to the
  UI** ([onit.py:367](../src/onit.py#L367) is a documented no-op for external clients).

So the harness knows it is at 85% and the model does not. A model that could see it would
write its findings down before the summarizer eats them. It has no way to.

### 4.2 Steps

Add a small in-process harness toolset — these are *not* MCP tools; they need access to
`messages` and the running loop's state, so they are dispatched inside
[`_execute_tool`](../src/model/serving/chat.py#L795) ahead of the registry lookup, the
same way `sandbox_download_file` is already intercepted at
[chat.py:837](../src/model/serving/chat.py#L837).

1. **`context_status()`** → `{used_tokens, max_tokens, pct, turns, tools_called}`. Reads
   `_last_prompt_tokens` / `max_context_tokens`, which the loop already tracks.
2. **`note_write(key, text)` / `note_read(key)`** → durable scratch memory under
   `data_path/.onit/notes/`. Survives compaction and decay because it lives on disk, not
   in the message list. This is the payoff for #1: seeing 85% is only useful if there is
   somewhere to put things.
3. **Announce compaction to the model, not just the UI.** When
   [`_compact`](../src/model/serving/chat.py#L2364) runs, append a short `user`-role
   notice to the compacted message list saying what happened and that `note_read` is
   available. The UI already gets `show_context_compaction`
   ([onit.py:371](../src/onit.py#L371)) — this is the model-facing twin.
4. **Register them in the instruction.** Add a block to
   [`build_assistant_instruction`](../src/mcp/prompts/prompts.py#L50), gated on a new
   `harness_tools_available` flag, following the existing `sandbox_block` /
   `local_search_available` pattern. Keep it in the **static half** of the instruction —
   it does not vary per request, so it stays prefix-cacheable.

### 4.3 Acceptance

- New `src/test/test_harness_tools.py`: `context_status` reports the loop's real numbers;
  `note_write` then `note_read` round-trips; notes land under `data_path`, not `$HOME`.
- Per the `mock-config-data-leaks-dirs` memory: mock `config_data` as a real dict with a
  `tmp_path` `data_path`, or the tests will create orphan `~/sandbox/<uuid>` directories.
- Extend `src/test/test_chat.py`: a compaction appends exactly one model-visible notice.

---

## 5. Phase 3 — Loop policy as configuration

**Effort**: ~half day · **Depends on**: nothing · **Priority**: third (fixes an unbounded loop)

### 5.1 The bug

[chat.py:2328](../src/model/serving/chat.py#L2328):

```python
MAX_CHAT_ITERATIONS = -1
...
if MAX_CHAT_ITERATIONS >= 0 and iteration_count > MAX_CHAT_ITERATIONS:
```

The turn cap is disabled. The loop is bounded only by `MAX_REPEATED_TOOL_CALLS` — which
requires the same tool name **and byte-identical arguments** 30 times
([chat.py:983](../src/model/serving/chat.py#L983)) — and by the safety queue. A model
alternating between two slightly-different calls runs until the user stops it.

### 5.2 Steps

1. Set a real default ceiling (suggest 50) and keep `-1` as an explicit opt-out.
2. Lift the constants at
   [chat.py:2328–2343](../src/model/serving/chat.py#L2328-L2343) —
   `MAX_CHAT_ITERATIONS`, `MAX_REPEATED_TOOL_CALLS`, `MAX_PLANNING_CONTINUATIONS`,
   `MAX_ACK_CONTINUATIONS`, `MAX_FINAL_CONTINUATIONS` — into `kwargs` with the current
   values as defaults, so `chat()`'s signature stays the single place each default lives.
3. Add them to `SERVING_PASSTHROUGH` at [onit.py:63](../src/onit.py#L63) so they are
   reachable from `serving:` in the config YAML. That list is already the established
   "forward only when the config sets it" mechanism — no new plumbing needed.
4. When the iteration cap trips, return a message that says the run hit its turn limit.
   The current fallback (`"Could you try to rephrase..."`) is indistinguishable from the
   repeated-tool-call bail-out, which makes the two impossible to tell apart in logs.

**Explicitly out of scope**: refactoring the recovery heuristics
(`_is_planning_response`, `_is_acknowledgment_response`, `_is_meta_commentary_response`,
`_is_content_free_response`) into a policy-object pipeline. It is tempting and it is not
worth it — those branches encode hard-won behavior against specific weak models, and the
comments explaining *why* are the valuable part. Make them configurable; leave them where
they are.

### 5.3 Acceptance

- Extend `src/test/test_chat.py`: a stub model that alternates between two distinct tool
  calls forever terminates at the configured cap, with a distinguishable message.
- Extend `src/test/test_onit.py`: `serving.max_chat_iterations` reaches `chat()`; an unset
  key does not appear in `kwargs` at all.

---

## 6. Phase 4 — Result store (pass by reference)

**Effort**: ~3 days · **Depends on**: nothing (Phase 2's note store shares infrastructure)
**Priority**: fourth — largest token win

### 6.1 The gap

Every byte a tool produces is serialized into the context window, then progressively
destroyed to make room for the next one:

| Site | Behavior | Cost |
|---|---|---|
| [`_truncate_tool_response`](../src/model/serving/chat.py#L79) | hard cut at `MAX_TOOL_RESPONSE = 16000` chars, head + tail kept | **the middle is gone permanently** |
| [`_decay_old_tool_results`](../src/model/serving/chat.py#L733) | all but the last 3 results trimmed to 6000 chars | older evidence reduced to its opening |
| [`_strip_old_images`](../src/model/serving/chat.py#L766) | base64 replaced with a placeholder in stale messages | image unrecoverable |
| [`_compact_context`](../src/model/serving/chat.py#L2081) | whole-conversation LLM summarization | an extra model call, and detail loss |

The decay marker is the tell:

```
… [trimmed: older tool result, call the tool again for the rest]
```

The recovery path for trimmed data is **re-executing the tool** — a full network round
trip to recover bytes the harness already had. And a research loop that opens six
documents re-sends all six on every subsequent turn, so prompt size grows quadratically
in tool calls while the answer is still being assembled.

### 6.2 Design

Full output goes to disk; a bounded preview plus a handle goes into the context.

```
data_path/.onit/results/0007-local_search.txt      full output, untouched
```

The tool message becomes:

```
[result:0007 · local_search · 48,320 chars · truncated to 2,000 below]
<first 2000 chars>
Full result: result_read("0007", offset=…, limit=…) or result_grep("0007", pattern)
```

Two new harness tools, dispatched in-process exactly like Phase 2's:

- `result_read(handle, offset=0, limit=4000)` → a window of the stored result
- `result_grep(handle, pattern, context=3)` → matching lines with surrounding context

Truncation stops being lossy, and re-reading costs a local file read instead of a network
round trip.

**Shortcut worth taking**: `bash`, `read_file`, and the document tools mostly operate on
files that already exist under `data_path`. For those, the handle can point at the
existing path rather than copying bytes into a second file.

### 6.3 Steps

1. `src/model/serving/results.py` — `ResultStore(data_path)` with `put(tool, text) -> handle`,
   `read(handle, offset, limit)`, `grep(handle, pattern, context)`. Path-jail every handle
   under `data_path/.onit/results/` and reject traversal; per the
   `session-isolation-data-path` memory, `data_path` is a trust boundary and the harness
   value must win over anything the model supplies.
2. Wire into [`_execute_tool`](../src/model/serving/chat.py#L795) at the point where
   `_truncate_tool_response` is called ([chat.py:950](../src/model/serving/chat.py#L950)):
   store first, then build the preview. Leave `_truncate_tool_response` in place for the
   no-store fallback path.
3. Make `_decay_old_tool_results` handle-aware — a decayed result keeps its handle line,
   so the marker points at `result_read` instead of telling the model to re-run the tool.
4. Register the tools in the instruction, same gating pattern as Phase 2.
5. **Retention**: results live under the session's `data_path` and die with it. Do not
   add a global cache — session isolation is load-bearing here.

### 6.4 Acceptance

- New `src/test/test_result_store.py`: round-trip; `offset`/`limit` windowing; `grep` with
  context; **traversal rejected** (`../`, absolute paths, symlinks).
- Extend `src/test/test_chat.py`: a 50k-char tool result puts ≤ ~2.5k chars into
  `messages` and is fully recoverable via `result_read`.
- Regression: prompt tokens across a 6-tool run grow roughly linearly, not quadratically.
  `TurnMetrics` already records `prompt_tokens` per turn — assert against that.

---

## 7. Phase 5 — Code as action

**Effort**: ~1–2 weeks · **Depends on**: Phase 4 · **Priority**: last — largest latency win

### 7.1 The gap

Every step of a multi-step operation costs a full model round trip. The parallel path
helps only when *all* calls are in
[`_READ_ONLY_TOOLS`](../src/model/serving/chat.py#L1947); anything containing a write is
strictly sequential — correctly so, per the comment there: *"a batch of those is a script,
and a script has an order."*

The article's answer is to let the model **write the script**. "Search for X, filter to the
three most recent, read each, extract the totals" is one Python block with a loop; today
it is six or more turns, each paying prefill on a context that grew since the last one.

This is the article's central claim and where OnIt has the most headroom.

### 7.2 Design

A `run_code` tool backed by a **persistent per-session Python interpreter** hosted in the
sandbox container ([`container_launcher.py`](../src/container_launcher.py) is already the
right place). Registered MCP tools are exposed inside it as callable Python functions, so
the model writes:

```python
hits = local_search("Q3 revenue")[:3]
totals = {h.title: extract_tables(h.path) for h in hits}
print(summarize(totals))
```

Only what the model `print`s enters the context. Intermediate values stay as live objects
in the interpreter across calls — which is the article's pass-by-reference, achieved
without giving up process isolation. **This is why Phase 4 comes first**: the same handle
machinery covers the case where a `print` is itself huge.

### 7.3 Steps

1. **Prototype the interpreter first, alone.** Persistent namespace across calls, stdout
   capture, timeout, and a clean way to kill and restart a wedged session. Do not touch
   `chat()` until this stands up on its own.
2. **Generate the tool bindings** from the registry's schemas — each MCP tool becomes a
   Python function whose call marshals to the existing `ToolHandler`. Schema-driven, so
   new MCP servers work with no extra code.
3. **Route the injected trust parameters.** `session_id` and `data_path` are injected by
   the harness at [chat.py:829–832](../src/model/serving/chat.py#L829-L832) and must
   **never** be settable from inside the interpreter. Bind them at generation time and
   drop them from the generated signatures entirely.
4. **Ship it as an additional tool, not a replacement.** `run_code` sits alongside
   `bash` and direct tool calling. Config flag, default off, opt in per deployment.
5. **Benchmark before adopting.** `benchmarks/` on Inspect-AI is the arbiter. Small models
   write worse Python than they write JSON tool calls — this phase can *lose* accuracy on
   8B-class models even as it cuts latency. Measure both, per model class, before making
   it a default anywhere.

### 7.4 Acceptance

- Namespace persists across two `run_code` calls in one session; does **not** leak across
  sessions.
- `session_id` / `data_path` cannot be overridden from inside the interpreter — assert the
  handler receives the harness values regardless of what the code passes.
- A wedged interpreter is killed by timeout and the session recovers.
- Benchmark comparison at equal task set: turns, wall time, and score, on at least one
  small model and one large one.

---

## 8. Phase 6 — Explicit run state

**Effort**: ~3 days · **Depends on**: nothing · **Priority**: schedule with Phase 4

### 8.1 The gap

Every piece of state describing a run lives in local variables inside `chat()` —
`iteration_count`, `tool_call_history`, `planning_continuation_count`,
`ack_continuation_count`, `_force_tool_call`, `_final_answer_prefix`,
`_prose_before_tools` — and dies on return. Consequences:

- **Nothing is inspectable mid-run.** A stalled agent cannot be asked what it has tried.
- **Testing requires driving the whole loop.** There is no way to construct "20 turns in,
  planning budget exhausted" and assert on the next decision.
- **Resume is prose-only.** Session persistence at
  [onit.py:1190](../src/onit.py#L1190) writes `{task, response, timestamp}`. A resumed
  session replays what was *said*, with no record of which tools ran, which files were
  written, or what was tried and failed.
- [`_record_trajectory`](../src/onit.py#L1212) captures some of this but is write-only and
  off unless the autonomy level is `observe` or higher.

### 8.2 Steps

1. `RunState` dataclass in `src/model/serving/state.py` holding the fields above, **owned
   by the caller and mutated in place** — the exact pattern `TurnMetrics` already uses
   with its `metrics` sink ([chat.py:2242](../src/model/serving/chat.py#L2242)). That
   pattern is proven in this codebase; do not invent a second one.
2. Replace the locals in `chat()` with `state.` attributes. Mechanical, and the diff is
   large but shallow — do it in one commit, no behavior change, tests green before and
   after.
3. Accept an optional `run_state` kwarg. Absent, `chat()` makes its own — identical to
   today's behavior, so nothing outside has to change.
4. Persist alongside the session JSONL as `<session_id>.state.json`, and reload on resume
   ([`_setup_session`](../src/onit.py#L868) already handles `resume_session_id`).
5. Feed `tool_call_history` from a resumed state into the instruction, so a continued
   session knows what has already been attempted.
6. **Coordinate with `docs/SELF_IMPROVEMENT.md`.** Layer 0 (trajectory recording) is
   already implemented and wants this same object — `RunState` should be what
   `_record_trajectory` serializes, rather than a parallel structure that drifts from it.

### 8.3 Acceptance

- Extend `src/test/test_chat.py`: a caller-supplied `RunState` is populated during the run
  and readable after it returns.
- New: a `RunState` constructed mid-run (planning budget exhausted) drives the expected
  next decision, without running the preceding turns.
- Extend `src/test/test_onit.py`: resume restores `tool_call_history`.
- `src/test/test_learn.py` still passes — trajectory recording must not regress.

---

## 9. Testing notes

Standing constraints for every phase, from prior sessions:

- **Run pytest from the repo root.** `src/mcp` shadows the pip-installed `mcp` package if
  you run from `src/`, producing a misleading `fastmcp` import error.
- **No new dependencies.** `pip install` is blocked by the user's deny rules — hand-roll
  (the Phase 1 schema validator especially).
- **Mock `config_data` as a real dict with a `tmp_path` `data_path`**, or tests create
  orphan `~/sandbox/<uuid>` directories.
- **Monkeypatch `src.setup.get_secret`** in anything touching credentials; the real
  keychain makes tests flaky.

---

## 10. Open questions

1. **Does any MCP server in `src/mcp/servers/tasks/` declare a non-trivial
   `outputSchema`?** If not, Phase 1 step 5 (output validation) is speculative — strip
   `returns` from the wire and stop there.
2. **Should `run_code` replace `bash` eventually, or coexist permanently?** Coexist for
   now; revisit only with benchmark evidence.
3. **Is `note_write` (Phase 2) redundant once the result store (Phase 4) exists?** They
   serve different things — notes are the model's own words, results are tool output — but
   they should share one on-disk root and one path-jail implementation.
4. **What is the right iteration cap?** 50 is a guess. Pull the real distribution from the
   trajectory store once enough runs are recorded.
