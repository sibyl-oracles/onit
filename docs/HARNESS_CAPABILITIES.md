# Agent Harness Capabilities — Gap Analysis and Build Plan

**Status**: **Phases 1, 2, 3 and 6 implemented** (see §3.5, §4.4, §5.4 and §8.4). Phases 4 and 5 are proposal / RFC.
**Date**: August 2026 · *Revised August 09, 2026 — reconciled with a second review (§11); Phases 1 and 3 shipped. Revised August 18, 2026 — Phase 2 shipped (§4.4), then Phase 6 shipped (§8.4).*
**Source**: [Six Agent Harness Capabilities for Higher Model Performance](https://developer.nvidia.com/blog/six-agent-harness-capabilities-for-higher-model-performance/) (NVIDIA, NOOA framework)
**Companion**: [`SELF_IMPROVEMENT.md`](SELF_IMPROVEMENT.md) — Phases 1–4 and 6 here are
that plan's **Phase 0.5**. Its §1.1 explains why they must land before its baseline is
pinned. Schedule from there, not from this document alone.

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
| 2 | Model-callable harness APIs | ~~None — context management is invisible to the model~~ **Shipped**: `context_status`, `note_write`/`note_read`, and a compaction the model is told about | ~~**Large**~~ | 2 ✅ |
| 3 | Programmable loop engineering | Plain Python loop ✅, but policy constants hardcoded | Small | 3 |
| 4 | Pass by reference | Everything serializes into context; truncation is lossy | **Large** | 4 |
| 5 | Code as action | `bash` only; one model round trip per step | **Large** | 5 |
| 6 | Explicit object state | ~~Run state is locals inside `chat()`, dies on return~~ **Shipped**: `RunState`, caller-owned, persisted per session and read back on resume | ~~Medium~~ | 6 ✅ |

### Build order

Ordered by (payoff ÷ effort), not by the article's numbering. Phases 1–4 and 6 together
are **Phase 0.5** of [`SELF_IMPROVEMENT.md`](SELF_IMPROVEMENT.md#L530) — they must all land
before that plan pins its baseline, because each one changes a metric it records:

```
Phase 1  Typed I/O            ~1 day     fixes a live bug          ┐ ✅ shipped
Phase 3  Loop policy config   ~half day  fixes an unbounded loop   │ ✅ shipped
Phase 2  Harness APIs         ~1 day     small, high leverage      │ ✅ shipped
Phase 6  Explicit run state   ~3 days    unlocks resume + testing  │ ✅ shipped
Phase 4  Result store         ~3 days    largest token win         ┘ Phase 0.5

Phase 5  Code as action       ~1-2 weeks largest latency win       ← after SELF_IMPROVEMENT Phase 2
```

Phases 1, 2, 3 and 6 were independent of each other and of everything else.
Phase 6 landed before Phase 4, so the result store's bookkeeping has a typed home rather
than three more locals in `chat()`. Phase 5 depends on Phase 4 (handles are what keep
interpreter output out of the context).

**Phase 5 is deliberately outside Phase 0.5.** It changes what a "tool call" *is*, so
`tool_calls` per task stops being comparable across the boundary — a baseline pinned before
it cannot survive it. It is also the largest, riskiest item and can lose accuracy on
8B-class models. Either land it after the memory track has an honest reading, or accept
that it forces a baseline re-pin and say so out loud when it does.

**Why this schedule costs nothing.** `SELF_IMPROVEMENT.md` is blocked on trajectory
accrual — its holdout wants ≈100 recorded tasks and `onit learn` reports 10. That is a wait
measured in weeks of real usage. This track needs no data, so it runs inside the wait
rather than competing for the slot after it.

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

   *Correcting an earlier note in this doc:* pydantic is **already a dependency** —
   [`OnIt`](../src/onit.py#L586) is a `BaseModel` — so a second review's suggestion to
   validate with pydantic is not blocked by the no-new-deps rule. It is still the harder
   path: pydantic validates *Python types*, and what arrives here is a **JSON Schema dict
   from an MCP server**, with no JSON-Schema→model path in pydantic core. Building models
   dynamically per tool at discovery time is more machinery than a ~120-line validator over
   the five keywords MCP servers actually emit. Hand-roll it — but for that reason, not for
   a dependency reason. (`jsonschema` itself *is* blocked by the user's deny rules.)
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
- Extend `src/test/test_tool_registry.py`: `parameters_schema()` returns the declared
  schema; the API payload has exactly `type` and `function`, and `function` has exactly
  `name`/`description`/`parameters`.
- Extend `src/test/test_chat.py`: a bad-typed call is refused before dispatch — assert the
  MCP handler was never awaited and the tool message names the parameter.

### 3.5 As shipped

Implemented August 09, 2026. 55 new tests; suite 966 → 1028, all green.

| Piece | Where |
|---|---|
| Schema validator (~200 lines, no new deps) | [`src/lib/schema.py`](../src/lib/schema.py) |
| Coerce + validate before dispatch | [`_execute_tool`](../src/model/serving/chat.py#L927) |
| `returns` stripped from the wire | [`_api_tool_payload`](../src/model/serving/chat.py#L79) |
| Schema accessor | [`ToolRegistry.parameters_schema`](../src/type/tools.py#L400) |
| **`required` preserved at discovery** | [`_build_parameters`](../src/lib/tools.py#L28) |

**Two decisions changed during implementation.**

*The projection lives in the serving layer, not the registry.* The plan put
`get_api_tool_items()` on `ToolRegistry`. But `type/tools.py` is a generic MCP registry,
and "the chat-completions API accepts exactly name/description/parameters" is knowledge
about **one provider's wire format** — it belongs next to the code that builds the request.
It also kept 15 existing test doubles meaningful instead of forcing them to mock a second
method.

*A third bug surfaced, bigger than the one this phase was named for.*
`_build_parameters` reconstructed each tool's schema from `inputSchema` keeping only `type`
and `properties` — **`required` was dropped**. Two silent consequences: the model was never
told which parameters are mandatory, and
[`blank_required_args`](../src/type/tools.py#L426) reads `required` to refuse a call before
dispatch, so it found an empty list for every MCP-discovered tool and **had never fired in
production**. `$defs`/`definitions` were dropped too, dangling any `$ref` a server used.

Honest scope on that fix: OnIt's own tools declare every parameter with a default
(`command: str | None = None`), so FastMCP emits no `required` for them and this changes
nothing for the bundled servers *today*. It is a latent-correctness fix that matters for
any third-party MCP server that does declare `required` — and `blank_required_args` remains
the mechanism that actually catches `bash(command="")`.

**Verified against the live bash MCP server** (12 tools), not just mocks:

```
{'command': 'ls', 'timeout': 300}    -> dispatch
{'command': 'ls', 'timeout': '300'}  -> coerced timeout: '300' → 300
{'command': 'ls', 'cwd': 123}        -> REFUSED: cwd: expected string, got 123
{'command': 'ls', 'timeout': True}   -> REFUSED: timeout: expected integer, got true
```

### 3.6 What the recorded trajectories said, and what it changed

Layer 0 had 42 tasks / 660 turns / 848 tool calls recorded. Checking Phase 1 against them
rather than against imagination changed the design.

**The failure mode was not type errors.** 27 calls failed (3%), 24 of them `read_file` (a
14% failure rate on that one tool). Grouped by argument names, the split is total:

```
FAILED     14x read_file(… offset …)      7x read_file(… limit, offset …)   1x read_file(… limit …)
SUCCEEDED 106x read_file(path)           22x read_file(max_chars, path)    22x read_file(mode, path)
```

`read_file` accepts `path, mode, encoding, max_chars, table_index, output_format,
output_dir, min_size, data_path`. It has no `offset` and no `limit` — the model is reaching
for the signature of a *different harness's* read tool. Not a type error, not a blank
argument: an **invented parameter**, and neither existing check looks for those.

**The first cut of this phase let every one of them through.** `validate_arguments` skipped
unknown parameters, on the stated reasoning that "servers accept extras, and the harness
injects session_id/data_path." Both halves were wrong: FastMCP emits
`additionalProperties: false`, and the harness injects only what `tool_accepts_param`
confirms is declared. Fixed — unknown parameters are refused when the schema forbids them,
and the error **names the parameters the tool does actually take**, which is what turns a
guess into a fix. `additionalProperties` was added to `_SCHEMA_KEYS` so it survives
discovery at all.

**Replaying the 27 recorded failures through the fixed validator: 23 caught pre-dispatch,
4 still round-trip.**

**What that is worth is not the round trip.** Those 27 calls cost 0.9s of wall time in
total — the tool rejected them fast. The cost is the **recovery turn**: the model has to
read the error and try again, and a turn here runs a median 4.0s and 44,080 prompt tokens.
23 avoided turns ≈ **91s of model time and ~1.0M prompt tokens re-prefilled** across those
42 tasks. Refusing locally is cheap; the turn it saves is not.

There is a prevention half too, and it may matter more than the detection half: `required`
and `additionalProperties` now reach the model in the tool schema, so it has the
information needed to not invent a parameter in the first place.

### 3.7 Tool name collisions — fixed

`ToolRegistry` merged every server offering a given tool name into one rotation and
resolved calls with [`random.choice`](../src/type/tools.py#L514). That is right for
**replicas** — the same tool on two hosts, which is load balancing — and wrong for
**collisions**, the same *name* over different parameters. OnIt ships two:

| Tool | Consolidated `tasks/tools` server | `os/bash` server |
|---|---|---|
| [`read_file`](../src/mcp/servers/tasks/tools/mcp_server.py#L234) | `path, mode, encoding, max_chars, table_index, output_format, output_dir, min_size, data_path` | `path, encoding, max_chars, data_path` |
| `search_document` | *(differs)* | *(differs)* |

Two consequences in any deployment running both, and the second is the serious one:

1. `read_file(path=…, mode="tables")` succeeded or failed **on a coin flip**.
2. `get_tool_items()` iterated `handlers`, keyed `name@url` — so the model was shown
   **`read_file` twice, with two different parameter lists, under one name**. That is an
   excellent way to teach a model to invent parameters, which is exactly what §3.6 caught
   it doing.

**The fix** ([`register`](../src/type/tools.py#L369)) compares the *parameter schema* of
each incoming copy against the one already holding the name — descriptions are excluded,
since two servers may word the same tool differently:

- **Same parameters** → a replica. Joins the rotation; `random.choice` keeps load
  balancing across hosts, which was always the intent.
- **Different parameters** → a collision. The first registration keeps the name; the other
  stays reachable through `get_handler_by(name, url)` but is out of the rotation and out of
  the advertised list. `discover_tools` registers in `mcp.servers` order, so **reordering
  the config is how an operator picks the winner** — it is no longer a race between
  servers.

`get_tool_items()` now returns one entry per name, in registration order, and always the
one `__getitem__` will dispatch to: the advertised schema and the executed tool cannot
disagree. Registration order also replaces set iteration, so the tool list is stable run to
run — a reordered list unsettles the model *and* breaks the server's prefix cache.

Collisions are reported at startup rather than resolved silently, because no symptom of one
points back here.

**Verified against both live servers** (worst case: both configured):

```
⚠ 2 tool name collision(s) — same name, different parameters:
  - read_file:       using http://tools:18201/sse, ignoring http://bash:18202/sse
  - search_document: using http://tools:18201/sse, ignoring http://bash:18202/sse

19 unique tools; 26 handlers registered
schemas advertised to the model: 19  (was 26)   duplicate names: none
read_file advertised: path, mode, encoding, max_chars, table_index, …   dispatches to: tools
```

`fetch_content` is defined in two servers as well and is correctly *not* flagged — the two
signatures are identical, so it is a replica.

### 3.8 …and then the duplication itself, removed

Collision *handling* left the duplication in place. The follow-up removed it.

**Why there were two.** `tasks/tools` is a **facade**: it wraps the bash, web-search and
local-search servers and re-registers their functions as one consolidated 14-tool server,
merging related tools behind a `mode` parameter to keep the advertised tool count down.
`read_file(mode=text|tables|images)` fanned out to *three* separate bash/web-search tools;
`search_document(mode=pattern|context)` to two. The sub-servers still registered their own
narrower versions, so each merged tool existed twice at different granularity.

The blocker to unifying them was placement: `extract_pdf_images` lived in the **web-search**
server, so only a server importing that module could offer `mode="images"`. The bash server
could not, which is precisely why the two signatures diverged.

**What changed**, following the pattern `shared.py` already documents — *"server-specific
behavior (path validation…) is injected via callable parameters"*:

| Moved to [`tasks/shared.py`](../src/mcp/servers/tasks/shared.py) | What it is |
|---|---|
| `extract_pdf_images_impl` | the ~108-line body, out of the web-search server |
| `read_file_impl` | the mode router, with the three readers injected |
| `search_document_dispatch_impl` | the mode router, with both searches injected |
| `READ_FILE_DESCRIPTION`, `SEARCH_DOCUMENT_DESCRIPTION` | one description each, shared verbatim — differing text is itself a collision |

Both servers now register the *same* consolidated definition. The registry sees identical
parameter schemas and classifies them as **replicas**, so they join one rotation:

```
collisions: 0        unique tools: 19        schemas advertised: 19
read_file        replicas=2
search_document  replicas=2
```

Verified functionally through both servers — identical results for every mode, including
`read_file(mode="images")`, which the bash server previously could not do at all:

```
bash  read_file(text/tables/images/nonsense) -> success / success / no images found / error
tools read_file(text/tables/images/nonsense) -> success / success / no images found / error
```

**Evidence that this was hurting the model.** The bash `search_document` description had
accumulated three warnings — `Do NOT use 'query'`, `Do NOT use 'context_chars'`,
`Do NOT use 'max_sections'` — naming exactly the parameters of the *facade's*
`search_document`. Someone had patched at the prompt level what was really a name
collision: a model shown both tools under one name, guessing which it would get. Both tools
now accept all of those parameters, and the warnings are deleted along with the condition
that produced them.

The same reading applies to §3.6's `read_file(offset=…, limit=…)`. The shared description
now documents every parameter and states plainly that there is no offset/limit paging — a
test asserts it, since that hallucination was a model filling a documentation gap with
another harness's signature.

**What this does not change.** The facade still exists and still merges tools; that is a
deliberate design choice about tool count, not a defect. What is gone is the *second,
narrower copy* of a merged tool. A bash-only sandbox deployment now offers the full
`read_file`, so the contract is the same in every deployment rather than merely
self-consistent within one.

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

   **Coordinate with the memory track.** This is the harness-level primitive for
   model-written memory; `SELF_IMPROVEMENT.md` Loops A and B are the *learned* version of
   the same idea. From the model's side they are one tool family, and shipping them under
   two naming schemes means it uses neither well. Share the on-disk root and the path-jail
   implementation with Phase 4's result store. **Do not share the scope**: notes are
   session-scoped and die with `data_path`; learned memory is cross-session and lives in
   `~/.onit/learned/`. The path from one to the other is Loop B's Reflector — offline and
   gated — never a shared directory, because `data_path` is a session isolation boundary.
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

### 4.4 As shipped

Implemented August 18, 2026. 58 new tests; suite 1275 → 1333, all green.

| Piece | Where |
|---|---|
| The three tools, their schemas, and the run state they report | [`src/model/serving/harness.py`](../src/model/serving/harness.py) |
| Dispatched ahead of the registry lookup | [`_execute_tool`](../src/model/serving/chat.py) |
| Offered in the same payload as the MCP tools | [`chat()`](../src/model/serving/chat.py) |
| Compaction notice, model-facing | [`_compact_context(harness_note=…)`](../src/model/serving/chat.py) |
| Prompt block, gated and cacheable | [`build_assistant_instruction`](../src/mcp/prompts/prompts.py) |
| `serving.harness_tools` | [`SERVING_PASSTHROUGH`](../src/onit.py#L65) → `chat()` |

**Four decisions changed during implementation.**

*The notice goes inside the compacted message, not after it.* The plan said append a
`user`-role notice to the compacted list. But `_compact_context` ends deliberately on a
`[Resume the task now…]` line, and the comment above it explains what happens when
anything else occupies that position — the model answers the trailing message instead of
resuming. The notice is placed immediately before that line, in the same message, so the
resume instruction stays last.

*A fourth tool was folded into `context_status` rather than added.* The model needs to
know which note keys exist, especially straight after a compaction. That could have been
`note_list`; it is a `notes_saved` field on `context_status` instead. One fewer tool in
every request, and the compaction notice can point at a single tool that answers both
"how bad is it" and "what did I keep".

*The raw-JSON tool-call path had to learn these names.* All three parsers
(`_parse_tool_call_from_content`, `_parse_truncated_tool_call`, `_parse_commands_format`)
gate on `name in tool_registry.tools`, and harness tools are by construction not in the
registry. A weak model that writes `{"name": "note_write", …}` as content — the exact
population this phase is aimed at — would have had it read as prose and handed to the
user as an answer. They now ask `_known_tool_name`, which checks both.

*They are withheld from a run with no tools of its own.* A plain question with no
registry behind it runs one turn and never compacts, so the schemas would be paid for on
every request to buy nothing — and a model told to keep notes in a conversation that
cannot outlive them will keep them. `enabled = bool(tools) and serving.harness_tools`.
The note tools additionally need a `data_path`; without one only `context_status` is
offered, rather than offering a tool that could only fail.

**Two things deliberately not done.**

*No cross-session memory.* Notes live under `data_path` and die with it. The path from
here to durable memory is `SELF_IMPROVEMENT.md`'s Loop B Reflector — offline and gated —
not a shared directory, and this phase does not shorten it.

*No context editing.* The plan's §4.1 named `drop_context_block` and
`summarize_context_block` as the kind of thing the article asks for; neither was built.
Reading the context and writing beside it is what a model can act on today. Letting it
delete from its own message list is a different risk profile, and it should wait for
Phase 6, where the run state has a typed home to be edited through.

**Cost, measured.** The three schemas are 1,150 characters of tool payload (~290 tokens)
and the prompt block is 632 more (~160), so a run that has them pays ~450 tokens on every
request. Both numbers came down once the descriptions stopped restating the prompt block:
a description ships whether or not the block is gated on, so what the tool *is* lives in
the description and when to reach for it lives in the block — the split §3.6 arrived at
for `local_search`, for the same reason. Against that: one compaction is a whole extra
LLM call plus everything the summary drops.

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

### 5.4 As shipped

Implemented August 09, 2026, as specified. All six ceilings are `chat()` kwargs defaulting
to their previous values; `MAX_CHAT_ITERATIONS` now defaults to **50** with `-1` as the
documented opt-out. Config path is `serving:` → `SERVING_PASSTHROUGH`
([`onit.py:63`](../src/onit.py#L63)) → `chat()`, documented inline in
[`src/configs/default.yaml`](../src/configs/default.yaml).

Two additions beyond the plan:

- **`_as_positive_or_disabled`** ([chat.py:98](../src/model/serving/chat.py#L98)) — YAML
  hands back whatever was typed, so `"25"` coerces but `"fifty"` falls back to the
  *default*, never to disabled. A typo must not silently restore the unbounded loop.
- **The turn limit returns partial work.** If prose was already written — a truncated
  answer being resumed, or text streamed before a final tool call — that is returned
  instead of an apology. The user watched it stream past; discarding it because the loop
  ran long is a second failure on top of the first.

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

### 6.5 Baseline impact — read before scheduling

This is the largest baseline shift of the five Phase 0.5 items. The trajectory schema's
`signals` block records `truncations` and `compactions`; a working result store collapses
both toward zero on the *same run*. Any `eval/baseline.json` pinned before this lands is
measuring a different harness. Two consequences:

- It must land **before** `SELF_IMPROVEMENT.md` freezes its holdout, or force a re-pin.
- Do not read a post-Phase-4 drop in `compactions` as a learning win. It is this change.

Corollary worth stating plainly: pass-by-reference **reduces** compaction pressure, it does
not eliminate compaction. A second review claimed it would remove the need for the
compaction pipeline entirely — it will not. Assistant turns and session history still
accumulate, and `_compact_context` stays as the backstop.

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
**Status**: ✅ **Shipped** — see §8.4 for what changed during implementation.

### 8.1 The gap

**First, what is *not* the gap.** A second review concluded there is "no agent object with
typed fields" and proposed building `src/agent.py`. That is wrong and the mistake is
expensive: [`OnIt`](../src/onit.py#L586) is already `class OnIt(BaseModel)` with ~50 typed
`Field()` declarations, and a new `Agent` base class would duplicate it. The accurate
distinction is **configuration state vs. run state**. OnIt's fields are configuration —
`web_port`, `a2a_name`, `theme`, `history_turns` — set at startup and stable for the
process. What is missing is state describing *this run*, and that is a narrower fix inside
`chat()`, not a new class hierarchy.

Every piece of that state lives in local variables inside `chat()` —
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

### 8.4 As shipped

| What | Where |
|---|---|
| `RunState` dataclass | [`src/model/serving/state.py`](../src/model/serving/state.py) |
| The loop reads and writes it | [`chat()`](../src/model/serving/chat.py) — `run_state` kwarg |
| Persisted per session | `<session_id>.state.json`, beside the `.jsonl` |
| Loaded on resume | [`_setup_session`](../src/onit.py) → `OnIt.run_state` |
| Fed back into the instruction | `prior_attempts` → [`build_assistant_instruction`](../src/mcp/prompts/prompts.py) |
| Tests | `test_run_state.py` (30), plus `TestRunState` in `test_chat.py` (7) and `test_onit.py` (10) |

**Four decisions changed during implementation.**

*Four more fields than the plan named.* §8.2 listed seven locals; the block they live in
holds eleven. `_last_prompt_tokens`, `_active_max_tokens`, `_force_compact` and
`_final_continuation_count` are the same category — mutable, run-scoped, dead on return —
and leaving them behind would have meant a state object that could not answer "what will
the next API call look like". `active_max_tokens` is the one that needed care: its opening
value is `max_tokens`, which the state object cannot know, so it defaults to `None` and the
loop fills it in only when nobody else has. A caller constructing a mid-run state keeps
whatever it set.

*Helper signatures were left alone.* `_execute_tool` and the two handlers still take
`tool_call_history: list` rather than the state object. The list is passed by reference, so
`state.tool_call_history` is mutated in place exactly as before — and a signature change
there would have rippled into six test call sites to buy nothing. The cost is that
`RunState` has no `record_tool_call`; the append-and-count lives where it always did.

*A resumed session does not inherit the repeated-call budget.* The obvious reading of step
5 — reload `tool_call_history` into the run — is wrong. `MAX_REPEATED_TOOL_CALLS` counts
against that list, so a session with a long history would spend the budget before the new
task made its first call. Each task gets a fresh `RunState`; the accumulated history is
folded into the session's file afterwards (`merge`) and reaches the model only as prose, in
the instruction. Answer text (`final_answer_prefix`, `prose_before_tools`) is neither merged
nor persisted: a half-written answer belongs to the task that was writing it.

*`stop_reason` was added, and it is what step 6 turned out to mean.* Whether the loop
finished or gave up is invisible in both the session file and the metrics sink — a turn-limit
stop and a completed answer both leave text behind. It is set at the five exits that know
which happened, persisted, reported in the resume note, and passed to `derive_signals` so
the trajectory record and the state file cannot disagree about how a run ended. That is the
"rather than a parallel structure that drifts from it" clause, discharged at the one field
where drift was actually possible.

**Cost.** Nothing on the common path: a new session's `prior_attempts` is `""`, and the
state file is one small write per task. A resumed session pays ~40–80 tokens for the note.

**Not done.** Context editing (§4.4 deferred `drop_context_block` to "Phase 6, where the
run state has a typed home to be edited through"). The home now exists — `messages` is
still a local, and handing the model edit rights over it is a risk profile this phase did
not take on.

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

1. ~~**Does any MCP server declare a non-trivial `outputSchema`?**~~ **Answered during
   Phase 1.** Yes — all 12 tools on the bash server declare one, so `returns` was genuinely
   being populated and shipped. But it totals only **360 characters across all 12 tools**,
   against 5,007 for `parameters`. So the token argument for stripping it, which this
   document leaned on, is weak; the real justification is the **rejection risk** on a
   provider that validates the tool schema strictly. Stripping it was still right, for a
   different reason than stated. Output *validation* (step 5) stays deferred — the schemas
   are thin wrappers with little to check.
2. **Should `run_code` replace `bash` eventually, or coexist permanently?** Coexist for
   now; revisit only with benchmark evidence.
3. **Is `note_write` (Phase 2) redundant once the result store (Phase 4) exists?** They
   serve different things — notes are the model's own words, results are tool output — but
   they should share one on-disk root and one path-jail implementation.
4. **What is the right iteration cap?** 50 is a guess. Pull the real distribution from the
   trajectory store once enough runs are recorded. Note the circularity: today's
   distribution is unbounded *because* the cap is disabled, so the first cap has to be a
   judgment call that later data corrects.
5. **Does the NOOA post report benchmark results?** A second review cites "+11.8 points on
   ARC-AGI-3" for the memory subsystem. The fetch behind this document surfaced only the
   six capabilities — no numbers, no memory section. Confirm before that figure is used to
   justify priority, since it is currently the only quantitative claim on the table.

---

## 11. Reconciliation with the second review

An independent analysis of the same article was produced separately. It agreed on five of
six capabilities in substantially the same terms — including "code as action is the single
biggest architectural gap." Recorded here are the differences that changed this document.

### 11.1 What it got right that this document missed

**A long-term memory subsystem, ranked #1.** This document treated
[`SELF_IMPROVEMENT.md`](SELF_IMPROVEMENT.md) as a footnote in Phase 6; the second review
made it the headline, and it was right to. The convergence is real —
[`SELF_IMPROVEMENT.md`](SELF_IMPROVEMENT.md#L530) already specifies `src/learn/recall.py`
and `src/learn/playbook.py`, and [`trajectory.py:188`](../src/learn/trajectory.py#L188)
already carries `learned_context` with `playbook_version` and `episodes_used`, wired for
exactly this.

Its sharpest point: the highest-impact next step may be **finishing a roadmap that already
exists** rather than starting a new one. That reframing produced §1.1 of
`SELF_IMPROVEMENT.md` and the Phase 0.5 schedule above, which is a better answer than
either document reached alone.

Two caveats retained. Memory is not one of the six capabilities — the second review says so
itself ("plus a long-term memory subsystem"), making it an extension rather than a mapping.
And its **SQLite-everywhere** proposal was resolved as a split, not adopted: the playbook
stays JSON (small, delta-edited, git-diffable, matching invariants I2/I3); trajectories get
a derived, rebuildable SQLite *index* at Phase 1, where retrieval scale actually justifies
it. See `SELF_IMPROVEMENT.md` §4.0.

### 11.2 Corrections carried into this document

| Claim in the second review | Status | Where fixed |
|---|---|---|
| "No agent object with typed fields"; build `src/agent.py` | **Wrong** — `OnIt` is a `BaseModel` with ~50 typed fields | §8.1 |
| Pass-by-reference "eliminates the need for compaction" | **Overclaim** — reduces pressure, doesn't remove the backstop | §6.5 |
| Pydantic validation blocked by no-new-deps | **Wrong** — pydantic is already a dependency; hand-roll for a different reason | §3.3 step 2 |
| Trajectories are `{task, response, timestamp}` | **Conflated** with the *session* JSONL; `build_record` writes per-turn views, signals and metrics | — |
| Loop runs "until a limit is hit" | **Missed** that `MAX_CHAT_ITERATIONS = -1` disables it | §5.1 |
| No validation of tool return values | **Understated** — `returns` is captured, never read, and shipped in the API payload | §3.1 |

### 11.3 The genuine disagreement, and how it resolved

The second review ordered by impact (memory first, weeks of work). This document ordered by
payoff ÷ effort (bug fixes first, days of work). Neither is wrong in isolation:
impact-first risks weeks before anything ships; effort-first defers the largest gain.

The resolution came from a fact neither analysis had used: **the memory track is blocked on
data that does not exist yet.** Its holdout wants ≈100 recorded tasks and there are 10. So
the ordering question was malformed — there is no contested slot. The memory track keeps
its priority; the harness track runs inside the wait it was already going to have. And the
scheduling constraint that fell out of examining both together — that harness changes
redefine the very metrics the memory track baselines against, so they must land *before*
the pin — appears in neither original document.

Where the two still conflict on priority: **the memory track wins on evidence** (ACE and
Reflexion are controlled results on small open models) and **this track wins on
availability** (no data dependency). That asymmetry is the whole argument for running them
in this order, and it is worth re-checking if the NOOA benchmark claim in open question 5
turns out to be real.
