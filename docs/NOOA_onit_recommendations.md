# Recommended Changes for Onit Based on NVIDIA NOOA's Six Capabilities

*Analysis date: August 08, 2026*

The NVIDIA NOOA framework identifies six model-facing interface ideas that drive agent performance, plus a long-term memory subsystem. Here's how each maps to Onit's current codebase and what should change.

---

## 1. Typed Input/Output

**What NOOA says:** Agentic calls have typed arguments and validated return values, not free text. Type annotations are enforced contracts.

**What Onit has today:** Tools are discovered from MCP servers and described via `FunctionSpec` (a `TypedDict` in `src/type/tools.py` with `parameters` and `returns`), but there's no runtime validation of tool arguments or return values. The `chat()` loop in `src/model/serving/chat.py` passes raw dicts between the LLM and tools.

**Recommended change:** Add Pydantic-based runtime validation to tool calls. The `ToolHandler.__call__` in `src/type/tools.py` should validate arguments against the declared JSON schema before dispatch and validate return values after. This would catch malformed tool calls early and give the model structured error feedback instead of opaque failures.

---

## 2. Pass by Reference

**What NOOA says:** The model operates on live Python objects, seeing bounded previews instead of serialized dumps. This eliminates the need for context compaction.

**What Onit has today:** Tool results are serialized as strings and stuffed into the conversation history. There's a crude `MAX_TOOL_RESPONSE = 16000` truncation in `chat.py`, and a context compaction mechanism that summarizes when the context window fills up. The `_truncate_tool_response()` function keeps only the head and tail of large results.

**Recommended change:** Instead of serializing full tool outputs into the message history, store tool results as live Python objects referenced by handle. The model sees a bounded preview (e.g., first N lines + metadata). When the model needs more, it calls a tool to expand the reference. This would dramatically reduce token usage and eliminate the need for the current compaction pipeline. The `MAX_TOOL_RESPONSE` truncation is a lossy stopgap — pass-by-reference is the proper solution.

---

## 3. Code as Action

**What NOOA says:** The model acts by writing Python, with control flow and inline method calls. Methods with `...` bodies are completed at runtime by an LLM-driven loop; methods with normal bodies run as deterministic Python.

**What Onit has today:** The model acts exclusively through MCP tool calls (function calling). It has a `bash` tool for shell commands, but no structured code-as-action capability. The agent cannot write and execute Python with control flow (conditionals, loops, inline method calls).

**Recommended change:** Add a `code` or `python` tool that lets the model write and execute Python snippets with access to the agent's own methods and state. This is the single biggest architectural gap. The model should be able to write something like:

```python
results = []
for url in urls:
    content = fetch(url)
    if "error" not in content:
        results.append(extract_summary(content))
return results[:5]
```

…instead of making N sequential tool calls with the LLM as the control-flow driver. This would collapse multi-turn tool-calling loops into single-turn code execution, saving tokens and latency.

---

## 4. Programmable Loop Engineering

**What NOOA says:** Orchestration loops are ordinary Python, writable by developers and by the model itself.

**What Onit has today:** The orchestration loop is a fixed `while True` inside `chat()` in `src/model/serving/chat.py`. It's hardcoded: call the LLM, execute any tool calls, append results, repeat until the model stops returning tool calls or a limit is hit. There's no way to customize the loop policy per task or per agent.

**Recommended change:** Extract the loop policy into a pluggable strategy. The `chat()` function should accept a loop strategy object (or the model should be able to supply one). A simple task might use a single-turn strategy; a research task might use a multi-turn strategy with different stopping conditions. The `@strategy(PredictStrategy())` decorator pattern from NOOA is directly applicable — different agent methods could use different loop strategies.

---

## 5. Explicit Object State

**What NOOA says:** Durable, typed state lives on the agent object, not just in conversation history. Fields are model-visible and passed by reference.

**What Onit has today:** State lives entirely in the conversation history (the `messages` list in `chat()`). There is no agent object with typed fields. The `learn/` package records trajectories to disk but doesn't inject state back into the agent. The `SELF_IMPROVEMENT.md` doc explicitly acknowledges this: *"OnIt today is a static agent: the prompt, the tool set, the loop policy and the model are fixed at process start."*

**Recommended change:** Introduce an `Agent` base class (similar to NOOA's) with typed, durable fields. An agent's state — preferences, learned facts, ongoing task context — should live on the object and persist across turns and sessions. The `learn/` trajectory store is the right substrate for this; Phase 1 (Recall) and Phase 2 (Playbook) in `SELF_IMPROVEMENT.md` are the implementation plan. The `build_record()` function in `src/learn/trajectory.py` already has the `learned_context` field ready for this.

---

## 6. Model-Callable Harness APIs

**What NOOA says:** Context blocks and event history are APIs the model can inspect and manage. The model can deliberately curate what's in its context.

**What Onit has today:** Context management is entirely harness-side. The model has no visibility into or control over what's in its context window, when compaction happens, or what gets dropped. The compaction logic in `chat.py` is opaque to the model.

**Recommended change:** Expose context management as tools the model can call: `list_context_blocks`, `drop_context_block`, `summarize_context_block`, `get_context_stats`. The model should be able to inspect its own context window and decide what to keep, summarize, or discard. This pairs with the memory subsystem — the model should be able to say "save this fact to memory and remove it from context."

---

## 7. Long-Term Memory Subsystem (the biggest gap)

**What NOOA says:** An agent-curated SQLite store with typed relationships (supports, contradicts, derived-from), background reflection for consolidation, spontaneous memory surfacing, and pass-by-reference to live state.

**What Onit has today:** The `learn/` package (Phase 0) records trajectories as JSONL files — `{task, response, timestamp}` plus tool-call details. This is read-only recording; nothing is injected back. The `SELF_IMPROVEMENT.md` roadmap already plans this as Phase 1 (episodic recall) and Phase 2 (playbook/procedural memory), citing ACE, Reflexion, and Memp as precedents.

**Recommended change:** This is the highest-ROI change and the one most directly validated by NOOA's results (+11.8 points on ARC-AGI-3). Implement:

- **SQLite store** (`src/learn/memory.py`): Replace flat JSONL with a SQLite database supporting typed records, importance scores, tags, and typed relationships.
- **Agent-curated writes**: Give the model tools to write, query, update, and delete memories deliberately.
- **Spontaneous recall**: Automatically surface memories relevant to the current turn into context (Phase 1 — `src/learn/recall.py`).
- **Background reflection**: A periodic pass that merges duplicates, links related records, distills episodes into insights, and prunes stale information.
- **Playbook** (Phase 2 — `src/learn/playbook.py`): An itemized store of learned procedures with delta updates, decay/eviction, and scope isolation, following the ACE paper's approach.

---

## Summary of Priority

| Priority | Change | Onit Files Affected | Effort |
|----------|--------|---------------------|--------|
| **1 (highest)** | Long-term memory (SQLite + recall + playbook) | New: `src/learn/memory.py`, `recall.py`, `playbook.py`, `reflect.py` | Weeks |
| **2** | Code as action (Python execution tool) | New MCP tool or built-in capability | Days |
| **3** | Typed input/output (Pydantic validation) | `src/type/tools.py`, `src/model/serving/chat.py` | Days |
| **4** | Pass by reference (object handles in context) | `src/model/serving/chat.py` | Days–weeks |
| **5** | Explicit object state (Agent base class) | New: `src/agent.py`; changes to `src/onit.py` | Weeks |
| **6** | Programmable loop engineering | `src/model/serving/chat.py` | Days |
| **7** | Model-callable harness APIs | `src/model/serving/chat.py`, new tools | Days |

The good news is that Onit's `SELF_IMPROVEMENT.md` already has a well-researched roadmap that converges with NOOA's recommendations. The trajectory recording (Phase 0) is done. The next step — Phase 2 (Playbook) with Phase 1 (Recall) as a ride-along — is exactly what NOOA's memory subsystem validates as the highest-impact change.

---

## References

- [Six Agent Harness Capabilities for Higher Model Performance — NVIDIA Technical Blog](https://developer.nvidia.com/blog/six-agent-harness-capabilities-for-higher-model-performance/)
- [Onit GitHub repository — sibyl-oracles/onit](https://github.com/sibyl-oracles/onit)
- `src/model/serving/chat.py` — the main orchestration loop
- `src/type/tools.py` — ToolRegistry and ToolHandler
- `src/learn/trajectory.py` — trajectory recording (Phase 0)
- `src/learn/config.py` — autonomy levels
- `docs/SELF_IMPROVEMENT.md` — the existing self-improvement roadmap
