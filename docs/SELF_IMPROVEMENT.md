# Self-Improving OnIt — Literature Review and Scaffold Proposal

**Status**: Phase 0 implemented (see §5). Phase 0.5 and Phases 1–4 are proposal / RFC.
**Date**: August 2026 · *Revised August 09, 2026 — added §1.1 and Phase 0.5 (harness track).*
**Companion**: [`HARNESS_CAPABILITIES.md`](HARNESS_CAPABILITIES.md) — the harness track this
plan now depends on. Read §1.1 before scheduling anything.

---

## 1. Executive summary

OnIt today is a **static agent**: the prompt, the tool set, the loop policy and the model
are fixed at process start, and the only thing that survives a session is a
`{task, response, timestamp}` line in a JSONL file. Every run starts from the same place
and learns nothing from the last thousand.

The SOTA literature converges on a clear split (see §2): **foundation-model improvement**
(changing weights) versus **scaffold improvement** (changing prompts, memory, tools, and
control logic around frozen weights). For OnIt — which serves local/self-hosted models,
often small ones (Qwen3.6-27B, 8B-class), across vLLM/Ollama/OpenRouter — scaffold
improvement is where nearly all the accessible gain is. It needs no gradients, no GPU
budget, and works with API-only endpoints.

This document proposes **four nested improvement loops**, ordered by cycle time and risk,
all hanging off one prerequisite:

| Loop | What it changes | Cycle | Risk | Precedent |
|---|---|---|---|---|
| **0. Experience substrate** ✅ | *nothing* — records full trajectories | per turn | none | prerequisite for all |
| **A. Episodic recall** | retrieved context per task | seconds | low | Reflexion, Dynamic Cheatsheet |
| **B. Playbook (procedural memory)** | the standing instruction | minutes–hours | low-med | ACE, ExpeL, AWM, Memp |
| **C. Skill synthesis** | the tool set (new MCP tools) | days | medium | Alita, Voyager |
| **D. Scaffold evolution** | prompt template + loop code | offline, gated | high | GEPA, DGM, HGM, SICA |

The central thesis: **OnIt cannot improve what it does not record.** Layer 0 is not
optional and not glamorous; it is the whole basis of everything else. Ship it first.

The second thesis: **every promotion must pass a frozen, held-out benchmark before it
goes live.** OnIt already has the fitness function — `benchmarks/` on Inspect-AI. That
harness is the difference between self-improvement and self-delusion.

The third thesis, added in the August 09 revision: **the harness must stop moving before
the baseline is frozen.** See §1.1 — this reorders the plan.

### 1.1 Relationship to the harness track

A separate analysis ([`HARNESS_CAPABILITIES.md`](HARNESS_CAPABILITIES.md)) scored OnIt
against NVIDIA's NOOA capability list and proposed six harness changes: typed I/O,
model-callable context APIs, loop-policy config, a tool-result store, code-as-action, and
explicit run state. Two independent reviews of that article converged on the same gaps,
and both asked the obvious question — *does the harness track compete with this one for
the next slot?*

**It does not, and the reason is a scheduling constraint neither analysis noticed.**

**1. This track is currently blocked; that one is not.** §8 says do Phase 2 next, *"but not
until there are enough trajectories to freeze a holdout from."* The holdout wants ≈100
tasks drawn from real sessions. `onit learn` currently reports 10 across 1 session. Phase 1
and Phase 2 cannot honestly start for weeks of accrual. The harness work needs no data at
all. **The harness track is what happens while trajectories accumulate** — it is not a
competitor for the slot, it is the thing that fills the slot that is empty anyway.

**2. Harness changes invalidate the baseline, so they must land first.** This is the
load-bearing point. The trajectory schema's `signals` block records `truncations` and
`compactions`, and `metrics` records `turn_count`, `tool_calls` and `prompt_tokens_max`.
Every one of those is *defined by harness behavior*:

| Harness change | What it does to the recorded signal |
|---|---|
| Result store (H4) | `truncations` and `compactions` collapse toward zero — same run, different numbers |
| Loop-policy config (H3) | `turn_count` gains a real ceiling; today `MAX_CHAT_ITERATIONS = -1` means there is none |
| Typed I/O (H1) | Malformed calls are refused pre-dispatch, so `tool_errors` shifts from server errors to harness refusals |
| Code as action (H5) | One `run_code` call replaces N tool calls — `tool_calls` per task is no longer comparable at all |

A baseline pinned in `eval/baseline.json` before these land is not comparable to anything
measured after them, and I4 (quality-gated efficiency) would be reading two different
rulers. Freezing the holdout early was already deferred for lack of data; it must now
*also* wait for the harness to settle.

**3. Three harness phases are prerequisites here, not detours.** They were designed
independently and land on this plan's own gap list:

- **H6 (explicit run state)** builds the `RunState` object that `_record_trajectory` should
  serialize. §4.1 already says the Layer 0 change is *"to persist the sink rather than
  log-and-drop it"* — H6 generalizes exactly that, and closes gap 2's dead `memories`
  parameter by giving it somewhere typed to live.
- **H2 (`note_write` / `note_read`)** is the harness-level primitive for durable
  model-written memory. Loop A and Loop B are the *learned* version of the same idea. They
  must share one on-disk root and one path-jail — see §4.0.
- **H1 (typed I/O)** fixes two live bugs that corrupt Layer 0's own data: `outputSchema` is
  captured at [`lib/tools.py:182`](../src/lib/tools.py#L182), never read, and shipped
  inside the `tools=` payload where it is not a valid field; and the disabled iteration cap
  lets `turn_count` run unbounded.

**Revised order**: Phase 0 (done) → **Phase 0.5 (harness, §5)** → freeze holdout → Phase 1
→ Phase 2. See §8.

---

## 2. Literature review

### 2.1 The organizing taxonomy

Three 2025–2026 surveys frame the field, and they agree on the axes:

- **[Self-Improvements in Modern Agentic Systems: A Survey](https://arxiv.org/abs/2607.13104)** (2026)
  splits the field into **foundation-model improvement (FM-θ)** — self-generated
  finetuning data, self-evaluative feedback, exploratory experience — and **scaffold
  improvement (Σ)** — prompt optimization, memory architecture, tool governance, and
  full-scaffold redesign. It recommends a **fast/slow loop** design (rapid scaffold
  exploration, slow parametric consolidation) with **layered gating** for safety.
- **[A Survey of Self-Evolving Agents](https://arxiv.org/abs/2507.21046)** (Gao et al., 2025)
  organizes around *what / when / how / where* to evolve.
- **[A Comprehensive Survey of Self-Evolving AI Agents](https://arxiv.org/abs/2508.07407)**
  (Fang et al., 2025) frames evolution as automatic enhancement from interaction data
  and environmental feedback.

The survey's own conclusion is the one that matters most for OnIt: the **memory problem
is the central infrastructure challenge**, and scaffold-level gains are available today
without touching weights.

### 2.2 Experience & memory (the highest ROI for OnIt)

| Work | Idea | Relevance to OnIt |
|---|---|---|
| [Reflexion](https://arxiv.org/abs/2303.11366) (2023) | Verbal self-reflection on failure stored in an episodic buffer, replayed on retry | The minimal viable loop; ~1 day of work on top of Layer 0 |
| [ExpeL](https://arxiv.org/abs/2308.10144) (2023) | Cross-task *insight extraction* into natural-language rules, no gradients | The model for distilling many sessions into a few durable rules |
| [Agent Workflow Memory](https://arxiv.org/abs/2409.07429) (2024) | Induce reusable *workflows* from trajectories, offline **and** online | Directly applicable: OnIt's tool sequences are workflows |
| [Dynamic Cheatsheet](https://arxiv.org/abs/2504.07952) (2025) | Persistent adaptive memory of strategies/snippets for black-box models at test time | Works with API-only endpoints — matches OnIt's OpenRouter/Ollama paths |
| [Memp](https://arxiv.org/abs/2508.06433) (2025) | Procedural memory at two granularities (step-level + script-level), with **update/correct/deprecate** | The eviction/decay policy OnIt needs to stop playbook bloat |
| [MemGPT](https://arxiv.org/abs/2310.08560) (2023) | OS-style paged memory hierarchy | Conceptual framing for OnIt's existing context compaction |

**[ACE — Agentic Context Engineering](https://arxiv.org/abs/2510.04618)** (2025, ICLR-track)
is the single most directly transplantable paper. It defines three roles — **Generator**
(runs the trajectory), **Reflector** (distills concrete lessons from successes and
errors), **Curator** (merges lessons into structured context) — and identifies the two
failure modes that kill naive memory systems:

- **brevity bias** — summarizing away the domain detail that was the whole point;
- **context collapse** — iterative full rewrites eroding accumulated knowledge.

Its fix is **delta updates over an itemized playbook** plus **grow-and-refine**, never a
destructive rewrite. Reported +10.6% on agent benchmarks and +8.6% on finance,
**without labeled supervision** (execution feedback only), with lower adaptation latency
and rollout cost than prompt-optimization or finetuning baselines. Small open-source
models with ACE matched top production agents on AppWorld.

### 2.3 Tool & skill creation

- **[Voyager](https://arxiv.org/abs/2305.16291)** (2023) — the original *skill library*:
  the agent writes executable skills, verifies them in the environment, and stores them
  for compositional reuse.
- **[Alita](https://arxiv.org/abs/2505.20286)** (2025) — *minimal predefinition, maximal
  self-evolution*: the agent autonomously generates **MCP servers** for capabilities it
  lacks, then reuses them. 75.15% pass@1 on GAIA validation; the MCP-creation component
  alone was worth ~15% pass@1; **generated MCPs transfer to other agents and lift smaller
  models**. This is a 1:1 architectural match for OnIt, which is already MCP-native.
- 2026 follow-ons — [AutoSkill](https://arxiv.org/abs/2603.01145),
  [SkillX](https://arxiv.org/abs/2604.04804),
  [From Raw Experience to Skill Consumption](https://arxiv.org/abs/2605.23899) — study
  skill *lifecycle* management: construction, retrieval, deprecation, and the failure
  mode where a bloated skill library degrades selection accuracy.

### 2.4 Prompt & scaffold optimization

- **[DSPy](https://arxiv.org/abs/2310.03714)** / **[TextGrad](https://arxiv.org/abs/2406.07496)**
  — declarative pipelines and "textual gradients"; the compiler view of prompt tuning.
- **[GEPA](https://arxiv.org/abs/2507.19457)** (2025, ICLR 2026 Oral) — **the recommended
  optimizer for OnIt.** Samples trajectories (reasoning, tool calls, tool outputs),
  reflects in natural language to diagnose failures, mutates prompts, and maintains a
  **Pareto frontier** over per-instance scores rather than a single best prompt (which is
  what prevents collapse into a local optimum). Beat GRPO by ~10% average / 20% peak
  using **up to 35× fewer rollouts**. Sample efficiency is exactly the constraint on a
  single-GPU or CPU-hosted deployment.
- **[ADAS](https://arxiv.org/abs/2408.08435)** (2024) — Meta Agent Search: a meta-agent
  programs new agents in code, accumulating an archive.

### 2.5 Code-level self-modification (the deep end)

- **[SICA](https://arxiv.org/abs/2504.15228)** (2025) — a coding agent that edits **its own
  codebase**, collapsing ADAS's meta-agent/target-agent split. Reported 17–53% gains on
  SWE-bench Verified; the improvements it found were *scaffold* improvements — better
  edit tools, context management, peer review.
- **[Darwin Gödel Machine](https://arxiv.org/abs/2505.22954)** (2025, Sakana/UBC) — the
  reference design: an **archive** of agents, sample a parent, let an FM propose a
  self-modification, evaluate on a benchmark, add to the archive regardless of whether it
  is the new best (open-endedness). SWE-bench 20.0% → 50.0%, Polyglot 14.2% → 30.7%.
  Critically, DGM's authors documented **reward hacking** — the agent learned to fake
  tool-use logs — and sandboxed everything.
- **[Huxley-Gödel Machine](https://arxiv.org/abs/2510.21614)** (2025) — fixes DGM's
  **metaproductivity–performance mismatch**: an agent's benchmark score is a *bad*
  predictor of how good a parent it is. HGM selects parents by **Clade Metaproductivity
  (CMP)** — aggregate performance of the whole subtree of descendants — reaching
  human-level coding-agent performance with fewer CPU hours. **If OnIt ever builds Loop D,
  select parents by CMP, not by score.**
- **[Red Queen Gödel Machine](https://arxiv.org/abs/2606.26294)** (2026) — co-evolves the
  evaluator with the agent, because a static evaluator gets gamed.

### 2.6 Parametric self-improvement (deferred, but noted)

- **[SEAL](https://arxiv.org/abs/2506.10943)** (2025, MIT) — the model generates its own
  finetuning data and update directives ("self-edits"); an outer RL loop rewards
  self-edits by the post-update downstream performance.
- **[Self-Rewarding LMs](https://arxiv.org/abs/2401.10020)** (2024),
  **[Absolute Zero](https://arxiv.org/abs/2505.03335)** (2025) — the model as its own
  judge / its own curriculum.

**Recommendation: out of scope for v1.** OnIt is a *harness*, not a training stack; it
serves models it does not own. A LoRA-distillation track is sketched in §7 as future work.

### 2.7 Safety, reward hacking, evaluation

This is not a footnote — it is a design constraint.

- **[Reward Hacking in LM Agents: Revisiting AI Safety Gridworlds](https://arxiv.org/abs/2606.15385)**
  (2026) — specification gaming emerges **zero-shot** across frontier and mid-scale
  models: high observed reward, degraded hidden objective. You do not need a
  superintelligence to get reward hacking; you need a proxy metric.
- **[Safety in Self-Evolving LLM Agent Systems](https://arxiv.org/abs/2606.23075)** (2026)
  — threat amplification across evolution cycles.
- **Benchmarks**: [LifelongAgentBench](https://arxiv.org/abs/2505.11942) (2025),
  [SIP-Bench](https://arxiv.org/abs/2601.20882) (2026) for longitudinal self-improvement,
  [TheAgentCompany](https://arxiv.org/abs/2412.14161) (2024) for consequential tasks.
  Benchmarks for *self-improvement* are themselves immature and gameable.
- The 2026 survey's open problems, restated as OnIt's acceptance criteria: signal quality
  (real gain vs. distribution shift), stability/forgetting, containment, evaluation
  robustness, and credit assignment.

**Design consequence**: proxy rewards (fewer turns, fewer tokens) must *always* be paired
with a quality gate, or the optimizer will discover that the cheapest correct-looking
answer is no answer at all.

---

## 3. Where OnIt stands today

An honest audit. Findings are file-referenced so they can be verified.

**What already exists and is directly reusable:**

| Asset | Location | Why it matters |
|---|---|---|
| Rich per-run telemetry | [`TurnMetrics`](../src/model/serving/chat.py#L95) — turns, tool calls, prefill/decode split, compactions, peak prompt tokens | Free efficiency features for the reward model — already computed |
| Benchmark harness | [`benchmarks/run.py`](../benchmarks/run.py#L29) — Inspect-AI: gsm8k, humaneval, mbpp, bigcodebench, livecodebench, SWE-bench runner | **The fitness function.** This is the single biggest head start |
| LLM-as-judge | [`benchmarks/scorers/onit_judge.py`](../benchmarks/scorers/onit_judge.py) | Quality gate for open-ended tasks |
| Results ledger | [`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md) | Where promotion evidence gets written |
| MCP tool discovery | [`discover_tools()`](../src/lib/tools.py#L213), [`ToolRegistry`](../src/type/tools.py#L347) | Skills can ship as an MCP server — no new plumbing concept |
| Containment | [`command_policy.py`](../src/mcp/servers/tasks/os/bash/command_policy.py), `installs_sealed()`, [`container_launcher.py`](../src/container_launcher.py) | Sandbox for validating self-generated code already exists |
| Prompt templating | [`build_assistant_instruction()`](../src/mcp/prompts/prompts.py#L43) with YAML `template_path` override | Evolved prompts have a place to land |
| Session index / ownership | [`sessions.py`](../src/sessions.py), `get_session_owner()` | Per-user memory scoping is already possible |

**The gaps** (1–3 closed by Phase 0; see §5):

1. ~~**No trajectory persistence.**~~ *Closed.* The session file still holds
   `{task, response, timestamp}`; [`src/learn/`](../src/learn/) now writes the run
   alongside it.
2. ~~**`memories` is a dead parameter.**~~ *Still dead, deliberately.* It is threaded from
   `onit.py` (`'memories': None`) through
   [`chat.py`](../src/model/serving/chat.py#L1806) into
   [`_build_messages()`](../src/model/serving/chat.py#L959), whose body never references
   it. Phase 0 records but injects nothing, so the hook stays unwired until Phase 1 —
   which is what makes level `observe` byte-for-byte identical to learning off.
3. ~~**No outcome signal.**~~ *Partly closed.* Tool errors, retries, truncations and
   compactions are derived per run; `POST /api/rating` records the human verdict. No
   verifier and no implicit signal yet.
4. **Tool set is frozen at startup.** `ToolRegistry.register()` exists but is only called
   during startup discovery; there is no runtime path to add a tool.
5. **No holdout discipline.** `benchmarks/RESULTS.md` mixes hosts and models with no
   pinned baseline for A/B comparison. Benchmark runs now stamp the autonomy level they
   ran at, which is the precondition for one.

**The one constraint that shapes everything:** the instruction is deliberately split into
a **static, prefix-cacheable half** and a **volatile half**
([`prompts.py`](../src/mcp/prompts/prompts.py#L365-L390)), so a server with prefix caching
skips prefilling the preamble on every request of every session. Any learned context
injected *in the middle* of that prefix invalidates the cache from that byte onward,
across all sessions. This is a real cost and it dictates the placement decisions in §4.2.

Custom templates are split on the same rule, decided by what a template interpolates
rather than by the fact that it is custom: one naming `{task}`, `{current_date}` or
`{data_path}` puts its preamble in the volatile half, a fixed one keeps the prefix. The
standing blocks appended to it stay cacheable either way. This matters for Loop D — an
optimizer's output is a fixed preamble, and under the old behaviour it would have
forfeited prefix caching and read as a latency regression.

---

## 4. Proposal: the self-improvement scaffold

### 4.0 Layout and invariants

One directory, one manifest:

```
~/.onit/learned/            # `learn.path`, or ONIT_LEARN_PATH
  pinned.yaml               # THE source of truth for what is live. Rollback point.
  trajectories/*.jsonl      # Layer 0 — raw experience              [implemented]
  playbook/
    global.json             # Loop B — itemized bullets w/ counters + provenance
    user/<owner-hash>.json  #          per-owner scope (web multi-user)
  skills/<name>/            # Loop C — code + manifest.yaml + tests
  scaffold/
    archive/<gen>-<id>/     # Loop D — prompt/code variants + fitness records
  eval/
    holdout.jsonl           # frozen; the improvement loops MUST NOT read this
    baseline.json           # pinned scores for A/B
```

**Not in this tree, and deliberately so** — the harness track's per-session stores
(`data_path/.onit/notes/`, `data_path/.onit/results/`) live under the **session's**
`data_path` and die with it. They are working memory for one run, not learned knowledge.
Keeping them out of `~/.onit/learned/` is what preserves session isolation: `data_path` is
a trust boundary, and a note written by one session must not be readable by another. The
promotion path from the first to the second is Loop B's Reflector — deliberate, offline,
and gated — never a shared directory.

**On SQLite.** A reviewer proposed replacing the flat JSON/JSONL stores with SQLite
throughout. Resolved as a split:

- **Playbook stays JSON** (`playbook/global.json`). It is small, hand-auditable, edited by
  delta ops (I2), rolled back by file (I3), and diffs usefully in review. SQLite would cost
  all four and buy nothing at this scale.
- **Trajectories get a SQLite *index* at Phase 1**, not a replacement. `trajectories/*.jsonl`
  stays the append-only source of truth; Loop A's retrieval over thousands of records is
  where a linear scan stops being acceptable. Index is derived, disposable, and rebuildable
  from the JSONL — so it never becomes a second source of truth.

**Invariants** (violating any of these turns self-improvement into self-delusion):

- **I1 — Frozen holdout.** No loop reads `eval/holdout.jsonl`. Ever. Promotion is decided
  on it; iteration happens on a separate dev split.
- **I2 — Delta-only updates.** Playbooks and skill libraries are edited by
  insert/update/deprecate on identified items — never regenerated wholesale (ACE:
  context collapse).
- **I3 — Pinned + reversible.** Everything live is named in `pinned.yaml`. `onit learn
  --rollback` restores the previous pin. No in-place mutation of live artifacts.
- **I4 — Quality-gated efficiency.** A change that reduces turns/tokens is accepted only
  if a quality score does not regress. Never optimize a proxy alone.
- **I5 — Off by default in benchmarks.** `ONIT_LEARN` state is recorded in every result
  row; baselines run with learning off.
- **I6 — Sandboxed generation.** Self-generated code is written, executed and tested only
  inside the container path, under the existing command policy.

### 4.1 Layer 0 — the experience substrate *(prerequisite, ~1 week)*

Extend the session record from a chat log to a **trajectory record**. New sink
`~/.onit/learned/trajectories/<session_id>.jsonl`, one object per completed task:

```json
{
  "schema": 1,
  "session_id": "...", "turn": 3, "ts": "2026-08-03T09:12:44Z",
  "owner": "sha256:...", "task": "...", "response": "...",
  "topic_hint": "...", "tools_available": ["local_search", "bash", "..."],
  "trajectory": [
    {"n": 1, "tools": [{"name": "local_search", "args_digest": "...",
                        "ok": true, "ms": 812, "result_chars": 4102}],
     "prompt_tokens": 5120, "completion_tokens": 240, "finish_reason": "tool_calls"}
  ],
  "metrics": {"turn_count": 4, "tool_calls": 6, "model_s": 22.1, "tool_s": 9.4,
              "compactions": 1, "prompt_tokens_max": 41002},
  "signals": {"tool_errors": 1, "retries": 0, "truncations": 0,
              "user_rating": null, "verifier": null},
  "learned_context": {"playbook_version": null, "episodes_used": []}
}
```

Implementation notes:

- `TurnMetrics` already produces `metrics` and per-turn `tools` — the change is to
  **persist the sink** rather than log-and-drop it
  ([`onit.py:1150-1154`](../src/onit.py#L1150-L1154)).
- `_execute_tool()` ([`chat.py:777`](../src/model/serving/chat.py#L777)) is the single
  place to capture per-tool ok/error/duration.
- **Store argument digests, not arguments.** Tool args carry file paths, queries and
  occasionally credentials. Hash by default; raw capture behind an explicit opt-in flag.
- `learned_context` closes the credit-assignment loop: you can later ask *"did runs that
  used playbook v7 do better than those that didn't?"* — which is the only way Loop B
  gets an honest verdict.
- Writing is best-effort and off the hot path (append, fire-and-forget), matching the
  existing `try/except: pass` posture around session writes.

**Also in Layer 0 — outcome signals**, cheapest first:

| Signal | Source | Cost |
|---|---|---|
| Tool error / retry / truncation counts | already in the loop | free |
| Efficiency (turns, tool calls, compactions, peak tokens) | `TurnMetrics` | free |
| Execution feedback (exit codes, test pass/fail) | sandbox `bash` results | free, strongest where available |
| Implicit user signal | next-turn rephrase ≈ failure; acceptance/thanks ≈ success | small heuristic |
| Explicit rating | 👍/👎 endpoint in [`src/ui/api.py`](../src/ui/api.py) | ~half a day, highest value per bit |
| LLM judge | `onit_judge` on a sample | tokens |

### 4.2 Loop A — episodic recall *(low risk, ~3 days)*

Retrieve the *k* most similar prior trajectories for the current task and inject their
distilled outcome ("what worked / what failed"), Reflexion- and Dynamic-Cheatsheet-style.

- **Wire the dead hook**: implement `memories` in
  [`_build_messages()`](../src/model/serving/chat.py#L959) and populate it at
  [`onit.py:1085`](../src/onit.py#L1085).
- **Placement**: a compact block in the **volatile half**, alongside the task — episodes
  are per-task and would thrash the cache if placed in the prefix.
- **Retrieval**: start with BM25/TF-IDF over the task text. OnIt already ships local
  search infrastructure ([`local/search/toolkit.py`](../src/mcp/servers/tasks/local/search/toolkit.py));
  reuse it before reaching for embeddings. Per user's constraint, no new pip deps —
  hand-roll or reuse what's vendored.
- **Budget**: hard cap (e.g. 800 tokens, k ≤ 3). Memory that crowds out the task is a
  regression.
- **Kill switch**: `ONIT_LEARN=0` / `learn: false` in `default.yaml` bypasses everything.

### 4.3 Loop B — the playbook *(the main event, ~3 weeks)*

An ACE-style **evolving playbook** of durable, itemized tactics, distilled from
trajectories by a Reflector/Curator pair, injected into the standing instruction.

**Item schema** (one bullet):

```json
{"id": "pb-0142", "scope": "global|owner:<hash>|topic:<slug>",
 "text": "When local_search returns a file whose *name* matches the query but whose "
         "excerpts look generic, open it anyway — the opening is a poor proxy for tail content.",
 "trigger": "local_search",
 "born": "2026-08-01", "last_used": "2026-08-03",
 "used": 41, "helped": 12, "hurt": 1, "status": "active|probation|deprecated",
 "provenance": ["traj:sess-a1#3", "traj:sess-b7#1"]}
```

**The three roles** map onto OnIt cleanly:

- **Generator** — the existing `chat()` loop. No change beyond Layer 0 logging.
- **Reflector** — an offline pass over recent trajectories: cluster by task/tool
  signature, contrast successes against failures, emit *candidate* deltas. Runs on the
  same endpoints via the existing `LoadBalancer`; on a second host if configured.
- **Curator** — applies deltas (insert / update / deprecate), enforces the token budget,
  and **never rewrites the playbook wholesale** (I2).

**Placement — this is the load-bearing decision.** The playbook goes at the **tail of the
static half**, immediately before `INSTRUCTION_SPLIT`
([`prompts.py:365-368`](../src/mcp/prompts/prompts.py#L365-L368)):

```
[ role · procedure · standing rules ]  ← unchanged bytes, still prefix-cached
[ PLAYBOOK v7                       ]  ← changes only between playbook versions
INSTRUCTION_SPLIT
[ date · cwd · file server · task   ]  ← volatile, as today
```

Everything before the playbook keeps its cache hit; only the playbook tail and the
volatile block get re-prefilled. And the **playbook version is pinned per session** — a
mid-session swap would invalidate the prefix for the rest of that conversation.

**Lifecycle** (Memp's update/correct/deprecate, made concrete):

- New bullets enter on `probation` and are injected for a sampled fraction of runs.
- `helped`/`hurt` are attributed via the `learned_context.episodes_used` field from
  Layer 0 — compare matched runs with and without the bullet.
- Decay: `active → probation` when unused for N days; `probation → deprecated` when
  `hurt > helped` or after M runs with no measured effect. Deprecated bullets are kept
  (audit trail) but not injected.
- Hard cap on total injected tokens; when over budget, evict by lowest
  `helped/used` × recency.

**Privacy**: bullets distilled from one user's sessions must never leak into another's.
Default scope is `owner:<hash>`; promotion to `global` requires (a) the bullet to be
derived from ≥3 distinct owners or from benchmark runs, and (b) a redaction pass. Given
`web_allowed_emails` multi-tenancy, treat this as a hard requirement, not a nicety.

**Acceptance**: playbook-on vs playbook-off on the frozen holdout + at least two
`benchmarks/` tasks, recorded in `RESULTS.md` with `ONIT_LEARN` state.

### 4.4 Loop C — skill synthesis *(~4 weeks)*

Alita's insight, in OnIt's own idiom: when the agent repeatedly solves the same subtask
with an ad-hoc bash/Python fragment, **promote the fragment to an MCP tool**.

- **Mining**: Layer 0 trajectories cluster naturally by tool signature. A recurring
  `bash` fragment across ≥N sessions is a skill candidate.
- **Synthesis**: the agent writes `~/.onit/learned/skills/<name>/` — `tool.py`, a
  `manifest.yaml` (name, description, JSON schema, dependencies), and `test_<name>.py`.
  The generated description is not an afterthought: tool-selection accuracy is mostly a
  function of description quality.
- **Validation gate** (all must pass, all inside the container per I6):
  1. tests pass in the sandbox;
  2. **replay** — the skill reproduces the outcomes of the trajectories it was mined
     from;
  3. no new dependency unless `installs_sealed()` is false and the operator approves;
  4. a static check against `command_policy.py`;
  5. benchmark delta ≥ 0 on the holdout — a new tool that adds nothing still costs
     schema tokens and dilutes selection accuracy.
- **Serving**: a new **`SkillsMCPServer`** (`src/mcp/servers/tasks/skills/`, port 18203)
  registered in [`src/mcp/servers/configs/default.yaml`](../src/mcp/servers/configs/default.yaml)
  and [`src/configs/default.yaml`](../src/configs/default.yaml), loading only skills
  listed in `pinned.yaml`. **No change to `ToolRegistry` semantics is needed** — the
  registry discovers it like any other MCP server. Hot-reload comes later; a restart is
  acceptable for v1.
- **Transfer**: per Alita, generated MCPs are reusable across agents and lift smaller
  models. A shipped skill library is a real product asset, not just an internal
  optimization.
- **Evaluation**: this is what [`benchmarks/tasks/agentic.py`](../benchmarks/tasks/agentic.py)
  (GAIA, currently scaffolded but unwired) exists for. Wire GAIA as part of Loop C.

### 4.5 Loop D — scaffold evolution *(offline, gated, ~6 weeks, do last)*

Two sub-tracks, in order:

**D1 — Prompt evolution (GEPA).** Optimize `build_assistant_instruction()`'s blocks —
research procedure, instructions, sandbox workflow — against `benchmarks/` tasks.
GEPA fits OnIt's budget precisely: reflective mutation from *trajectory text* (which
Layer 0 now stores), Pareto-front retention over per-instance scores, ~35× fewer rollouts
than RL. Output is a YAML `instruction_template` — an artifact
[`prompts.py`](../src/mcp/prompts/prompts.py#L125-L136) already knows how to load, and
which now keeps its prefix cache when it interpolates nothing volatile (§3).

**D2 — Code-level self-modification (DGM/HGM).** The agent proposes patches to its own
loop policies — compaction thresholds, retry logic, tool-result decay, parallel-call
heuristics in `chat.py` — evaluated on `benchmarks/`, archived in
`scaffold/archive/<gen>-<id>/`.

Non-negotiables for D2:

- **Select parents by CMP, not by score** (HGM): benchmark performance is a poor
  predictor of a variant's value as a parent. This is the field's most recent and most
  actionable finding.
- **Keep the archive open-ended** (DGM): retain interesting-but-not-best variants;
  greedy hill-climbing stalls.
- **Never patch the running process.** Variants are evaluated in a worktree/container;
  promotion is a human-reviewed PR.
- **Assume reward hacking.** DGM's agent faked tool-use logs. Hold out an unseen suite,
  rotate the judge, and inspect the diff of anything that improves suspiciously fast.
- **Scope the blast radius**: an allowlist of files Loop D may touch. `command_policy.py`,
  auth, and the containment layer are **never** on it.

### 4.6 Autonomy levels

One knob, `learn.autonomy` in `default.yaml`, defaulting to 1:

| Level | Behavior |
|---|---|
| **0 — off** | No recording, no injection. Reproducible baseline |
| **1 — observe** *(default)* | Layer 0 records; nothing is injected. Safe to ship broadly |
| **2 — adapt** | Loops A + B live; playbook promotions require `onit learn --promote` |
| **3 — extend** | Loop C live; skills auto-promote on a green gate, operator notified |
| **4 — evolve** | Loop D; **always** human-reviewed PR, never auto-merge |

---

## 5. Implementation plan

Phases are sequential; each ends in something shippable and measurable.

### Phase 0 — Instrument · autonomy 1 — **implemented**

Shipped:

- [`src/learn/config.py`](../src/learn/config.py) — the autonomy ladder, resolved from the
  `learn:` config block or `ONIT_LEARN`. Levels above what exists are capped, so a config
  asking for `evolve` today gets `observe` rather than a run that believes it is evolving.
- [`src/learn/trajectory.py`](../src/learn/trajectory.py) — schema v1, append-only writer,
  argument redaction, owner hashing, ratings folded in at read time.
- Per-tool ok/error/duration/result-size from
  [`_execute_tool()`](../src/model/serving/chat.py#L777) — every path that answers a call
  records one, failures included — plus an API-retry counter on `TurnMetrics`.
- Recording wired into all three task paths in [`onit.py`](../src/onit.py): `process_task`
  (web/A2A), the interactive CLI, and loop mode.
- 👍/👎 under every answer in the web UI, backed by `POST /api/rating`. The buttons are
  quiet until hovered, restore the verdict already given when history reloads, and a
  second click on the same thumb retracts it. They are not shown at all when recording is
  off — `/api/config` carries `rating_enabled` — so no click is ever silently dropped.
- `onit learn` — the answer to *"is any of this working?"*. Prints the autonomy level,
  the store path, what has been recorded, and the tool table ranked by failure rate.
  `--json` for machines, `--session <id>` to dump one session's records. Read-only, and
  deliberately usable when the model endpoint is down.
- Benchmarks default to `learn=off` and stamp the level into `run_meta.json` next to the
  logs, which `benchmarks/report.py` surfaces in the summary (I5).
- Tests: [`test_learn.py`](../src/test/test_learn.py) (51), plus tool-record coverage in
  `test_chat_metrics.py`, the rating endpoint in `test_web_api.py`, and the `onit.py`
  wiring in `test_onit.py`.

Deferred, now on **two** gates rather than one:

- **Freeze `eval/holdout.jsonl` and pin `eval/baseline.json`.** The holdout is supposed to
  be ≈100 tasks drawn from real sessions; there are no recorded sessions to draw from
  until this has been running for a while. Do it after a few weeks of trajectories, and
  before Phase 1 ships anything that could be tuned against it.
  - **Gate 1 — data.** ≈100 tasks recorded. Currently 10.
  - **Gate 2 — a settled harness.** Phase 0.5 complete. A baseline pinned while the
    harness is still changing measures the harness, not the learning (§1.1). Of the two
    gates this is the one that is easy to forget, because the data gate is visible in
    `onit learn` and this one is not.

**Exit criterion**: a week of normal use yields trajectories rich enough to answer *"which
tool fails most, and on what?"* — a useful result even if every later phase is cancelled.
`onit learn` answers it:

```
Recorded     10 task(s) across 1 session(s)
Trouble      20 tool error(s), 0 API retry/retries, 0 truncation(s)

Tools (worst failure rate first)
  tool                      calls  errors   fail   avg ms
  read_file                   129      20   16%       20
  bash                        300       0    0%    1,246
```

Recording also surfaced a pre-existing telemetry gap it depends on: the Ollama
*streaming* path read `prompt_eval_count` but never `eval_count`, so every streamed
Ollama run accounted for zero generated tokens. Fixed in
[`_ollama_process_streaming_response`](../src/model/serving/chat.py#L1198); the
non-streaming path had always read it.

### Phase 0.5 — Settle the harness *(~1 week)* · autonomy 1 — **proposal**

Runs *during* trajectory accrual, so it costs no calendar time against Phase 1. Full
detail in [`HARNESS_CAPABILITIES.md`](HARNESS_CAPABILITIES.md); this is the subset that
must land before the baseline is pinned, and why.

| # | Change | Why it gates the baseline | Effort |
|---|---|---|---|
| **H1** | Typed I/O — validate args pre-dispatch; stop shipping `returns` on the wire | Moves `tool_errors` from server stack traces to harness refusals | ~1 day |
| **H3** | Loop policy to config; **restore the iteration cap** | `turn_count` is unbounded today, so its distribution is not a baseline | ~½ day |
| **H2** | `context_status`, `note_write`, `note_read` | Shares its on-disk root and path-jail with Loop A/B (§4.0) | ~1 day |
| **H6** | `RunState` object | Becomes what `_record_trajectory` serializes; closes gap 2 | ~3 days |
| **H4** | Tool-result store (pass-by-reference) | Collapses `truncations`/`compactions` — the largest baseline shift of the five | ~3 days |

**H5 (code as action) is explicitly *not* in Phase 0.5.** It changes what a "tool call"
means, so `tool_calls` per task stops being comparable across the boundary entirely. It is
also the largest and riskiest piece, and it can *lose* accuracy on 8B-class models. Land it
after Phase 2 has an honest reading, or accept that it forces a baseline re-pin — and say
so out loud when it does.

**Exit criterion**: `onit learn` reports the same signal set it does today, but the numbers
are stable run-to-run on a fixed task. That stability *is* the baseline precondition.

**Ordering within the phase**: H1 → H3 → H2 → H6 → H4. The first two are bug fixes worth
shipping on their own merits; H6 before H4 so the result store's bookkeeping has a typed
home rather than three more locals in `chat()`.

### Phase 1 — Recall *(~3 days)* · autonomy 2

- Implement `memories` in `_build_messages()`; populate at the three `'memories': None`
  sites — into the `RunState` from H6, not a fourth ad-hoc parameter.
- `src/learn/recall.py` — lexical retrieval over trajectories, token-budgeted.
- SQLite index over `trajectories/*.jsonl`, derived and rebuildable (§4.0).
- Reconcile with H2: `note_read` (this session's own notes) and recall (prior sessions'
  distilled outcomes) are one tool family from the model's point of view. Ship them with
  one naming scheme, or the model will use neither well.
- A/B on the holdout. **Ship only if it wins.**

### Phase 2 — Playbook *(~3 weeks)* · autonomy 2

- `src/learn/playbook.py` — item store, delta ops, decay/eviction, scope isolation.
- `src/learn/reflect.py` — Reflector + Curator passes (offline, batch).
- Playbook block at the tail of the static half in `build_assistant_instruction()`;
  version pinned per session.
- `onit learn` CLI: `--status`, `--reflect`, `--promote`, `--rollback`, `--show`,
  `--forget <id>`.
- A/B on holdout + ≥2 benchmark tasks; write to `RESULTS.md`.

**Exit criterion**: a measurable, reproducible win on held-out tasks with playbook on vs.
off — or an honest negative result and a decision to stop.

### Phase 3 — Skills *(~4 weeks)* · autonomy 3

- `src/learn/skills.py` — mining, synthesis, five-gate validation.
- `src/mcp/servers/tasks/skills/mcp_server.py` — serves pinned skills; config entries.
- Wire GAIA in `benchmarks/tasks/agentic.py` as the tool-creation benchmark.
- `onit skills` CLI: `list`, `test`, `pin`, `unpin`, `rm`.

### Phase 4 — Evolution *(~6 weeks)* · autonomy 4

- Fix the custom-template prefix-split bug **first**.
- `src/learn/evolve/` — GEPA-style prompt optimizer over `benchmarks/` tasks (D1).
- Archive + CMP-based parent selection for code variants (D2), worktree-isolated.
- Every promotion is a PR with the fitness record attached.

---

## 6. Measuring it honestly

Report as a fixed panel, per candidate, always against the pinned baseline:

| Axis | Metric | Guard |
|---|---|---|
| **Quality** | holdout accuracy; `benchmarks/` task scores | frozen set; judge rotated |
| **Efficiency** | turns/task, tool calls/task, peak prompt tokens, compactions | must be paired with quality (I4) |
| **Latency** | wall-clock, prefill share | catches prefix-cache regressions |
| **Cost** | total tokens incl. reflection passes | reflection is not free |
| **Stability** | score on tasks that passed *last* generation | catches forgetting |
| **Attribution** | per-bullet / per-skill helped−hurt | catches dead weight |

Rules: (1) A candidate ships only on a **holdout** win, never a dev-split win. (2) Point
estimates do not promote — require the delta to exceed the benchmark's stderr, which
`RESULTS.md` already reports. (3) Regression on the *previous* generation's passing tasks
blocks promotion even if the average improved. (4) Anything that improves implausibly
fast gets its diff read by a human before it goes anywhere.

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| **Reward hacking** — proxy metrics gamed ([2606.15385](https://arxiv.org/abs/2606.15385)) | Frozen holdout; quality paired with efficiency (I4); judge rotation; human read of fast wins |
| **Context collapse** — rewrites erode knowledge (ACE) | Delta-only updates (I2); itemized store; never regenerate |
| **Playbook bloat** — memory crowds out the task | Hard token cap; decay + eviction; attribution-driven pruning |
| **Prefix-cache thrash** — learned context invalidates the prefix | Playbook at tail of static half; version pinned per session; latency is a tracked metric |
| **Cross-user leakage** — one tenant's data in another's prompt | Owner-scoped by default; redaction; ≥3-owner rule for global promotion |
| **Skill-library dilution** — more tools, worse selection | Benchmark delta ≥ 0 required per skill; deprecation path; description quality gate |
| **Runaway self-modification** | Sandbox (I6); allowlisted files; PR-gated; `pinned.yaml` rollback (I3) |
| **Forgetting** | Stability axis blocks promotion on prior-generation regressions |
| **Baseline drift under the harness** — a learning win that is really a harness change, or a harness regression hidden by a learning win | Phase 0.5 gate before the pin (§1.1); every result row stamps the harness phase alongside `ONIT_LEARN`; any post-pin harness change to the loop, the result store or the tool contract forces an explicit re-pin, announced in `RESULTS.md` |
| **Effort sink with no payoff** | Phase gates. Each phase must win on the holdout or the track stops. Phase 0 has standalone value regardless |

**Explicit non-goals for v1**: no weight updates or finetuning; no unattended modification
of the running process; no autonomous changes to the security/containment layer; no
network-reachable self-modification surface.

---

## 8. Recommendation

*Revised August 09, 2026. The destination is unchanged — Phase 2 is still the main event.
What changed is what happens in the weeks before it can honestly start.*

Phase 0 is done. **Do Phase 0.5 next (the harness), then freeze the holdout, then Phase 2
with Phase 1 as a cheap ride-along.**

The original recommendation — Phase 2 next — was right about the destination and silent
about a dependency. Phase 2 cannot start until there are enough trajectories to freeze a
holdout from, and there are 10. That is a wait of weeks measured in real usage, not in
engineering days, and nothing in this document was scheduled to fill it. Phase 0.5 fills
it, needs no data, and removes five sources of baseline drift on the way through (§1.1).

The sequencing is therefore not a compromise between two roadmaps. It is one roadmap whose
critical path was always *data accrual*, with the harness work slotted into the wait:

```
  now ──────────────── trajectories accrue (weeks) ─────────────────▶ ≈100 tasks
        │                                                                  │
        └── Phase 0.5: H1 → H3 → H2 → H6 → H4 (~1 week of work) ──┐        │
                                                                  ▼        ▼
                                                        both gates met → freeze holdout
                                                                           │
                                                                           ▼
                                                              Phase 1 + Phase 2
```

Phase 2 still carries low risk and still captures the majority of the literature's
demonstrated scaffold gains — ACE reports double-digit improvements on exactly this
mechanism, without labels and without gradients, on small open models. Nothing in the
harness analysis contradicts that, and the harness track has no comparable evidence behind
it: NOOA's capability list is a vendor framework post, not a controlled result. Where the
two conflict on priority, **this document wins on evidence and that one wins on
availability** — which is precisely why the answer is to run the available work during the
wait rather than to reorder the destination.

One thing to hold onto if the schedule slips: if only part of Phase 0.5 lands, **H3 and H1
are the ones that must** (they are bug fixes, and H3's unbounded `turn_count` corrupts a
Layer 0 metric today). H4 is the largest baseline shift and the easiest to defer, at the
cost of a re-pin later.

Phase 3 (skills) is the highest-*ceiling* work and the strongest product story — Alita's
generated MCPs transfer across agents and lift smaller models, which is directly on
OnIt's thesis of running well on self-hosted models. But it depends on Phase 0's data to
know what to build.

Phase 4 is a research track, not a product track. Start it only after Phases 0–2 have
produced a trustworthy holdout and an honest reading of `RESULTS.md` — and when it does
start, select by CMP, keep the archive open, and assume the agent will try to cheat.

---

## Sources

Surveys — [Self-Improvements in Modern Agentic Systems (2607.13104)](https://arxiv.org/abs/2607.13104) ·
[Self-Evolving Agents: What/When/How/Where (2507.21046)](https://arxiv.org/abs/2507.21046) ·
[Comprehensive Survey of Self-Evolving AI Agents (2508.07407)](https://arxiv.org/abs/2508.07407) ·
[Awesome-Self-Improving-Agents](https://github.com/FrontisAI/Awesome-Self-Improving-Agents)

Memory & experience — [ACE (2510.04618)](https://arxiv.org/abs/2510.04618) ·
[Reflexion (2303.11366)](https://arxiv.org/abs/2303.11366) ·
[ExpeL (2308.10144)](https://arxiv.org/abs/2308.10144) ·
[Agent Workflow Memory (2409.07429)](https://arxiv.org/abs/2409.07429) ·
[Dynamic Cheatsheet (2504.07952)](https://arxiv.org/abs/2504.07952) ·
[Memp (2508.06433)](https://arxiv.org/abs/2508.06433) ·
[MemGPT (2310.08560)](https://arxiv.org/abs/2310.08560)

Tools & skills — [Alita (2505.20286)](https://arxiv.org/abs/2505.20286) ·
[Voyager (2305.16291)](https://arxiv.org/abs/2305.16291) ·
[AutoSkill (2603.01145)](https://arxiv.org/abs/2603.01145) ·
[SkillX (2604.04804)](https://arxiv.org/abs/2604.04804) ·
[Raw Experience to Skill Consumption (2605.23899)](https://arxiv.org/abs/2605.23899)

Prompt & scaffold — [GEPA (2507.19457)](https://arxiv.org/abs/2507.19457) ·
[DSPy (2310.03714)](https://arxiv.org/abs/2310.03714) ·
[TextGrad (2406.07496)](https://arxiv.org/abs/2406.07496) ·
[ADAS (2408.08435)](https://arxiv.org/abs/2408.08435)

Self-modifying agents — [Darwin Gödel Machine (2505.22954)](https://arxiv.org/abs/2505.22954) ·
[Huxley-Gödel Machine (2510.21614)](https://arxiv.org/abs/2510.21614) ·
[SICA (2504.15228)](https://arxiv.org/abs/2504.15228) ·
[Red Queen Gödel Machine (2606.26294)](https://arxiv.org/abs/2606.26294)

Parametric — [SEAL (2506.10943)](https://arxiv.org/abs/2506.10943) ·
[Self-Rewarding LMs (2401.10020)](https://arxiv.org/abs/2401.10020) ·
[Absolute Zero (2505.03335)](https://arxiv.org/abs/2505.03335)

Safety & evaluation — [Reward Hacking in LM Agents (2606.15385)](https://arxiv.org/abs/2606.15385) ·
[Safety in Self-Evolving LLM Agent Systems (2606.23075)](https://arxiv.org/abs/2606.23075) ·
[LifelongAgentBench (2505.11942)](https://arxiv.org/abs/2505.11942) ·
[SIP-Bench (2601.20882)](https://arxiv.org/abs/2601.20882) ·
[TheAgentCompany (2412.14161)](https://arxiv.org/abs/2412.14161)
