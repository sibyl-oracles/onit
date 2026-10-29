# Self-Improving OnIt — Literature Review and Scaffold Proposal

**Status**: Deliverables 1–6 of §5.0 are **built** — the ruler, Loop A, the implicit
signals, Loop B, the lifecycle actuator and Loop C.  Loop D (evolution) is not
started, by design.  None of it has been shown to *win*: the code exists, the
tests pass, the paths are reachable — and the A/B that would make the claim
stronger has not been run.  See §5.0 and §5.1.
**Date**: October 2026 · *Revised October 22, 2026 — full codebase review against `main`
(commit `0f50822`) and against the live corpus. Every count in this document was
re-measured; the previous revision's figures (209 records, 23 sessions, "~29 usable") are
two months stale and are corrected in place. The data gate that had been holding Phase 1
is now open — the blocker is the missing ruler, not the missing data. See §3.0.*
*Amended October 23, 2026 — the six deliverables are built and the test suite that
holds them is in place; §5.1 records what is measured and what is not.*
*Revised October 23, 2026 (second pass) — the four blockers that made the A/B
unrunnable are now fixed: the holdout no longer leaks into recall or reflection,
the benchmark no longer pollutes the corpus, the model-based Reflector exists,
and the eval runner exists. §5.2 records what was broken, what it now does, and
the three pre-existing bugs found on the way. The A/B is still unrun — it now
can be run.*
*Previous revisions: August 20, 2026 (Phase 0.5 closed out) · August 09, 2026.*
**Companions**: [`HARNESS_CAPABILITIES.md`](HARNESS_CAPABILITIES.md) — the harness track,
now complete. [`SELF_IMPROVEMENT_GAPS.md`](SELF_IMPROVEMENT_GAPS.md) — the standing gap
assessment against `main`; §3 and §8 here are kept consistent with it, and where the two
differ this document carries the reason why.

---

## 1. Executive summary

**OnIt records, and now it reads back.** The substrate is done and healthy: every
task's full trajectory — every turn, every tool call, every failure, every token
count — is written to disk and survives the session.  As of this revision the
reading-back half is built too: recall, the playbook, the skill library, and the
ruler that is supposed to say whether any of it helps.  What is missing is the
measurement itself.

The precise statement of the gap, as of this revision:

> 1,010 task records and 221 loop events are on disk across 58 session files
> (Aug 4 – Oct 21, 2026).  As of this revision the first byte of it has been read back into a prompt:
> Loops A (recall), B (the playbook) and C (skills) are built, wired and reachable at
> autonomy 2 and above.
>
> What has **not** happened is the part that would make any of it a claim rather
> than a capability: **the A/B on the holdout has not been run.**  The ruler exists;
> the loops exist; the measurement does not.  "It is built" and "it wins" are
> different claims, and only the first one is currently true.

That is the whole problem, and it is a good one: the expensive half of
self-improvement (instrumenting a system honestly) is done, the loops that consume the
instrumentation are done, and what is left is to run the measurement and be honest
about the answer.  The remaining work is additive on top of a substrate that does not
need redesigning.

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
goes live.** OnIt has most of that fitness function already — `benchmarks/` on Inspect-AI —
and a held-out suite is the difference between self-improvement and self-delusion.
*Amended August 20: what it has is aimed at the model, not the scaffold. Every registered
task is single-turn and near-saturated, and the two agentic tasks that would actually move
under these loops are written but unregistered. See §3, gap 5.*
*Amended October 22: GAIA and SimpleQA are now registered and run (SimpleQA scores 0.940,
stderr 0.024). The remaining gap is not registration — it is that **there is no holdout at
all**: no `eval/holdout.jsonl`, no `pinned.yaml`, and the pinned baseline holds two rows
for one model. See §3.0 and §4.0.*

The third thesis, added in the August 09 revision: **the harness must stop moving before
the baseline is frozen.** See §1.1 — this reordered the plan, and the harness has now
stopped moving.

The fourth thesis, added August 20: **the cheapest outcome signal in the system is one
already being computed and thrown away.** [`verify.py`](../src/model/serving/verify.py)
fact-checks every finished answer by default, and its verdict is already sitting in each
trajectory's `metrics` blob. `derive_signals()` now reads it. Meanwhile the human rating
path has been wired for weeks and used **zero** times across 1,010 tasks. Unlabeled
experience is what blocks Loops A and B; the label is already on disk. See §3, gap 3.
*Amended October 22: the verifier now labels 820 clean / 70 issues / 120 unchecked out of
1,010 — 88% weakly labeled. The label is no longer the blocker. The *rating* path is
still at zero, and the implicit-signal replacement has not been built.*

### 1.0 What the October 2026 review changed about the plan

Five findings from the codebase review reshaped the schedule in §5. Each is a claim
about the current tree, and each is verifiable.

| # | Finding | Evidence | Status | Effect on the plan |
|---|---|---|---|---|
| 1 | **The data gate is open.** 1,010 records / 58 files, 820 weakly labeled. | `~/.onit/learned/trajectories/` | still true | The "wait for data" framing in earlier revisions is dead. Phase 1 is no longer blocked. |
| 2 | **There is no ruler.** No `eval/`, no `holdout.jsonl`, no `pinned.yaml`; `benchmarks/baselines/full.json` has 2 rows. | `benchmarks/`, `~/.onit/learned/` | **closed** | `src/learn/eval.py` draws the holdout and the dev split and pins the baseline, and `assert_outside_agent_surface()` keeps the ruler out of the agent's reach. |
| 3 | **The injection path does not exist.** The `memories` kwarg was threaded then deleted; `_build_messages()` has no learned-context input. | `src/model/serving/chat.py` | **closed** | `src/learn/recall.py` and `src/learn/playbook.py` both inject now. |
| 4 | **`src/learn/lifecycle.py` is dead code.** 178 lines of policy, zero production callers, one test importer. | `src/learn/lifecycle.py` | **closed** | `src/learn/actuator.py` is the missing half, and the policy had a real bug in it that no caller had ever hit. |
| 5 | **Owner isolation is inert.** All 1,010 records carry `owner: null`; `iter_records()` does not filter by owner. | `src/learn/trajectory.py` | **closed** | `iter_records()` filters by owner and model at the point the record is produced, and every retrieval path passes an explicit filter. |

Finding 5 was the one that gated the others. A retrieval path that returns one user's
sessions into another's prompt is not a feature, it is a breach, and it had to be closed
before the first episode was injected rather than after.  It is closed: `iter_records()`
takes an `owner` and a `model` and filters at the point the record is produced, so a
loop that filtered afterwards would still have read another tenant's data.

The review also found three things that are *not* gaps but are worth recording as
constraints the implementation must respect:

- **Tool output is not marked untrusted in the message stream.** The standing rule
  exists in the static half (`prompts.py`, gated on search tools), but tool results land
  as bare `{'role': 'tool', 'content': result}`. For a self-improving agent this is worse
  than for a static one: the agent will be *writing* what it learned from that output
  into a durable artifact that later runs read.
- **Learned artifacts are prompt-injection surfaces by construction.** A playbook bullet
  distilled from a web page is a stored instruction. It needs the same untrusted-data
  treatment, and the recall path must be **write-isolated** from the read path: the agent
  must not be able to write into the store it reads from.
- **`redact_tool_args` defaults to on**, so tool arguments are stored as digests. The
  recurring bash fragments that Loop C would mine as skills are **not recoverable from the
  current corpus** — *unless* the digest is enough to identify the repetition.  It is:
  `args_digest` is the SHA-256 of the whole argument mapping, so two calls with the same
  digest made the same call with the same arguments.  The miner groups by
  `(tool, args_digest)` and counts distinct sessions, and never needs the values.  The
  redaction stays exactly as strict; the repetition becomes visible anyway.  On the real
  corpus that is 36 candidates with 36 unique names.


### 1.1 Relationship to the harness track

> **Resolved, August 20, 2026.** Phase 0.5 is complete (§5), so this section's scheduling
> argument has expired. It is kept because its *second* point — that harness changes
> redefine the very signals Layer 0 records — proved to be the important one, and to have
> been understated: the drift it predicted has already landed inside the recorded corpus
> (§3, gap 6). The rule outlives the schedule that motivated it.

A separate analysis ([`HARNESS_CAPABILITIES.md`](HARNESS_CAPABILITIES.md)) scored OnIt
against NVIDIA's NOOA capability list and proposed six harness changes: typed I/O,
model-callable context APIs, loop-policy config, a tool-result store, code-as-action, and
explicit run state. Two independent reviews of that article converged on the same gaps,
and both asked the obvious question — *does the harness track compete with this one for
the next slot?*

**It does not, and the reason is a scheduling constraint neither analysis noticed.**

**1. This track is currently blocked; that one is not.** §8 says do Phase 2 next, *"but not
until there are enough trajectories to freeze a holdout from."* The holdout wants ≈100
tasks drawn from real sessions. `onit learn` reported 10 across 1 session *at the time of
writing; it reports 1,010 across 58 as of October 22, of which 820 carry a verifier
label — see §3 gap 6*. *This scheduling argument expired in October 2026: the data
arrived, and Phase 1 was unblocked. The blocker is now the missing ruler (§3.0), not the
missing data.* The harness work needs no data at
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
  log-and-drop it"* — H6 generalizes exactly that, and closes gap 2's missing `memories`
  path by giving it somewhere typed to live. *(The dead kwarg was removed in the
  September 2026 simplification pass; Loop A re-adds it as a live, wired path.)*
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

### 3.0 Current-state audit — October 22, 2026

Re-measured against `main` (`0f50822`) and the live corpus. This subsection is the
authoritative count; the gap list that follows it in §3 is unchanged in substance but
carries two-month-old figures, which are corrected in place and marked.

**The corpus.**

| Measure | Count | Note |
|---|---|---|
| Session files | **58** | `~/.onit/learned/trajectories/` |
| Task records | **1,010** | Aug 4 → Oct 21, 2026 |
| Loop events | **221** | `tool.registry_loaded` and friends |
| Verifier labels | **820 clean / 70 issues / 120 unchecked** | 88% weakly labeled |
| Human ratings | **0** | the 👍/👎 path has never been pressed |
| Records with `owner` set | **0** | every record is `owner: null` |
| Distinct models | **12** | Qwen3.8-27B 462 · glm-5.3-flash 333 · deepseek-v4-pro 142 · … |

The corpus is **multi-model**. Any learned artifact is therefore *model-conditional*:
retrieving an episode written by a different model retrieves a strategy that model may
not be able to execute. The "pin per model, not globally" rule in §4.3 is load-bearing,
not pedantry.

**What the numbers change.** The previous revision said the corpus held "209 records, ~29
usable" and that Phase 1 and Phase 2 "cannot honestly start for weeks of accrual." That
is expired. The data is there; the label is there. The blocker is the ruler.

**The five gaps that matter, in the order they bite.**

| # | Gap | Where it lives | Closed? |
|---|---|---|---|
| 1 | **No ruler.** No `eval/`, no `holdout.jsonl`, no `pinned.yaml`; `benchmarks/baselines/full.json` pins 2 rows for 1 model. | `benchmarks/` | **Closed.** `src/learn/eval.py` builds the holdout, the dev split and the pinned baseline, and `assert_outside_agent_surface()` enforces I7 in code. |
| 2 | **No injection path.** `memories` was threaded `onit.py` → `chat()` → `_build_messages()` and never used; the Sept 22 simplification pass deleted the dead kwarg entirely. | `src/model/serving/chat.py` | **Closed.** `src/learn/recall.py` and `src/learn/playbook.py` inject at autonomy 2, and the recall rides in the volatile half while the playbook rides at the tail of the static half. |
| 3 | **No outcome signal beyond the fact-checker.** 0 ratings; no implicit signals; `onit learn` has no text-UI equivalent. | `src/learn/`, `src/ui/commands.py` | **Closed.** `src/learn/signals.py` derives rephrase/acceptance signals, `src/ui/rating.py` adds a `\rate` command, and the derived labels land beside the trajectories rather than overwriting them. |
| 4 | **Dead policy module.** `should_propose_creation()` / `should_update()` / `should_archive()` / `pending_proposals()` have **zero production callers**; the only importer is `src/test/test_learn_events.py`, and the module is not exported from `learn/__init__.py`. | `src/learn/lifecycle.py` | **Closed.** `src/learn/actuator.py` is the other half: it reports the policy, withholds what the policy says to withhold, and restores it. Withholding is reversible by construction. |
| 5 | **No self-modification surface, and the autonomy ladder is decorative.** `_MAX_IMPLEMENTED = OBSERVE`; `adapt` / `extend` / `evolve` are exported, documented in `default.yaml`, and **silently degrade to `observe`**. | `src/learn/config.py` | **Closed.** The cap is now `ADAPT` — the highest rung implemented — and the cap is loud: `requested_level()` and `capped()` say what was asked for and what was granted. Loop C's skill server is the self-modification surface. |

Two of these are not gaps but **constraints the implementation must respect** — they are
recorded in §1.0 and repeated where they bind: tool output carries no untrusted marking
in the message stream, and `redact_tool_args` defaults to on.

**The redaction constraint, and how Loop C gets around it without weakening it.**
The values of a tool call's arguments are redacted before they reach the
trajectory store, because the values are where file paths, search queries and
the occasional pasted credential live.  That made the recurring bash fragments
*unrecoverable* from the current corpus — or so it seemed.

What makes them recoverable is the `args_digest` that already sits beside every
call: the SHA-256 of the full argument mapping, order-independent.  Two calls
with the same digest made the same call with the same arguments.  So the miner
groups by `(tool, args_digest)` and counts the distinct sessions each group
spans, and it never needs the values — and never writes them down.  The
redaction stays exactly as strict; the repetition becomes visible anyway.

On the real corpus that is **36 candidates across 36 unique names**, mined
from 1,010 records without loosening a single redaction rule.

**The one that gates everything else, and is now closed.** Owner isolation used to be
inert: all 1,010 records carry `owner: null`, because `get_session_owner()` reads the
session index, which is empty for CLI-recorded runs, and `iter_records()` did not filter
by owner at all. `owner_hash()` existed; nothing enforced it. A recall path built that
way would have returned **global** results — in a multi-user web deployment, one tenant's sessions
in another's prompt.

It is closed now, and closed in the right place.  `iter_records()` takes an `owner`
and a `model` and filters **at the point the record is produced**, not downstream:
a loop that reads every record and filters afterwards has still read another
tenant's data.  The default is unfiltered, and that is deliberate — `onit learn`
reports on the whole store, and a status report that hid another tenant's tasks would
be lying about what is on disk.  Every *retrieval* path passes an explicit filter.
The recall path's default is owner-less-only, with `ALL_OWNERS` as an explicit
sentinel for the paths that genuinely mean "every owner"; a sentinel that has to be
written down is a sentinel that will be written down deliberately.

An honest audit. Findings are file-referenced so they can be verified.

**What already exists and is directly reusable:**

| Asset | Location | Why it matters |
|---|---|---|
| Rich per-run telemetry | [`TurnMetrics`](../src/model/serving/chat.py#L95) — turns, tool calls, prefill/decode split, compactions, peak prompt tokens | Free efficiency features for the reward model — already computed |
| Benchmark harness | [`benchmarks/run.py`](../benchmarks/run.py#L29) — Inspect-AI: gsm8k, humaneval, mbpp, bigcodebench + METR time-horizon layer | **The fitness function** — the single biggest head start, but aimed at the model rather than the scaffold. See gap 5 |
| Answer verifier | [`verify.py`](../src/model/serving/verify.py) — fact-checks the finished answer against the evidence that produced it; on by default | **An outcome signal that already runs**, already recorded per task and not yet read. See gap 3 |
| Explicit run state | [`RunState`](../src/model/serving/state.py#L86), [`ResultStore`](../src/model/serving/results.py), [harness tools](../src/model/serving/harness.py) | Phase 0.5 (H2/H4/H6). `stop_reason` is the signal metrics cannot supply |
| LLM-as-judge | [`benchmarks/scorers/onit_judge.py`](../benchmarks/scorers/onit_judge.py) | Quality gate for open-ended tasks |
| Results ledger | [`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md) | Where promotion evidence gets written |
| MCP tool discovery | [`discover_tools()`](../src/lib/tools.py#L213), [`ToolRegistry`](../src/type/tools.py#L347) | Skills can ship as an MCP server — no new plumbing concept |
| Containment | [`command_policy.py`](../src/mcp/servers/tasks/os/bash/command_policy.py), `installs_sealed()`, [`container_launcher.py`](../src/container_launcher.py) | Sandbox for validating self-generated code already exists |
| Prompt templating | [`build_assistant_instruction()`](../src/mcp/prompts/prompts.py#L43) with YAML `template_path` override | Evolved prompts have a place to land |
| Session index / ownership | [`sessions.py`](../src/sessions.py), `get_session_owner()` | Per-user memory scoping is already possible |

**The gaps**, re-verified against `main` on August 20, 2026. Gap 1 is closed; 2 and 3 moved;
5 grew; 6 and 7 are new and were not visible until there was a corpus to look at:

1. ~~**No trajectory persistence.**~~ *Closed.* The session file still holds
   `{task, response, timestamp}`; [`src/learn/`](../src/learn/) now writes the run
   alongside it.
2. **`memories` is a dead parameter.** *Still dead, deliberately — and now the one thing
   standing between a recording agent and a learning one.* It is threaded from `onit.py`
   (`'memories': None`, at [three](../src/onit.py#L1486) [call](../src/onit.py#L1806)
   [sites](../src/onit.py#L2355)) through [`chat.py`](../src/model/serving/chat.py#L2759)
   into [`_build_messages()`](../src/model/serving/chat.py#L1360), whose body never
   references it. Phase 0 records but injects nothing, so the hook stays unwired until
   Phase 1 — which is what makes level `observe` byte-for-byte identical to learning off.
   Every other gap below is a question of *what* to inject. This one is the aperture, and
   nothing is a learning loop until it opens.
3. **No outcome signal.** *Closer to closed than this document has been claiming.* Three
   sources exist and the plan had counted one and a half:
   - **Free process signals** — tool errors, retries, truncations, compactions — derived
     per run by `derive_signals()`. Present on all 1,010 records. These say a run went
     *badly*, never that an answer was *wrong*.
   - **Human verdict** — `POST /api/rating` (👍/👎) appends a `rating` line, wired end to
     end through the web UI. **Used zero times across 1,010 tasks.** A signal conditional on
     someone bothering to click is not a signal, and no amount of further UI work changes
     that. Implicit signals are the replacement, and are now a Phase 1 deliverable.
   - **A verifier that already runs.** [`verify.py`](../src/model/serving/verify.py)
     fact-checks each finished answer against the evidence that produced it, on by
     default, and `TurnMetrics` already records `verify_s`, `verify_issues` and
     `verify_revisions`. Those flow straight into each record's `metrics` blob.
     **This one is now closed** — `verifier_signal()` reads the field, and the corpus
     carries **820 clean / 70 issues / 120 unchecked** labels (§3.0). The gap that
     remains is not the label but the *strength* of the signal: a weak binary label from
     a model judge is enough to rank episodes, not enough to train a playbook on. That
     is what implicit signals and the text-UI rating path (§5, deliverable 3) are for.
     *Verified October 22, 2026 — this bullet was open in the August 20 revision.*
4. **Tool set is frozen at startup.** `ToolRegistry.register()`
   ([`type/tools.py`](../src/type/tools.py#L595)) is called from exactly one site —
   [`lib/tools.py`](../src/lib/tools.py#L266), during discovery. No runtime registration,
   no unregistration, no reload. Loop C has no insertion point to build against.
5. **No holdout discipline — and the fitness function measures the wrong thing.** This gap
   grew since August 09, because settling the harness fixed the ruler's *stability* and
   not what it is a ruler *of*:
   - `benchmarks/baselines/` holds `full.json` (pinned September 05, 2026) and
     nothing else. No `holdout.jsonl`, no
     `pinned.yaml`. Invariants I1 and I3 are prose, not code.
   - The tasks registered in [`run.py`](../benchmarks/run.py#L30-L41) are gsm8k, humaneval,
     mbpp and bigcodebench — **single-turn, and near-saturated** at 0.978 /
     0.915 / 0.957 in `RESULTS.md`. A playbook cannot move humaneval: there is no
     multi-step tool use in it for a learned workflow to improve, and almost no headroom
     left if there were. Freezing a holdout from these would formalize the error rather
     than fix it — a ruler that cannot detect what Loops A–D change.
   - The two benchmarks that *would* be sensitive are **registered and run** (September
     04–06, 2026): GAIA ([`tasks/agentic.py`](../benchmarks/tasks/agentic.py)) and
     SimpleQA ([`tasks/factuality.py`](../benchmarks/tasks/factuality.py)). Sampled-tier
     rows: simpleqa 0.940 / stderr 0.024, humaneval 0.970 / 0.017, mbpp 0.990 / 0.010 —
     all on `glm-5.3-flash:cloud`, 14 tools live, learn off. GAIA loads from the local
     dataset cache with no token, but runs text-only (its wrapper ignores `file_path`),
     so it is not yet holdout-ready.
   - `RESULTS.md` still mixes models across hosts with no pinned row per model, and the
     sampled-tier rows are reference points rather than pin-gated regression numbers.
6. **The recorded corpus is not comparable to itself.** `onit learn` reports 1,010 tasks
   across 58 sessions, which reads as Gate 1 met. It is not. `stop_reason` is empty on
   **179 of them**, because `RunState` only began recording it on August 18. H1 and H3
   (August 09) redefined what `tool_errors` counts — and `_build_parameters` had been
   dropping `required`, so `blank_required_args` never fired before then. H4's result store
   collapses `truncations` and `compactions`. Roughly **29 records** were produced under
   the settled harness. §7's "baseline drift" risk is not hypothetical: it has already
   happened, inside the store, before anything was pinned. The store needs an era or
   harness-version stamp per record so this is legible later instead of inferred from
   timestamps, as it had to be here.
   *Updated October 22, 2026 — the corpus is 1,010 records, not 209; the 29-usable
   figure describes the **pre-Phase-0.5 era** and does not describe the corpus as it
   now stands. The comparability problem is real but is now bounded: the unusable
   records are identifiable by timestamp, which is what the era stamp would make
   explicit. The holdout must be drawn from the settled era only.*
7. **The corpus mixes twelve models with no stratification.** Qwen3.8-27B (462),
   glm-5.3-flash (333), deepseek-v4-pro (142), deepseek-v4.1-flash (46), plus eight
   others. A lesson distilled from a frontier
   cloud model's trajectories may be actively wrong for an 8B local model — which is the
   deployment this plan exists to serve, and the case where the literature's gains are
   largest. §4.3's scope rules cover *owner* isolation and say nothing about model
   isolation. They need to, and baselines must be pinned per model rather than globally.

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

Two trees, one manifest each:

```
~/.onit/learned/            # `learn.path`, or ONIT_LEARN_PATH — the AGENT'S tree
  trajectories/*.jsonl      # Layer 0 — raw experience
  playbook/
    global.json             # Loop B — itemized bullets w/ counters + provenance
    user/<owner-hash>.json  #          per-owner scope (web multi-user)
  skills/<name>/            # Loop C — code + manifest.yaml + tests
  scaffold/
    archive/<gen>-<id>/     # Loop D — prompt/code variants + fitness records
  index/
    episodes.sqlite         # derived retrieval index (rebuildable, disposable)
```

```
benchmarks/                 # the REPO'S tree — the ruler, outside the agent's reach
  eval/
    holdout.jsonl           # frozen; the improvement loops MUST NOT read this
    dev.jsonl               # iteration split; loops may read this
    baseline.json           # pinned scores for A/B
  baselines/
    full.json               # the existing model-level pin (2 rows, 1 model)
```

**The split is the point.** Everything in `~/.onit/learned/` is the agent's own
writable surface — it is where trajectories land, where the playbook is promoted, where
skills are installed. Everything in `benchmarks/eval/` is the measuring instrument, and
it lives outside that surface by layout, not by convention (I7). A promotion gate that
reads a holdout the agent can edit is not a gate.

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
- **I7 — The ruler is outside the agent's writable surface.** *(Added October 22, 2026.)*
  The holdout, the pinned baseline, the judge configuration and the promotion gate live
  **outside** `data_path` and **outside** the learned store. No loop writes them, no
  approval scope reaches them, and no `learn:` config key points at them. Without this,
  I1 is a convention: a Loop C or D that can edit its own exam grades itself.

**On I1 and I7 together.** I1 says no loop *reads* the holdout. I7 says no loop can
*write* it. They are different failures and each is fatal alone: I1 without I7 is a
convention against a cooperative agent, and I7 without I1 is a sealed box nobody opens.
Together they are the difference between a self-improving system and a self-certifying
one. Both must land **in the same commit** as the first promotion gate, because a
promotion decided against a holdout the agent can edit is not a measurement.

**On where the ruler physically lives.** Two candidate placements, and the tradeoff:

| Placement | Pro | Con |
|---|---|---|
| `~/.onit/learned/eval/` | Next to the store it scores; one root to back up. | The agent's own writable tree. A skill with write access to `learn.path` could edit its own exam. |
| **Repo-side, outside `learn.path`** *(chosen)* | Not under the agent's writable surface by construction. A deployment that gives the agent write access to `data_path` still cannot touch it. | Two roots to back up; a benchmark run needs both on disk. |

The chosen placement is the repository's `benchmarks/` tree, with the holdout and the
pinned baseline committed alongside the code they measure. A deployment that wants them
elsewhere can point `eval.path` at a read-only mount — but the *default* must be
somewhere the agent cannot reach, and "somewhere the agent cannot reach" has to be true
by layout rather than by policy.

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

| Signal | Source | Cost | Status (Oct 22) |
|---|---|---|---|
| Tool error / retry / truncation counts | already in the loop | free | ✅ recorded, all 1,010 |
| Efficiency (turns, tool calls, compactions, peak tokens) | `TurnMetrics` | free | ✅ recorded, all 1,010 |
| **Verifier verdict** | [`verify.py`](../src/model/serving/verify.py) — on by default; `verify_issues`, `verify_revisions` | **free, already computed** | ✅ **read into `signals.verifier`; 820 clean / 70 issues / 120 unchecked** |
| Execution feedback (exit codes, test pass/fail) | sandbox `bash` results | free, strongest where available | not extracted |
| Implicit user signal | next-turn rephrase ≈ failure; acceptance/thanks ≈ success | small heuristic | not built — Phase 1 deliverable 3 (§5) |
| Explicit rating | 👍/👎 endpoint in [`src/ui/api.py`](../src/ui/api.py) | ~half a day, highest value per bit | ✅ shipped · **used 0 times in 1,010 tasks** · no text-UI equivalent |
| LLM judge | `onit_judge` on a sample | tokens | not wired to trajectories |

The August 20 re-verification reordered this table. "Highest value per bit" was true of the
explicit rating and irrelevant, because the bits never arrive: a signal gated on a human
choosing to click produced nothing across 1,010 tasks. The verifier row is the correction —
it is free, it is *already running*, and its verdict is already in the store. Reading it is
item 1 of §8.

### 4.2 Loop A — episodic recall *(low risk, ~3 days)*

Retrieve the *k* most similar prior trajectories for the current task and inject their
distilled outcome ("what worked / what failed"), Reflexion- and Dynamic-Cheatsheet-style.

- **Build the path** — there is no hook to wire. The `memories` kwarg was threaded
  `onit.py` → `chat()` → `_build_messages()` and never referenced; the September 22
  simplification pass **deleted the dead kwarg entirely**. `_build_messages()` today
  takes no learned-context input. Loop A builds the path rather than re-enabling a
  stub, and the value rides in the `RunState` (H6) rather than as a fourth ad-hoc
  parameter.
- **Placement**: a compact block in the **volatile half**, alongside the task — episodes
  are per-task and would thrash the cache if placed in the prefix. The seam is
  `split_instruction()` ([`lib/text.py`](../src/lib/text.py)); the prefix contract is
  documented at [`chat.py`](../src/model/serving/chat.py) and is worth ~2,800
  tokens/request on every turn of every task.
- **Retrieval**: start with BM25/TF-IDF over the task text. OnIt already ships local
  search infrastructure ([`local/search/toolkit.py`](../src/mcp/servers/tasks/local/search/toolkit.py));
  reuse it before reaching for embeddings. Per user's constraint, no new pip deps —
  hand-roll or reuse what's vendored. `toolkit.py` already has BM25, an optional dense
  path and RRF; the corpus is small enough that BM25 alone is the right first cut.
- **Budget**: hard cap (e.g. 800 tokens, k ≤ 3). Memory that crowds out the task is a
  regression.
- **Kill switch**: `ONIT_LEARN=0` / `learn: false` in `default.yaml` bypasses everything.
- **Owner scoping is not optional** *(added October 22, 2026)*. `iter_records()` does
  not filter by owner today, and every one of the 1,010 records is `owner: null`
  (§3.0). Recall must filter by owner *and* by model before it returns anything:
  the corpus is multi-model (§3, gap 7), and an episode written by a different model
  is a strategy that model may not be able to execute.
- **Untrusted marking** *(added October 22, 2026)*. A recalled episode is distilled
  from a run that read tool output, which may itself contain a prompt injection. The
  injected block must be marked the way the standing rule marks tool output, and the
  recall path must be **write-isolated** from the read path: the agent must not be able
  to write into the store it recalls from.

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
| **3 — extend** | Loop C live: a skill pinned by the curator is served as a tool. Nothing is served below this rung |
| **4 — evolve** | Loop D; **always** human-reviewed PR, never auto-merge |

**What happens when a level is asked for and does not exist.**
`_MAX_IMPLEMENTED = EXTEND` in [`src/learn/config.py`](../src/learn/config.py), and
`autonomy()` returns `min(level, _MAX_IMPLEMENTED)`. A config asking for `evolve` gets
`extend` — the highest rung that is implemented. The cap is loud: `requested_level()`
and `capped()` say what was asked for and what was granted, and `learn report` prints
the difference.

*October 22, 2026 — the cap was `OBSERVE`, and it was silent.* The first fix (chosen)
was to make it fail loudly and raise the cap as each rung landed, rather than to rename
the levels; the names keep their meaning and `requested_level()` preserves the request.

*October 28, 2026 — the cap moved from `ADAPT` to `EXTEND`.* The rung was documented
before it worked: `default.yaml` said "the autonomy level has to be `extend` before a
skill may be served at all" and no code said it, so the documented gate could never open
*and* the skills server served at every level, including `off`. Both are fixed: the
server checks `EXTEND` in `load_skills()`, and the cap admits `EXTEND` so the check can
pass. The level travels to the spawned server as `ONIT_LEARN_AUTONOMY`, because a stdio
server has no config of its own.

The cap stays: the agent must not be able to talk itself into a level it has not earned.

---

## 5. Implementation plan

Phases are sequential; each ends in something shippable and measurable.

### 5.0 Build order *(added October 22, 2026)*

The phases below are the *shape* of the work. This is the *order* to do it in, with the
reason each item earns its place. The order changed in October 2026 because the blocker
changed: the data arrived, so the ruler became the first thing to build rather than the
last.

| # | Deliverable | Size | Status | Why here |
|---|---|---|---|---|
| 1 | **Holdout + baseline + the I1/I7 ownership invariant** | days | **built, and readable** | Without it, nothing after it is measurable. It also converts "the agent feels better" into a number the agent doesn't own. *The split file was written and never read back — three pre-existing bugs, now fixed (§5.2).* |
| 2 | **Recall (Loop A)** | ~3 days | **built** | Smallest change that closes the loop end-to-end. Reuses `toolkit.py` BM25. A/B on the holdout; **ship only if it wins** — the null result is the common one. *The holdout no longer leaks into it (§5.2).* |
| 3 | **Implicit signals + text-UI rating** | ~2 days | **built** | Turns 888 weakly-labeled records into a training set. Retroactive on existing data. |
| 4 | **Playbook (Loop B)** | ~3 weeks | **built** | The main event. Reflector + Curator, delta-only updates, token-capped, version-pinned per session. *The mechanical Reflector could only emit templates; the model-based one now exists (§5.2).* |
| 5 | **Lifecycle actuator, or delete `lifecycle.py`** | days | **built** | Either make the dead policy live or stop it reading as done. |
| 6 | **Skills (Loop C)** | ~4 weeks | **built** | Never core rewrites — coded extensions only. |
| 7 | **Evolve (Loop D)** | ~6 weeks | **not started** | Only after A and B show reproducible wins. |
| 8 | **The eval runner — the instrument** | ~1 week | **built** | *Added October 23.* Without it the ruler is a file nobody replays. Now: `onit learn --eval-run [--ab]`. |
| 9 | **Run the A/B** | hours | **not run** | The one thing left. It needs a model endpoint and a drawn holdout, and both are the operator's to provide. |

**What "built" means, and what it does not.**  It means the code exists, the tests
pass, and the path is reachable from a config file.  It does **not** mean any
of it has been shown to *win* on the holdout.  Deliverable 2's exit criterion is
a measured A/B, and that A/B has not been run — so the honest statement is that
the loop is *built and wired*, and the loop is *unproven*.  Those are different
claims, and the difference is the whole subject of this document.

**What "the tests pass" is worth, exactly.**  §5.1 records this in full: the
suite is 3,024 tests with the `src/learn/` package at 91% line coverage, and
it establishes that the scaffold is wired, bounded, reversible and
unreachable-by-the-agent.  It does not establish that the agent is better,
and it is not intended to — nothing in a unit test can.  The A/B is the
instrument that answers that.

**The A/B can now be run.**  It could not, until October 23: the holdout
leaked into the agent's own prompt (§5.2, blocker 1), so any measurement
would have reported the answer key; and there was no code path that
replayed a split and graded the answer at all (§5.2, blocker 2).  Both are
fixed.  What remains is the run itself, which needs an endpoint and a
drawn holdout.

**The one thing that was true before and is not now.**  Through October 22, 2026,
`autonomy()` returned `min(level, OBSERVE)`, so every injection path that gated
on `autonomy() >= ADAPT` was unreachable code.  Loops A and B could be committed,
reviewed, and shipped, and none of them could ever run.  The cap now sits at
`ADAPT` — the highest rung that is actually implemented — and the cap is
*loud*: `requested_level()` and `capped()` tell the request apart from the
grant, and `learn report` prints `requested evolve (4) -- capped to adapt (2)`.
A silent cap is a run that believes it is evolving and is not.

**Why the ruler is first, and not last.** Three arguments converge on it.

1. **The measurement must precede the change.** A recall path built before a holdout
   exists can only be evaluated by feel, and feel is exactly the thing this track exists
   to eliminate. PAST-Bench's finding is that the null result is the common one for
   memory scaffolds — so a change shipped without a ruler is most likely to be shipped
   on the strength of a coincidence.
2. **The ruler must not be the agent's.** I7 (§4.0) is easiest to honor when the
   holdout is created *before* the loops exist, because there is no loop yet that could
   have influenced it. A holdout assembled after Loop A ships inherits whatever the
   loop already changed.
3. **It is the smallest deliverable with the largest downstream dependency.** Everything
   else in this list ends in a promotion decision, and a promotion decision needs a
   holdout. It is days of work, and it unblocks the other three weeks.

**What "ship only if it wins" means concretely.** Deliverable 2 does not ship because
it works; it ships because it wins on the holdout by more than the benchmark's stderr,
with no regression on the previous generation's passing tasks (§6, rule 2 and 3). A null
result is a *result*: it goes in `RESULTS.md` and the loop is not shipped. "The recall
path made the answers feel more informed" is not a result.

**The dependency that is easy to miss.** Deliverable 2 cannot be safely built before
owner isolation is real. Today all 1,010 records carry `owner: null` and `iter_records()`
does not filter by owner (§3.0, finding 5). A recall path built against that returns
**global** results — in the multi-user web deployment OnIt already supports, one tenant's
sessions land in another's prompt. Owner scoping is therefore part of deliverable 1, not
a follow-up: it is a precondition for the first episode ever being injected.

**What is explicitly not in this order.** No deliverable here changes the serving
stack's prefix-cache contract (§4.3), the security/containment layer, or the model
weights. The playbook rides at the *tail* of the static half so it survives the cache;
the recall rides in the volatile half because episodes are per-task and would thrash the
prefix. That placement is the single most load-bearing implementation detail in Loop A
and it is specified in §4.2.


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
  (web), the interactive CLI, and loop mode.
- 👍/👎 under every answer in the web UI, backed by `POST /api/rating`. The buttons are
  quiet until hovered, restore the verdict already given when history reloads, and a
  second click on the same thumb retracts it. They are not shown at all when recording is
  off — `/api/config` carries `rating_enabled` — so no click is ever silently dropped.
- `onit learn` — the answer to *"is any of this working?"*. Prints the autonomy level,
  the store path, what has been recorded, and the tool table ranked by failure rate.
  `--json` for machines, `--session <id>` to dump one session's records. Read-only, and
  deliberately usable when the model endpoint is down.
- `onit learn --skills` — the skill library, and the chain that fills it:
  `--skills-action mine` finds the recurring fragments, `synthesize` writes a draft,
  `gate` runs the five checks, `pin` serves it, and `rollback` unserves it.  Each step
  runs on its own, so the gate's verdict can be read before anything is served, and the
  gate refuses a skill that has not been measured unless `--no-benchmark` says so
  explicitly.

Deferred, on **two** gates. As of August 20 the second is met and the first is not,
*despite appearing to be* — which is the more useful of the two findings:

- **Freeze `eval/holdout.jsonl` and pin `eval/baseline.json`.** Still not done, and now
  squarely on the critical path (§8).
  - **Gate 1 — data. Met, as of October 22, 2026.** `onit learn` now reports **1,010
    tasks across 58 sessions** (was 209 across 23), and **820 carry a verifier label**.
    The "~29 usable" figure in earlier revisions described the pre-Phase-0.5 era and no
    longer describes the corpus. The remaining constraint is the *era* boundary, not the
    volume: the holdout must be drawn from records produced under the settled harness,
    identified by timestamp until the era stamp lands (§3, gap 6).
  - **Gate 2 — a settled harness. Met.** Phase 0.5 is complete. What it leaves behind is
    an obligation rather than more work: the `code_execution` stamping rule in §5, and the
    era stamp in gap 6.
  - **Gate 3 — a benchmark that can move. New, and not met.** Added August 20. A holdout
    is only a gate if a scaffold change can register on it; every registered task today is
    single-turn and near-saturated (§3, gap 5). This gate is cheap — GAIA and SimpleQA are
    already written — but it was invisible until the other two were nearly closed.

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

### Phase 0.5 — Settle the harness · autonomy 1 — **complete**

Ran *during* trajectory accrual, so it cost no calendar time against Phase 1. Full detail
in [`HARNESS_CAPABILITIES.md`](HARNESS_CAPABILITIES.md); this was the subset that had to
land before the baseline could be pinned, and why.

**All five shipped — and a sixth that this section had explicitly excluded shipped too.**

| # | Change | Why it gated the baseline | Status |
|---|---|---|---|
| **H1** | Typed I/O — validate args pre-dispatch; stop shipping `returns` on the wire | Moved `tool_errors` from server stack traces to harness refusals | ✅ done (Aug 09) |
| **H3** | Loop policy to config; **restore the iteration cap** | `turn_count` was unbounded, so its distribution was not a baseline | ✅ done (Aug 09) |
| **H2** | `context_status`, `note_write`, `note_read` | Shares its on-disk root and path-jail with Loop A/B (§4.0) | ✅ done — [`harness.py`](../src/model/serving/harness.py#L95-L128) |
| **H6** | `RunState` object | What `_record_trajectory` now serializes; supplies `stop_reason` | ✅ done — [`state.py`](../src/model/serving/state.py#L86) |
| **H4** | Tool-result store (pass-by-reference) | Collapses `truncations`/`compactions` — the largest baseline shift of the five | ✅ done — [`results.py`](../src/model/serving/results.py) |
| **H5** | Code as action — `run_code` + per-session interpreter | **Was excluded from this phase. Shipped anyway.** See below | ⚠️ done, default off — [`interpreter.py`](../src/model/serving/interpreter.py) |

**H1 and H3 shipped August 09, 2026** (suite 966 → 1028, all green). Two findings that
change this plan's own reading of its data:

- `turn_count` was genuinely unbounded — `MAX_CHAT_ITERATIONS` was `-1`, and the only other
  ceiling keys on byte-identical arguments. Any `turn_count` distribution recorded before
  this fix has no ceiling in it and must not be used to set one.
- `_build_parameters` dropped `required` from every discovered tool schema, so
  `blank_required_args` had **never fired in production**. Layer 0's `tool_errors` signal
  was therefore measuring a slightly different harness than the code appeared to describe.
  Trajectories recorded before this date carry that caveat.

**On H5 — the exclusion that did not hold.** The August 09 revision said code as action was
"explicitly *not* in Phase 0.5," because one `run_code` call replaces N tool calls and
`tool_calls` per task therefore stops being comparable across the boundary. It shipped
regardless: [`interpreter.py`](../src/model/serving/interpreter.py), offered as `run_code`
via [`harness.py`](../src/model/serving/harness.py#L199).

What saves the baseline is that it is **default off**
([`onit.py`](../src/onit.py#L89-L93)), for reasons unrelated to this plan — it runs
model-written Python with the process's privileges, and small models write worse Python
than they write JSON. So the corpus is uncontaminated *today*, and the constraint converts
from a schedule into a standing rule:

> **`code_execution` must be stamped into every trajectory record and every benchmark result
> row, and a baseline pinned with it off is not valid for runs with it on.**

That is stricter than deferring the feature would have been, and it is permanent rather
than temporary. It is not implemented — `build_record()` records `model` and
`tools_available` but nothing about the harness configuration that produced the run, which
is the same omission as gap 6 and should be fixed in the same change.

**Exit criterion — met.** `onit learn` reports a stable signal set, and `stop_reason` now
arrives on every record rather than none. The cost of getting here is recorded in gap 6:
the pre-August-18 corpus predates that stability and cannot be pinned against.

**Ordering within the phase** was H1 → H3 → H2 → H6 → H4, and held.

### Phase 1 — Recall *(~3 days)* · autonomy 2 — **built, unmeasured**

**Reordered October 22, 2026.** The "prefixed by the four cheap items in §8" note is
resolved: the verifier promotion is done (§3, gap 3), the benchmark registration is done
(GAIA and SimpleQA now run), and the data gate is open (§3.0). The one item that is
*not* done is the pin — and it is now the first deliverable of the whole track, ahead of
recall itself. See §5.0.

Phase 1 A/B-ing against an unmovable holdout with unlabeled episodes would be theatre.
The episodes are no longer unlabeled, and the holdout now exists as of deliverable 1 —
so the ordering is: build the ruler, then build the loop, then measure them against it.

- Implement `memories` in `_build_messages()`; populate at the three `'memories': None`
  sites — into the `RunState` from H6, not a fourth ad-hoc parameter.
- Rank retrieved episodes by the outcome signal from §8 item 1. Recall without a notion of
  which episode went *well* retrieves the agent's own mistakes as readily as its successes,
  which is the documented failure mode Reflexion exists to avoid.
- `src/learn/recall.py` — lexical retrieval over trajectories, token-budgeted.
- SQLite index over `trajectories/*.jsonl`, derived and rebuildable (§4.0).
- Reconcile with H2: `note_read` (this session's own notes) and recall (prior sessions'
  distilled outcomes) are one tool family from the model's point of view. Ship them with
  one naming scheme, or the model will use neither well.
- A/B on the holdout. **Ship only if it wins.**

### Phase 2 — Playbook *(~3 weeks)* · autonomy 2 — **built, unmeasured**

- `src/learn/playbook.py` — item store, delta ops, decay/eviction, scope isolation.
- `src/learn/reflect.py` — Reflector + Curator passes (offline, batch).
- Playbook block at the tail of the static half in `build_assistant_instruction()`;
  version pinned per session.
- `onit learn` CLI: `--status`, `--reflect`, `--promote`, `--rollback`, `--show`,
  `--forget <id>`.
- A/B on holdout + ≥2 benchmark tasks; write to `RESULTS.md`.

**Exit criterion**: a measurable, reproducible win on held-out tasks with playbook on vs.
off — or an honest negative result and a decision to stop.

**As built (October 22, 2026).**  `src/learn/playbook.py` — the item store, the
Reflector (`reflect`), the Curator (`curate`), delta-only updates
(`apply_deltas`), the token-capped selection (`select`), and the per-session
version pin.  The block rides at the *tail of the static half* of the
instruction, so the prefix cache survives it; the pin lives on the `OnIt`
instance rather than the run state, because a per-task pin would change the
static bytes per task.  `onit learn --playbook`, `--reflect` and `--rollback`
drive it.  **The A/B has not been run.**

**As built (October 22, 2026).**  `src/learn/recall.py` — BM25 over the recorded
trajectories, reusing the retrieval kit from `toolkit.py`, ranked by the outcome
signal so the agent's own failures do not come back as lessons.  The recalled
episodes ride in the *volatile* half of the prompt (§4.2).  The result is
`src/learn/eval.py`: the holdout, the dev split, the pinned baseline, and the
`assert_outside_agent_surface()` check that makes I7 a code path rather than a
promise.  `onit learn --holdout` draws the split; `onit learn --eval` reports
it.  **The A/B has not been run** — the split is drawable, the baseline is
writable, and the win is unproven.

### Phase 3 — Skills *(~4 weeks)* · autonomy 3 — **built, unmeasured**

- `src/learn/skills.py` — mining, synthesis, five-gate validation.
- `src/mcp/servers/tasks/skills/mcp_server.py` — serves pinned skills; config entries.
- Wire GAIA in `benchmarks/tasks/agentic.py` as the tool-creation benchmark.
- `onit skills` CLI: `list`, `test`, `pin`, `unpin`, `rm`.

**As built (October 22, 2026).**  Four modules and a server:

| Module | What it is |
|---|---|
| `src/learn/skills.py` | The store: manifests, the pin, delta-only updates (`insert` / `update` / `deprecate`) |
| `src/learn/skills_mining.py` | The Reflector: groups calls by `(tool, args_digest)`, counts distinct *sessions* |
| `src/learn/skills_gate.py` | The five checks: manifest, source, dependencies, tests, regression |
| `src/learn/skills_curator.py` | The Curator: synthesize → gate → pin → rollback |
| `src/mcp/servers/tasks/skills/mcp_server.py` | Serves exactly what the pin names |

**The mining trick.**  Trajectories redact argument *values* — the values are
where file paths, search queries and the occasional pasted credential live — but each
call also carries an `args_digest`, the SHA-256 of the full argument mapping.
Two calls with the same digest made the same call with the same arguments.
So the miner finds the repetition by digest and never writes a value down.  On
the real corpus that is **36 candidates with 36 unique names**.

**The safety property, stated as one sentence.**  A skill is written, gated,
and only then pinned — in that order, every time.  A skill that fails the gate
exists on disk and is not in the pin, so it is not served; there is no path
from "exists on disk" to "in the model's schema" except through the gate.

*October 28, 2026 — the gate was bypassable, and the pin was not the only
condition.*  The source check ran a denylist of regexes over the source
**text**, over an allowlist that contained `os`, `sys`, `subprocess`, `socket`,
`pathlib`, `pickle`, `importlib` and `urllib`.  A skill could therefore import
only `sys`, subscript `sys.modules` to reach `os`, and assemble `system` from
two string fragments — passing all five checks and then running `echo pwned`
in-process, because the server `exec_module`s the skill's `tool.py` in the
agent's own process.  Three changes close it:

- **the allowlist is pure computation.**  No module that reaches the
  filesystem, a process, the network or the import machinery.  `os` and `sys`
  are refused *because they are the escape*, not because they are unusual.
- **the check is structural, not textual.**  The source is parsed and walked:
  a forbidden builtin called by name (`open`, `eval`, `exec`, `compile`,
  `globals`, `setattr`, `__import__`, …), a `getattr` whose attribute name is
  not a literal, any dunder attribute (`().__class__.__bases__` reaches every
  loaded class with no import at all), any subscript of `.modules`.  The
  regexes stay as a backstop for source that is compiled but never parsed.
- **the server honours the level.**  Serving requires `extend`; below it
  `load_skills()` returns nothing and logs which skills it withheld.  Before
  this, the server served at *every* level including `off`, so the gate was
  the only thing between a mined fragment and a running tool — and the gate
  was bypassable.

*October 28, 2026, the second round.*  Those three changes refused the audit's
payload and were still not enough: five more spellings of the same reach passed
`check_source`.  Each one defeated a *different* screen, and each one is a
lesson about the shape of the problem:

| what passed | which screen it walked past | why |
|---|---|---|
| `__builtins__['open'](p)` | the attribute rule | the dunder is a `Name`, not an `Attribute` |
| `f = open` · `def run(f=open)` | the call rule | `open` never appears in front of a bracket |
| `io.FileIO(p).read()` | the allowlist | `io` was on it, and it exports file objects |
| `'{0.__globals__}'.format(f)` | both AST rules | the dunder is *data*; `str.format` walks the chain |
| `attrgetter('__globals__')(f)` | both AST rules | the same reach through a callable |

The fix is four more rules, all structural: a bare dunder name is refused
(`__name__`, `__doc__`, `__all__`, `__version__`, `__init__` excepted, because
they are metadata); a forbidden builtin used as a *value* rather than called is
refused; a dunder spelled inside a string literal is refused, with docstrings
exempt, because prose is not a capability; and `io` and `types` came off the
allowlist, which now admits a module only if every name it exports is pure
computation.

*October 29, 2026, the third round.*  The second round was checked against the
one payload the audit had written, and it refused that payload.  It was still
not enough, and the reason is the same reason as before, one level down: every
rule added in the second round was about an **import** or a **dunder**, and the
next escape needed neither.

Every module on `STDLIB_ALLOWED` is a stdlib module, and stdlib modules import
each other.  `uuid` imports `os` and keeps it as `uuid.os`; `random` keeps it as
`random._os`; `warnings`, `typing`, `enum`, `fractions` and `statistics` keep
`sys`; `json` keeps `codecs`, which keeps `builtins`.  So:

| what passed | why every screen missed it |
|---|---|
| `import uuid; uuid.os.system(cmd)` | the import is `uuid`; `os` is an attribute |
| `import random; random._os.system(cmd)` | the same, and the regex `\bos\.system\b` never matches `_os.system` — `_` is a word character, so the boundary does not fall before `os` |
| `operator.attrgetter('os')(uuid).system(cmd)` | the name is *data*; no `Attribute` node for `os` exists |
| `getattr(uuid, 'os').system(cmd)` | the same, through the builtin |
| `json.codecs.builtins.open(p)` | three attributes deep, all on the allowlist |
| `warnings.sys.modules` | `sys`, as an attribute of an allowed module |
| `fnmatch.os.remove(p)` · `argparse._os.popen(c)` | the file and process capabilities, one dot from an allowed import |

The first two were confirmed to **execute a shell command** through
`load_skills`.  The fix is one rule about attributes rather than imports: a
skill may not reach an attribute named like a forbidden module (`os`, `sys`,
`builtins`, `_os`, …) or named like a process or file capability (`system`,
`popen`, `execv`, `remove`, …) **off a module it imported**.  Scoping to the
module root is what keeps `xs.remove(1)` and `json.loads(t)` admissible; a rule
about bare attribute names would refuse both, and a screen that refuses ordinary
code is a screen somebody deletes.

**And the gate now runs twice.**  Curation ran it once and recorded the verdict
in the pin.  But the pin is a file, and the file a running server reads is not
necessarily the file the gate approved — a copied store, a hand-edited pin, a
gate that was weaker when the skill was pinned.  `load_skills()` therefore
re-runs `check_source` on every serve and skips a skill that no longer passes,
logging why.  Without that, the pin is the boundary and the gate is advice.

**The lesson, and it is the one that matters for the rest of this document.**
A denylist of *spellings* is a denylist an attacker writes around, and a static
screen is a screen that only ever proves the spellings someone thought of. The
gate is now good enough to stop the escapes above; it is not and cannot be a
sandbox. This document says the tests run in a sandbox (I6) and they do not, and
until they do the honest statement is: **the gate is a filter, not a boundary,
and the boundary is the container.**  Running a pinned skill in-process is the
remaining risk, and it is not a risk a static check can retire.

**What is not done.**  The gate does not *run* the skill's tests yet — that is
the sandbox's job (I6) and it is not wired.  GAIA is not wired as the
tool-creation benchmark.  The gate's regression check therefore has no
measured delta to read, which is why it refuses an unmeasured skill by
default: `onit learn --skills pin NAME` without a `--delta` says no.

### Phase 4 — Evolution *(~6 weeks)* · autonomy 4 — **not started**

- Fix the custom-template prefix-split bug **first**.
- `src/learn/evolve/` — GEPA-style prompt optimizer over `benchmarks/` tasks (D1).
- Archive + CMP-based parent selection for code variants (D2), worktree-isolated.
- Every promotion is a PR with the fitness record attached.

---

### 5.1 What the tests actually establish *(added October 23, 2026)*

The six deliverables are built.  This subsection records **what the test
suite proves about them and what it does not**, because the two are easy
to confuse and the difference is the subject of this document.

**The suite.** 2,915 passing, 2 skipped, across the repository.  The
`src/learn/` package went from 88% line coverage to **93%** when these
were added:

| Module | Before | After | What the gap was |
|---|---:|---:|---|
| `_search.py` | 48% | **99%** | The BM25 *fallback* — the path that only runs when the optional toolkit is missing, and therefore the path nobody reproduces before shipping. |
| `skills.py` | 78% | **98%** | The pin's delta operations, the manifest validation rules, and the name rules — the three things that decide what the model is allowed to call. |
| `playbook.py` | 90% | 91% | The delta operations and the selection ordering. |
| `signals.py` | 91% | **96%** | The degenerate inputs and the determinism property. |
| `recall.py` | 95% | **96%** | The budget edge and the outcome ranking. |
| `eval.py` | 93% | 93% | The write paths and the sealed checks. |
| `actuator.py` | 88% | 88% | The write side of the withholding. |

**Five new test files, and why each exists.**  The per-module test files
that shipped with each deliverable check that module's own contract.
These five check the things that only exist *between* modules, which is
where the failures that matter live:

- [`test_learn_invariants.py`](../src/test/test_learn_invariants.py) — the
  cross-cutting invariants (I1, I2, I3, I4, I7), written adversarially:
  each test tries to *break* an invariant rather than confirm it.
- [`test_learn_signal_properties.py`](../src/test/test_learn_signal_properties.py) —
  the implicit signals as **properties**, with generated inputs rather
  than a list of examples.  The failure mode of a heuristic labeler is
  not a case someone thought of; it is a case nobody thought of.
- [`test_learn_chain.py`](../src/test/test_learn_chain.py) — the whole chain
  end to end: record → label → split → mine → gate → promote → serve → roll
  back.  A chain test is the only kind that catches a signature that
  drifted or a path that is right in isolation and wrong in sequence.
- [`test_learn_search.py`](../src/test/test_learn_search.py) — the search
  fallback, checked against the toolkit it mirrors.  The two must fold
  plurals *identically*, quirks included ("this" → "thi" in both),
  because a query folded one way against a document folded another is a
  match that silently misses.
- [`test_learn_skill_store.py`](../src/test/test_learn_skill_store.py) — the pin,
  the deltas, the manifest rules and the name rules.

**One real bug, found by the new tests.**  The skills gate's screen for
paths outside the agent's writable surface matched a forbidden directory
name only when a *quote* came immediately before it.  A real path has a
*separator* before the directory name, so `open('/x/eval/dev.jsonl')` and
`open('../eval/dev.jsonl')` both passed the gate — which is the I7 hole
(§4.0), open to any skill that wanted to read the measuring instrument.
The screen now matches the name as a **path component**, so
`'/x/eval/dev.jsonl'` is refused while `'myeval/x'` and `'evaluate'` are
not.  The regression test is in `test_learn_invariants.py`.

**What the tests do not establish.**  They say nothing about whether any
of this *helps*.  A test suite that proves the recall path is wired,
that the playbook is token-capped, that the pin is reversible and that
the ruler is unreachable by the agent is a statement about the *scaffold*.
Whether the scaffold makes the agent better is a measurement, and the
measurement has not been made.  That is the next piece of work, and it
is the only piece that matters for the claim this document is about.

**The one thing the tests cannot do.**  They cannot run the A/B.  The
holdout exists and the gate can read it, but the gate reads a
*reported* delta: `onit learn --skills pin NAME --delta D` refuses
without one, and there is no code path that produces the delta for it
yet.  Until the benchmark harness runs the candidate against the pinned
baseline and reports the number, the promotion gate is a door with
nothing behind it — which is the correct state for it to be in, and
much better than a gate that says yes to everything.

### 5.2 The four blockers, and what unblocked them *(added October 23, 2026)*

The six deliverables were built, and the A/B was still unrunnable.  Four
structural problems stood between the code and a measurement, and the
first of them would have produced a **fake win** rather than a failure.

#### Blocker 1 — the holdout leaked into the agent's own prompt

`recall()` searched the whole trajectory store, and the holdout is drawn
*from* that store.  So on a holdout item, the agent was handed its own
previous answer before it was asked the question.  Measured on the real
corpus: **158 of 160 holdout items recalled a run of their own
question.**  Any A/B measured on that split would have been reporting
the answer key.

The obvious fix — exclude the holdout's *record ids* — is not enough,
and this is the part worth stating plainly.  The corpus holds the same
question asked many times: "weather today" was asked 21 times, "marikina
river water level latest" 6.  Those 21 runs are one question with 21
answers, and the holdout excluded exactly one of them.  The other 20
stayed in the recall pool, each carrying the answer.

So the exclusion is by **question**, not by record.  `eval.task_key()`
normalises a task to its identity as a question (case, punctuation and
whitespace removed), and `eval.holdout_exclusions()` returns both sets —
the ids *and* the question keys.  Every loop that reads the corpus now
goes through one predicate, `eval.is_excluded()`.

| | before | after |
|---|---:|---:|
| holdout items recalling their own question | 158 / 160 | **0 / 160** |

The loops that read the corpus and now exclude the holdout: `recall()`,
`reflect()`, skill mining, and the tool-lifecycle statistics.  The
exclusion is *on by default* in each, so a new loop that forgets to ask
for it is the only way to leak — and the test suite pins the default.

The second half of the leak was subtler, and the final verification
pass found it.  A bullet's `provenance` named the **session**, and a
session file holds one record per turn — so a session could contain a
turn whose question was in the holdout *and* a turn whose question was
not.  The exclusion dropped the holdout turn correctly, but the
surviving turn's session id still pointed at a session that contained
one.  The holdout's *answer* was never in a lesson; its *existence* was
in the pointer that says where the lesson came from, and the pointer is
what someone checking "did this come from the exam?" would follow.  The
provenance is now the record id (`session_id#turn`) — the same identity
the exclusion set is keyed on, so a bullet's provenance can be checked
against the exclusion set directly.  Measured on the real corpus: 27 of
57 sessions held both kinds of turn; the leak is closed.

#### Blocker 2 — there was no eval runner

`eval.py` held the ruler as *data*; nothing replayed it.  There was no
code path that took a holdout item, asked the agent the question, and
graded the answer.

`src/learn/runner.py` is that path, and `src/learn/ab.py` drives it:

```
onit learn --eval-run                 # one arm, the numbers
onit learn --eval-run --ab             # both arms, and the difference
onit learn --eval-run --eval-split holdout --eval-limit 50
```

The agent is driven through `OnIt.process_task` — the same entry point
the benchmark provider uses — so a number produced here means what it
would mean in a real run.  The grader is a **separate model** with no
tools (`src/learn/llm.py`): a judge that could run the agent's tools
would not be a judge.

Two things the runner refuses to get wrong:

- **The arms do not contaminate each other.**  Each gets its own
  `learn.path`, so the treatment arm's recall pool is the control arm's
  recorded runs and nothing else.  Neither arm writes into the
  operator's own corpus.
- **The comparison carries an error bar.**  The delta between the arms
  is reported with `1.96 × sqrt(se_control² + se_treatment²)` — the
  standard error of the *difference* of two independent proportions, at
  the two-sided 95% quantile — and a delta that does not clear it is
  reported as **not significant**, with the instruction to ship nothing
  on that evidence.

  *October 28, 2026.*  The bar used to be the **sum** of the two arms'
  standard errors, and the docstring called it "conservative".  It was
  the opposite: with equal `n` the sum is `2s` and the standard error of
  the difference is `sqrt(2)s`, so the test was `1.414σ`, not `1.96σ`.
  A null difference cleared it **15.7%** of the time instead of 5% —
  measured at 15.9% over 3000 simulated null pairs.  A placebo arm had
  roughly a one-in-six chance of being reported as a win, which is the
  failure that matters: not a missed real effect, but a shipped nothing
  that looked like something.  The bar is now `1.96σ_diff`, the null
  rate is measured at 5.2%, and `TestTheErrorRate` asserts it.

#### Blocker 3 — benchmark runs polluted the corpus they were measured against

`benchmarks/config.py`'s `resolve_learn()` set `autonomy` but not
`learn.path`, so a benchmark's trajectories landed in
`~/.onit/learned/trajectories` — the same corpus the recall path
reads.  A run at `ONIT_LEARN=adapt` would have fed benchmark traffic
straight back into the path under test.

`resolve_learn()` now sets `learn.path` to the benchmark's own store,
beside its data root, overridable with `ONIT_BENCH_LEARN_PATH`.

#### Blocker 4 — the Reflector could only produce templates

The mechanical Reflector clusters runs by their tool signature and
contrasts the good ones against the bad ones.  But a cluster *is* a
tool signature, so within a cluster the good runs and the bad runs ran
the same tools — and the "a tool the good runs used and the bad runs did
not" branch was **empty by construction**.  Measured on the real corpus:
that branch fired **0 times** in 677 clusters.  Every bullet came from
the weakest branch, which is why the playbook read as four near-identical
templates with a tool name in them.

Three fixes, in order of how much they matter:

1. **The cluster key is now the shape of the task** (`playbook.task_shape()`
   — the first three content words), and the contrast runs along the
   tools *inside* a cluster.  Two runs of the same kind of question are
   comparable; that is what a contrast needs.  Branches 1 and 2 now fire
   **48 times** where they fired 0.
2. **The model-based Reflector exists** (`src/learn/reflector.py`).  It
   hands a model two contrasting runs and asks for the difference a
   person reading both transcripts would name — the arguments, the
   order, the reasoning — and parses the reply strictly.  The model is
   injected as an `ask` callable, so it is testable without a network
   and it never borrows the agent's tool registry.
3. **`onit learn --reflect` now runs the model-based reflector first**
   and the mechanical one second, with `--mechanical-only` and
   `--model-only` to choose.  The mechanical bullets are weaker, so they
   go last, and a run with no model available still reflects.

#### Three pre-existing bugs found on the way

These were not introduced by the work above; they were *found* by it,
and each one had made the ruler unusable.

| # | Bug | Effect |
|---|---|---|
| 1 | `write_split()` wrote `split.json` with only the metadata — the `holdout` and `dev` lists were stripped out and never written, and the file was double-encoded through the JSONL writer. | `read_split()` returned **empty lists for every split ever drawn**. The split filter had nothing to filter on, and the CLI's `--eval` reported 0/0 for a split that existed on disk. |
| 2 | `cluster()`'s sort used `len(buckets)` — the number of *keys* — instead of `len(buckets[key])`, the number of *records*. | The "largest cluster first" order was alphabetical by accident. |
| 3 | `run_arm()` called `make_agent(config)` without awaiting it, and the real factory is async (the agent's constructor cannot run inside a running event loop). | Every item recorded an error, both arms scored zero, and **a run that measured nothing looked exactly like a run that measured a null result**. |
| 4 | A bullet's `provenance` named the *session* (`session_id`) rather than the *run* (`session_id#turn`). A session file holds one record per turn, and 27 of the corpus's 57 sessions hold both a holdout turn and a surviving one. | The holdout's answer was never in a lesson — the exclusion dropped that turn correctly — but the pointer that says where a lesson came from named a session that contained a holdout run. **The holdout's existence leaked through its provenance**, and the provenance is exactly what an audit follows. The provenance is now the record id, the same identity the exclusion set is keyed on, so a bullet's provenance can be checked against the exclusion set directly. |

Bug 1 is the one to remember: the ruler had been written and never read
back.  The holdout existed on disk and `onit learn --eval` said `0`
because the file it read was not the file that had been written.

#### What is still missing

**The A/B has not been run.**  It can now be run, and it has not been:
that needs a model endpoint and a drawn holdout, and both are the
operator's to provide.  Until it is run, the honest statement is
unchanged — the scaffold is built, the instrument is built, the
measurement is *not made*.

And the measurement has a shape it will not have had before: the
holdout exclusion is by question, so a holdout item is genuinely
unfamiliar to the agent even when the corpus contains the same question
twenty times.  That makes the holdout smaller and the result harder to
win — which is the right direction for an instrument that has to be
believed.


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

| Risk | Mitigation | Status |
|---|---|---|
| **Reward hacking** — proxy metrics gamed ([2606.15385](https://arxiv.org/abs/2606.15385)) | Frozen holdout; quality paired with efficiency (I4); judge rotation; human read of fast wins | partial |
| **Context collapse** — rewrites erode knowledge (ACE) | Delta-only updates (I2); itemized store; never regenerate | built |
| **Playbook bloat** — memory crowds out the task | Hard token cap; decay + eviction; attribution-driven pruning | built |
| **Prefix-cache thrash** — learned context invalidates the prefix | Playbook at tail of static half; version pinned per session; latency is a tracked metric | built |
| **Cross-user leakage** — one tenant's data in another's prompt | Owner-scoped by default; redaction; ≥3-owner rule for global promotion | built |
| **Skill-library dilution** — more tools, worse selection | Benchmark delta ≥ 0 required per skill; deprecation path; description quality gate | partial |
| **Runaway self-modification** | Sandbox (I6); allowlisted files; PR-gated; `pinned.yaml` rollback (I3) | partial |
| **Forgetting** | Stability axis blocks promotion on prior-generation regressions | open |
| **Baseline drift under the harness** — a learning win that is really a harness change, or a harness regression hidden by a learning win | Phase 0.5 gate before the pin (§1.1); every result row stamps the harness phase alongside `ONIT_LEARN`; any post-pin harness change to the loop, the result store or the tool contract forces an explicit re-pin, announced in `RESULTS.md` | partial |
| **An insensitive benchmark** — the holdout cannot register a scaffold change, so every loop reads as a null result | Gate 3 (§5): at least one agentic, unsaturated task registered before the pin. A loop that cannot lose on the holdout cannot win on it either | open |
| **Signal starvation** — loops trained on unlabeled runs learn confident nonsense | Promote the verifier already recorded (§3, gap 3); implicit signals; never rely on a human remembering to click | built |
| **Effort sink with no payoff** | Phase gates. Each phase must win on the holdout or the track stops. Phase 0 has standalone value regardless | open |
| **The agent grades its own homework** *(added Oct 22)* — a loop edits the holdout, the judge, or the baseline, and "improvement" becomes a rewritten exam | I7: the ruler lives outside `data_path` and outside the learned store, by layout not by policy (§4.0). Holdout and baseline are read-only to the agent's own write surface | built |
| **Cross-tenant leakage through recall** *(added Oct 22)* — owner isolation is inert, so episode retrieval returns global results | Owner-scoped recall, enforced in the query rather than by convention (§4.2, §5.0). `iter_records()` gains an owner filter before the first episode is injected | built |
| **Stored prompt injection** *(added Oct 22)* — a playbook bullet or episode distilled from a fetched page is a durable instruction the agent later obeys | Untrusted marking on injected blocks; write-isolation between the recall path and the store it reads; the injection surface is treated as untrusted by construction, not by hope | partial |
| **The autonomy ladder lies** *(added Oct 22)* — a config asks for `adapt`, gets `observe`, and the run believes it is adapting | Either fail loudly when a level above what exists is requested, or rename the levels to what exists. The cap is right; the silence is the bug (`_MAX_IMPLEMENTED`) | built |

**What the status column means.**  *built* — the mitigation is code that exists and is
tested.  *partial* — the mechanism exists, but the thing it protects has not been
exercised, so it is unproven.  *open* — not started.  The distinction between *built* and
*partial* is the distinction this document is about: a gate that has never
fired has not been shown to hold.

**Explicit non-goals for v1**: no weight updates or finetuning; no unattended modification
of the running process; no autonomous changes to the security/containment layer; no
network-reachable self-modification surface.

---

## 8. Recommendation

*Revised October 22, 2026, twice on the same day: once after the codebase review,
and again after the six deliverables of §5.0 were built.  The destination is unchanged — the playbook is
still the main event — but the recommendation has changed, because the thing that was
missing is no longer the code.*

**The recommendation in one line: run the measurement, before building anything else.**

The previous version of this section said *build the ruler before the loop*.  That was
right, and it is done: `src/learn/eval.py` draws the holdout and the dev split, pins the
baseline, and keeps the whole apparatus outside the agent's writable surface with an
assertion rather than a convention.  Loops A, B and C are built, tested, and reachable
from a config file.

**So the remaining question is not "what should we build" but "does any of it work".**
And that question has one answer, and it costs a benchmark run:

1. **Draw the holdout** → `onit learn --holdout`.  Freezing it *after* the loops
   exist is a weaker version of I1 than freezing it before, and the RFC says so; the
   honest move is to draw it now and to treat the draws' provenance as recorded.
2. **Run the A/B** → the same tasks, `ONIT_LEARN=off` against `ONIT_LEARN=adapt`,
   scored against the pinned baseline.  Write the numbers to `RESULTS.md` whether they
   are good or not.
3. **Ship only what wins.**  PAST-Bench's finding is that the null result is the common
   one for memory scaffolds.  A null result is a *result*: it goes in `RESULTS.md`, and
   the loop does not ship.  "The recall path made the answers feel more informed" is
   not a result.
4. **Then, and only then, Loop D.**  Six weeks of scaffold evolution on top of an
   unmeasured loop is six weeks of measuring the wrong thing.

**Why the order has not changed, only the position on it.**  The RFC's warning is
still the governing one: a self-improvement loop that ships before the ruler exists can
only be evaluated by feel, and feel is the thing this entire track exists to eliminate.
That argument is now about the *measurement* rather than about the *ruler*, and it is
the same argument.  The ruler exists; the measurement does not.  Those are different
states, and only the second one decides whether any of this is worth having.

**The six deliverables, and where each one stands.**

| # | Deliverable | Status |
|---|---|---|
| 1 | Holdout + baseline + the I1/I7 ownership invariant | **built** |
| 2 | Recall (Loop A) | **built, unmeasured** |
| 3 | Implicit signals + text-UI rating | **built** |
| 4 | Playbook (Loop B) | **built, unmeasured** |
| 5 | Lifecycle actuator | **built** |
| 6 | Skills (Loop C) | **built, unmeasured** |
| 7 | Evolve (Loop D) | **not started** — and it should not start until 2 and 4 have won |

**The one-line summary of what changed on October 22, 2026.**  The track stopped being
a plan and became a codebase, and in the process it acquired the thing a plan cannot
have: a ruler, and the obligation to use it.

**Items 1–4 of the old list, and where each one now stands.** Items 1 and 2 are done;
item 3 is deliverable 3; item 4 — the holdout — has been promoted from last to first.

**1. Promote the verifier signal already being recorded** *(~hours)*. `verify.py` runs on
every answer by default; `verify_issues` and `verify_revisions` are already in the `metrics`
blob. `derive_signals()` writes `"verifier": None` next to them. Read the field instead.
This is the highest ratio of unlocked capability to lines changed in the
document: it converts an unlabeled corpus into a weakly-labeled one retroactively, across
data already on disk. Ratings were never going to do this — 1,010 tasks produced zero.
**Done September 04, 2026** — `verifier_signal()` in `src/learn/trajectory.py` derives
`issues` / `clean` / None (None = the check never ran, which is not "clean"), reported in
`learn report`; `scripts/backfill_verifier_signals.py` relabeled the corpus in place:
395 records → 298 clean, 29 issues, 68 unchecked (the corpus has since grown to 1,010
records: 820 clean / 70 issues / 120 unchecked). The issue-labeled records are the
seed of Loop B's training set.

The August 09 revision assumed the critical path was *data accrual* and filled the wait
with harness work. That was right, and it worked. But the corpus it produced is thinner
than it counts (the corpus has since grown to 1,010 records, 820 labeled — §3.0), and the
ruler it was going to be
measured with cannot detect the thing being measured (§3 gap 5). So the blocker is no
longer data. It is a **signal** and a **ruler**, and both are days away rather than weeks —
which means Phase 2 can start *sooner* than the last revision projected, not later.

**2. Register GAIA and SimpleQA** *(~hours)*. Written, unreachable, two lines (§3, gap 5).
Until the suite holds one task whose score can *move* when the scaffold changes, §6's
measurement panel is decorative and freezing a holdout would only formalize the mistake.
**Registered September 04, 2026** — `simpleqa` and `gaia` in `benchmarks/run.py` TASKS +
`factuality` / `agentic` categories; `python -m benchmarks.run --list` shows both.
**First run September 05, 2026** — SimpleQA smoke: 1.000 (5/5), judge path validated,
judge-rotation regrade agreed 5/5 (see item 4's progress log). GAIA still not run:
`GatedRepoError 401` without `HF_TOKEN` (accept the dataset terms, then `hf auth login`
or export `HF_TOKEN`); its wrapper's `trust=True` was also removed — datasets 5.x
rejects it, and the hub repo is Parquet-backed since October 2025.

**3. Add implicit outcome signals** *(~2 days)*. A same-session rephrase of a task ≈
failure; a session that ends after an answer ≈ acceptance. Both are already latent in the
corpus and can be mined retroactively, like item 1. Together the two turn ~1,010 unlabeled
records into a usable training set for Loop B's Reflector without waiting for new usage.

**4. Then freeze the holdout and pin the baseline** *(days)* → **now deliverable 1.**
Per model, not globally (§3, gap 7), stamped with `code_execution` and a harness era
(§5). Draw the tasks from real sessions but **re-run them under the current harness to
score them**, which sidesteps Gate 1's thinness without waiting weeks for it to resolve
on its own.
**Partially done September 05, 2026** — `benchmarks/baselines/full.json` pins the two
verifiable full-tier rows (gsm8k 0.977/1,319; bigcodebench 0.564/486 recorded) with a
provenance block naming the harness era. humaneval/mbpp are deliberately absent: their
only surviving logs are 0-tool harness-failure runs and their superseding runs left no
logs — re-run them before they enter the pin. The holdout split itself is still not
drawn.
**Updated October 22, 2026** — the holdout wait described above is over. The sampled-tier
runs landed (simpleqa 0.940 / stderr 0.024, humaneval 0.970 / 0.017, mbpp 0.990 / 0.010
— all on `glm-5.3-flash:cloud`, 14 tools live, learn off), so the measurement panel has
tasks whose score can *move* and a validated judge path. GAIA's blocker is also
resolved by a route the old note did not anticipate: the dataset was cached locally on
2026-09-06, so the task now loads offline with `HF_DATASETS_OFFLINE=1` and needs no
token. What remains is to **draw the split**, pin it, and put it where the agent cannot
reach it (I7).

**One caveat carried into the split.** Those sampled rows are *cloud-endpoint reference
points*, not pin-gated regression-gate numbers — they are on `glm-5.3-flash:cloud`, not
the pinned `Qwen/Qwen3.8-27B`. The holdout must be pinned per model, and the corpus
says why (§3, gap 7): the same task on a different model is a different measurement.
GAIA also runs **text-only by construction** today — its wrapper passes only
`Question`/`Final answer` and ignores `file_path`, so attachment tasks are unanswerable
as specified. That is a known scope limit to fix before GAIA enters the holdout.

**Progress log (September 05, 2026):** item 2's runs started. SimpleQA smoke: **1.000**
(5/5, model-judged, 14 tools live, 2 m 43 s) — the judge path is validated and the task
enters the measurement panel. Judge rotation (glm-5.1:cloud regrade) agreed 5/5. The
session also fixed a benchmark-harness bug found by the run: MCP server *spawn children*
crash-looped (`ModuleNotFoundError: mcp.types`) because they inherit the runner's
`sys.path` with `src/` first — the `55b7d0e` fix pinned the SDK in the parent's
`sys.modules` only, and spawn children start with a fresh module table. Fixed by
demoting the `src/` path entry before the SDK imports in `benchmarks/onit_provider.py`
and, as defense in depth, at module import in `src/mcp/servers/run.py`. GAIA attempted
and documented: `GatedRepoError 401` without `HF_TOKEN`; also fixed the task wrapper's
`trust=True`, which datasets 5.x rejects outright (the hub repo has been
Parquet-backed since October 2025, so no `trust` flag is needed).

**Progress log (September 06, 2026):** item 2's sampled-tier runs **completed** —
simpleqa **0.940** (94 C / 6 I / 0 not-attempted, stderr 0.024), humaneval **0.970**
(stderr 0.017), mbpp **0.990** (stderr 0.010), all on the agent's own preferred endpoint
(`onit/glm-5.3-flash:cloud`, 14 tools live, learn off). The humaneval/mbpp exclusion
from the pin now has a fix in flight: these sampled rows are the first verifiable
verbatim-dataset measurements for both tasks. GAIA's token blocker is **bypassed for
cached configs**: the dataset was cached locally on 2026-09-06 (validation 165 rows),
and the task loads offline with `HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1` — no token
needed for a config already in `~/.cache/huggingface/`. GAIA smoke run started; scope
caveat: the task wrapper ignores `file_path`, so GAIA runs are text-only by
construction. SimpleQA full-tier run (4,472 samples) started in the background.
Remaining for item 4: consolidate the sampled rows into the pin (per-tier), draw the
holdout split, and complete the GAIA + SimpleQA-full runs.

Only then Phase 1, then Phase 2. Phase 2 still carries low risk and still captures the
majority of the literature's demonstrated scaffold gains — ACE reports double-digit
improvements on exactly this mechanism, without labels and without gradients, on small open
models.

**If only one of the four lands, make it (1).** It is also the cheapest, which is a rare
alignment. Loops A and B both consume outcome-labeled trajectories: without labels, Loop A
retrieves episodes it cannot rank — surfacing the agent's own failures as readily as its
successes — and Loop B reflects on runs it cannot tell apart. That is precisely the failure
mode where a self-improving agent confidently learns the wrong lesson, and where a frozen
holdout is the only thing standing between that and production.

**A note on this document's own reliability.** Between August 09 and August 20 it claimed
H2, H4 and H6 were pending when all three had shipped, and claimed H5 was excluded when it
had shipped too. Three weeks of completed work were invisible to the plan that scheduled
it. Layer 0 exists because an agent cannot improve on experience it never wrote down; the
same applies to the roadmap. **Re-verify §3 and §5 against the code at the top of every
revision**, as this one did — the gap list is the part that rots, and it is also the part
every scheduling decision is made from.

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
