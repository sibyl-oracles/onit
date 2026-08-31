# Self-Improvement Gaps

**Status as of commit `f1552ca` (main, Aug 31 2026).**
Companion to `docs/SELF_IMPROVEMENT.md` (the RFC). This file is the gap assessment: what the framework already has for self-improvement, what is missing, and the critical path to close the loop.

**Updated Aug 31 2026** with the framework from [Recursive Self-Improvement](https://www.philschmid.de/recursive-self-improvement) (Philschmid, Aug 21 2026) — see *External lens* below. All code claims re-verified at `f1552ca`.

## Verdict

**Not self-improving yet — it is a *recording* agent, not a *learning* agent.**
The substrate (the hardest part to retrofit) is done, and the plan for the rest is well-specified. What is missing is the entire feedback half: the path from "recorded" to "changed behavior." The framework is ready to *build* a self-improving agent; as of this commit it learns nothing it can act on.

The external lens sharpens the target: what Loops A/B would deliver is **bounded self-improvement** — the agent changes its own prompts, memory, and tools, and keeps what works. That is worth having and is not recursion. **Recursion** additionally requires the system to face a *harder, still-honest bar* each round — i.e. the verifier improves too, without the system capturing it. No public system does this; it is explicitly out of scope here. What is in scope is not accidentally defeating the one we have.

## Already in place (verified in code)

| Piece | Status | Location |
|---|---|---|
| Trajectory recording | ✅ Done — per-turn tool calls, errors, retries, truncations, compactions, token metrics, per-session JSONL | `src/learn/trajectory.py` |
| Outcome signals | ✅ Partial — tool errors/retries are free; `POST /api/rating` (👍/👎) wired into web UI and lands in the store | `src/learn/`, `src/ui/api.py` |
| Autonomy ladder (`learn.autonomy`) | ✅ Defined 0–4 (off/observe/adapt/extend/evolve), but **capped at 1** — `_MAX_IMPLEMENTED = OBSERVE`; a config asking for `adapt` silently gets observe | `src/learn/config.py` |
| Fitness function | ✅ `benchmarks/` (inspect-ai: bigcodebench, gsm8k, humaneval, mbpp, SWE-bench) with real numbers | `benchmarks/`, `RESULTS.md` |
| Harness (Phase 0.5) | ✅ Mostly shipped — `RunState` (resume), `ResultStore` (handles), harness tools, verify pass, interpreter (default off) | `src/model/serving/` |
| Extension-friendly harness | ✅ MCP-tool-based; tools discovered/loaded at runtime (`src/type/tools.py`, `src/lib/tools.py`) — the substrate for the "harness as extensions" direction (see External lens) | `src/mcp/` |

Trajectory data is already accumulating in `~/.onit/learned/` (23 trajectory files at assessment time).

## What is missing — the open loop

1. **The injection path is literally dead (biggest gap).** `memories` is threaded from `src/onit.py` (always `None`, lines 1517/1837/2402) → `chat()` (`chat.py:3115`) → `_build_messages()` (`chat.py:1618`) — and the function body **never references it**. Nothing learned is ever fed back to the model. The loop records but does not close.

2. **No episodic recall (Loop A).** No `recall.py`, no index over trajectories, no retrieval of "what worked last time on a similar task." Estimated ~3 days: wire the dead `memories` hook, BM25/TF-IDF retrieval over the trajectory store, token-budgeted injection into the volatile half of context. **Caution from the external lens:** PAST-Bench found stored experience often does *not* help later episodes — memory on is not memory working. The RFC's A/B-on-holdout gate is the right defense; do not ship recall on intuition.

3. **No playbook (Loop B).** No `playbook.py` (itemized procedural memory with delta updates, decay, eviction), no `reflect.py` (ACE-style Reflector/Curator offline pass), no `onit learn --reflect/--promote/--rollback` CLI.

4. **No frozen holdout or pinned baseline.** `benchmarks/baselines/` contains **only a README** — no pinned baseline, no held-out suite. The external lens elevates this from "prerequisite for measurement" to **the thing that makes any gain evidence at all**: every convincing self-improvement result has a ruler the system does not own. If the agent can edit the evaluation, the score stops being evidence. This is still the single most valuable next commit.

5. **No skill synthesis (Loop C) or scaffold evolution (Loop D).** No skill mining, no serving validated skills as MCP tools, no variant generation/evaluation in worktrees, no promotion gating. These are the high-risk rungs and correctly deferred. The external lens reframes Loop D's endpoint: not "rewrite the core," but **coded extensions that integrate automatically** — see below.

6. **Weaker outcome signals.** No LLM-judge pass, no implicit-signal heuristics (rephrase ≈ failure, acceptance ≈ success). Ratings exist but are sparse — they only arrive when a human bothers. **Caution:** a judge the agent can edit gets gamed; judge prompts/schemas must sit on the human-owned side of the boundary too.

7. **No written ownership boundary for the evaluator.** The RFC's promotion gate implies humans own the ruler, but nothing states or enforces it. Needed invariant, cheap to write down and enforce: `benchmarks/`, `baselines/`, the holdout, and judge configs are **outside the agent's writable surface** (never under `data_path`, never reachable via an approval scope). This is the one thing that keeps future Loops A–D honest by construction.

## External lens: what "recursive" adds (Philschmid, Aug 21 2026)

**Two curves, not one.** A rising task score over rounds proves self-improvement. Recursion is a claim about the *other* curve: does the next round face a harder, still-honest bar? Most reported "self-improving" results (e.g. Kimi K3's harness edits moving 69→79 tasks) are bounded: the changes stuck, but nothing shows the updated system got better at finding its next change.

**Benchmarks to watch.** HarnessOpt-Bench (edit harness on dev feedback, verify on hidden tasks — gains varied sharply by model and task), PAST-Bench (memory often doesn't help), Hyperagents (closest so far, still bounded — fitness function stayed outside editable code). Useful as design checks when Loop A/B land, not as goals.

**Harness as extensions — where onit sits.** The industry is moving from "bigger prompt" to "code the model writes that the runtime loads without a human touching the core": Pi (4 built-in tools, <1k-token prompt, auto-discovered `.pi/extensions/`), Amp (project plugins), DeepSeek (Cordis kernel — models, sandboxes, even the control loop swappable; `Agent = Model + Harness`). Onit's MCP-tool architecture is already the conservative end of this spectrum: fixed core, runtime-loadable tools. That is the right end to be on — the same properties that make agent-written extensions powerful make failures persistent, and DeepSeek-style "everything is a plugin" is exactly where permission boundaries get weak. Loop D should target **agent-written MCP tools/skills that the runtime auto-discovers**, never core rewrites; onit's approval/command-policy layer is the enforcement point that makes that safe.

**Verifier ownership.** Today, everywhere: the agent owns prompts, skills, tools, memory, harness code; humans own held-out tasks and the hidden score. RSI would mean the system strengthens the verifier without capturing the signal — nobody does this, and until then, "we are doing self-improvement. Not recursion."

**Three opinions (loosely held), applied here:**
- *Near future is local recursion* — agents writing extensions the next session treats as native, trajectories becoming training data when good enough. Maps to Loops A/B first, C/D later; no architecture redesign in the dark.
- *Verification is the bottleneck* — the reason gap #4 is priority 1 and gap #7 exists.
- *Taste lives outside the loop* — agents hill-climb any verifiable metric; choosing what counts as a win stays human (a Princeton-led study found agents execute AI-research engineering well but struggle to pick original, useful directions). Promotion criteria and judge design are human-owned, permanently.

## Mapping: article claim → what it changes here

| Article claim | Implication for onit | Lands in |
|---|---|---|
| Rising score ≠ recursion; the bar must rise and stay honest | Holdout + pinned baseline is not bookkeeping, it is what makes Loop A/B results *evidence* | Gap #4, priority 1 unchanged |
| Memory often doesn't help (PAST-Bench) | Loop A ships only if it wins the A/B; recall without measured benefit is dead weight | Gap #2, exit criterion |
| Harness edits ≠ learning (Kimi K3) | Don't book retry/loop-detection fixes as "learning"; label them bounded self-improvement | Verdict |
| Extensions auto-integrate (Pi/Amp/DeepSeek) | Loop D = agent-writable MCP tools/skills with auto-discovery, not core surgery | Gap #5 reframe |
| Agent must not own the ruler | Write down and enforce the eval-ownership invariant now, before any loop can violate it | New gap #7 |
| Judges get gamed; taste is human | LLM-judge prompts/schemas on the human side of the boundary | Gap #6 caution |

## Critical path (priority order)

1. **Freeze the holdout + pin the baseline** (days) — without this, every later "improvement" is unmeasurable; with the external lens, unmeasurable *and unevidenced*. Write the ownership invariant (gap #7) in the same commit.
2. **Loop A — episodic recall** (~3 days) — smallest change that closes the loop: wire `memories`, retrieve prior trajectories, inject distilled outcomes. A/B on the holdout; ship only if it wins (the RFC's own exit criterion, and the right one — PAST-Bench says the null result is common).
3. **Loop B — playbook** (~3 weeks) — the durable, compounding memory.
4. **Loops C/D** — only after A and B show reproducible wins. Loop D targets auto-integrating extensions, never core rewrites.

The most valuable thing in the repo right now is the plan (`docs/SELF_IMPROVEMENT.md`); the most valuable next commit is the holdout, because it turns "the agent feels better" into a number — and, per the external lens, a number the agent does not own.

## Key references

- [Recursive Self-Improvement](https://www.philschmid.de/recursive-self-improvement) — Philschmid, Aug 21 2026: bounded vs. recursive, verifier ownership, harness-as-extensions
- `docs/SELF_IMPROVEMENT.md` — the RFC: loops A–D, phases, exit criteria, gap list (§3)
- `docs/archive/NOOA_onit_recommendations.md` — Aug 8 2026 NOOA capability analysis; superseded by this file, kept for its per-capability code citations. **Open item absorbed from the retired `docs/TODO.md`:** integrate [baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) for OCR support.
- `src/learn/config.py` — autonomy ladder, `_MAX_IMPLEMENTED = OBSERVE`
- `src/learn/trajectory.py`, `src/learn/report.py` — what is recorded and how it is summarized
- `src/model/serving/chat.py:1618,3115,3199` — the dead `memories` hook
- `src/onit.py:1517,1837,2402` — `memories: None` at every call site
- `benchmarks/` + `RESULTS.md` — existing fitness function; `baselines/` empty