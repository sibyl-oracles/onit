# Self-Improvement Gaps

**Status as of commit `c3bb028` (main, Aug 20 2026).**
Companion to `docs/SELF_IMPROVEMENT.md` (the RFC). This file is the gap assessment: what the framework already has for self-improvement, what is missing, and the critical path to close the loop.

## Verdict

**Not self-improving yet — it is a *recording* agent, not a *learning* agent.**
The substrate (the hardest part to retrofit) is done, and the plan for the rest is well-specified. What is missing is the entire feedback half: the path from "recorded" to "changed behavior." The framework is ready to *build* a self-improving agent; as of this commit it learns nothing it can act on.

## Already in place (verified in code)

| Piece | Status | Location |
|---|---|---|
| Trajectory recording | ✅ Done — per-turn tool calls, errors, retries, truncations, compactions, token metrics, per-session JSONL | `src/learn/trajectory.py` |
| Outcome signals | ✅ Partial — tool errors/retries are free; `POST /api/rating` (👍/👎) wired into web UI and lands in the store | `src/learn/`, `src/ui/api.py` |
| Autonomy ladder (`learn.autonomy`) | ✅ Defined 0–4 (off/observe/adapt/extend/evolve), but **capped at 1** — `_MAX_IMPLEMENTED = OBSERVE`; a config asking for `adapt` silently gets observe | `src/learn/config.py` |
| Fitness function | ✅ `benchmarks/` (inspect-ai: bigcodebench, gsm8k, humaneval, mbpp, SWE-bench) with real numbers | `benchmarks/`, `RESULTS.md` |
| Harness (Phase 0.5) | ✅ Mostly shipped — `RunState` (resume), `ResultStore` (handles), harness tools, verify pass, interpreter (default off) | `src/model/serving/` |

Trajectory data is already accumulating in `~/.onit/learned/` (23 trajectory files at assessment time).

## What is missing — the open loop

1. **The injection path is literally dead (biggest gap).** `memories` is threaded from `src/onit.py` (always `None`) → `chat()` → `_build_messages()` (`chat.py:1360`) — and the function body **never references it**. Nothing learned is ever fed back to the model. The loop records but does not close.

2. **No episodic recall (Loop A).** No `recall.py`, no index over trajectories, no retrieval of "what worked last time on a similar task." Estimated ~3 days: wire the dead `memories` hook, BM25/TF-IDF retrieval over the trajectory store, token-budgeted injection into the volatile half of context.

3. **No playbook (Loop B).** No `playbook.py` (itemized procedural memory with delta updates, decay, eviction), no `reflect.py` (ACE-style Reflector/Curator offline pass), no `onit learn --reflect/--promote/--rollback` CLI.

4. **No frozen holdout or pinned baseline.** `benchmarks/baselines/` contains **only a README** — no pinned baseline, no held-out suite. The plan's central gate — "every promotion must pass a frozen, held-out benchmark before it goes live" — has no implementation. This is the prerequisite for knowing whether *any* improvement loop actually helps.

5. **No skill synthesis (Loop C) or scaffold evolution (Loop D).** No skill mining, no serving validated skills as MCP tools, no variant generation/evaluation in worktrees, no promotion gating. These are the high-risk rungs and correctly deferred.

6. **Weaker outcome signals.** No LLM-judge pass, no implicit-signal heuristics (rephrase ≈ failure, acceptance ≈ success). Ratings exist but are sparse — they only arrive when a human bothers.

## Critical path (priority order)

1. **Freeze the holdout + pin the baseline** (days) — without this, every later "improvement" is unmeasurable.
2. **Loop A — episodic recall** (~3 days) — smallest change that closes the loop: wire `memories`, retrieve prior trajectories, inject distilled outcomes. A/B on the holdout; ship only if it wins (the RFC's own exit criterion, and the right one).
3. **Loop B — playbook** (~3 weeks) — the durable, compounding memory.
4. **Loops C/D** — only after A and B show reproducible wins.

The most valuable thing in the repo right now is the plan (`docs/SELF_IMPROVEMENT.md`); the most valuable next commit is the holdout, because it turns "the agent feels better" into a number.

## Key references

- `docs/SELF_IMPROVEMENT.md` — the RFC: loops A–D, phases, exit criteria, gap list (§3)
- `src/learn/config.py` — autonomy ladder, `_MAX_IMPLEMENTED = OBSERVE`
- `src/learn/trajectory.py`, `src/learn/report.py` — what is recorded and how it is summarized
- `src/model/serving/chat.py:1360,2759,2832` — the dead `memories` hook
- `src/onit.py:1486,1806,2355` — `memories: None` at every call site
- `benchmarks/` + `RESULTS.md` — existing fitness function; `baselines/` empty
