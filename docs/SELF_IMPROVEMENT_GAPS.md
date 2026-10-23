# Self-Improvement Gaps

**Status as of the Loop C commit (main, Oct 22 2026).**
Companion to `docs/SELF_IMPROVEMENT.md` (the RFC). This file is the gap assessment: what the framework already has for self-improvement, what is missing, and the critical path to close the loop.

**Updated Oct 22 2026, after the six deliverables of §5.0 were built.**  This file
described what was missing *before* the track was implemented.  Nearly everything it
lists as open is now built: the injection path, the ruler, the outcome signals, the
playbook, the lifecycle actuator, and the skill library.  The gap that is left is
not a gap in the code — **it is the measurement.**  The A/B on the holdout has not been
run, so every loop is *built and unproven*.  The table below marks each row; the
detailed per-gap prose is kept for history and is now describing a past state.

**Updated Oct 23 2026, after the test suite was built.**  The six deliverables now
have a suite that holds them: **2,915 tests pass, 2 skipped**, and the `src/learn/`
package sits at **93%** line coverage (up from 88%).  Five new files check the things
that exist only *between* the modules — the cross-cutting invariants, the signals as
properties, the whole chain end to end, the BM25 fallback, and the skill store.  That
work found and fixed **one real bug**: the skills gate's path screen matched a
forbidden directory name only when a quote came immediately before it, so
`open('/x/eval/dev.jsonl')` passed — the I7 hole, open to any skill that wanted to
read the measuring instrument.  It is fixed and regression-tested.  What the suite
does *not* do is measure the agent: it proves the scaffold is wired, capped,
reversible and unreachable-by-the-agent, and says nothing about whether any of it
makes the agent better.  See §5.1 of the RFC.

**The three things that were believed and turned out otherwise:**

1. *"The recurring bash fragments are unrecoverable from the current corpus, because
   argument values are redacted."*  Not so.  `args_digest` is the SHA-256 of the whole
   argument mapping, so the miner can find the repetition by digest and never needs
   the values.  36 candidates, 36 unique names, on the real corpus, with the redaction
   left exactly as strict.
2. *"Owner isolation is inert and recall would leak across tenants."*  It was inert, and
   it is not any more: `iter_records()` filters by owner and model at the point the
   record is produced, and every retrieval path passes an explicit filter.
3. *"The autonomy ladder is decorative."*  It was: `_MAX_IMPLEMENTED = OBSERVE` made
   every `autonomy() >= ADAPT` guard unreachable, so the recall and playbook paths
   could never run.  The cap is now `ADAPT` and it is loud.

**Updated Sep 22 2026** after the codebase-simplification pass: the dead `memories` kwarg was removed entirely (gap 1's hook no longer exists — Loop A wires it fresh), `baselines/full.json` is pinned (Sep 05, 2026), and line-number references were dropped because the refactors moved the code.

**Updated Aug 31 2026** with the framework from [Recursive Self-Improvement](https://www.philschmid.de/recursive-self-improvement) (Philschmid, Aug 21 2026) — see *External lens* below. All code claims re-verified at `f1552ca`.

**Updated again Aug 31 2026 at `a995876`:** the **event store has landed** (`f91554a`) — tool-lifecycle events (`proposed`/`loaded`/`rejected`/`archived` via `record_tool_event`, `src/learn/events.py`) now append to the same per-session JSONL as trajectories, giving Loop A memory and future tool-lifecycle decisions a single shared log. Gap numbering below predates this commit: the event store closes part of the Loop A substrate (recording half), so the remaining Loop A work is the retrieval/injection half only.

## Verdict

**Now it reads back — and it has not yet been shown to help.**

The substrate (the hardest part to retrofit) is done, and so is the feedback half:
the path from "recorded" to "changed behavior" exists in `src/learn/recall.py`,
`src/learn/playbook.py` and `src/learn/skills_*.py`, and it is reachable from a
config file at autonomy 2 and above.  What is missing is the part that would make
it a claim rather than a capability: **the A/B on the holdout.**

That distinction is the whole point of this track, so it is worth being blunt about
it: a system that can change its own behavior and has never been measured against
a frozen holdout is not a self-improving agent, it is a self-*modifying* one.  The
ruler is built.  The measurement is the next task, and it is the only task that
matters until it is done.

The external lens sharpens the target: what Loops A/B would deliver is **bounded self-improvement** — the agent changes its own prompts, memory, and tools, and keeps what works. That is worth having and is not recursion. **Recursion** additionally requires the system to face a *harder, still-honest bar* each round — i.e. the verifier improves too, without the system capturing it. No public system does this; it is explicitly out of scope here. What is in scope is not accidentally defeating the one we have.

## Already in place (verified in code)

| Piece | Status | Location |
|---|---|---|
| Trajectory recording | ✅ Done — per-turn tool calls, errors, retries, truncations, compactions, token metrics, per-session JSONL | `src/learn/trajectory.py` |
| Outcome signals | ✅ Partial — tool errors/retries are free; `POST /api/rating` (👍/👎) wired into web UI and lands in the store | `src/learn/`, `src/ui/api.py` |
| Autonomy ladder (`learn.autonomy`) | ✅ Defined 0–4, **capped at 2** — `_MAX_IMPLEMENTED = ADAPT`, the highest rung implemented — and the cap is *loud*: `requested_level()` and `capped()` say what was asked for and what was granted | `src/learn/config.py` |
| Fitness function | ✅ `benchmarks/` (inspect-ai: bigcodebench, gsm8k, humaneval, mbpp, SWE-bench) with real numbers | `benchmarks/`, `RESULTS.md` |
| Harness (Phase 0.5) | ✅ Mostly shipped — `RunState` (resume), `ResultStore` (handles), harness tools, verify pass, interpreter (default off) | `src/model/serving/` |
| Extension-friendly harness | ✅ MCP-tool-based; tools discovered/loaded at runtime (`src/type/tools.py`, `src/lib/tools.py`) — the substrate for the "harness as extensions" direction (see External lens) | `src/mcp/` |
| Recall (Loop A) | ✅ Built — BM25 over the trajectories, ranked by the outcome signal, injected in the volatile half | `src/learn/recall.py` |
| The ruler (I1/I7) | ✅ Built — holdout, dev split, pinned baseline, and `assert_outside_agent_surface()` | `src/learn/eval.py` |
| The playbook (Loop B) | ✅ Built — item store, Reflector, Curator, delta-only updates, token cap, per-session version pin | `src/learn/playbook.py` |
| The lifecycle actuator | ✅ Built — the policy now has an actuator; withholding is reversible by construction | `src/learn/actuator.py` |
| The skill library (Loop C) | ✅ Built — miner, five-gate validator, curator, and a server that serves only what the pin names | `src/learn/skills*.py` |
| Text-UI rating | ✅ Built — the `\rate` command | `src/ui/rating.py` |

Trajectory data is already accumulating in `~/.onit/learned/` — **58 trajectory files,
1,010 task records and 221 loop events** as of October 22, 2026 (Aug 4 – Oct 21), across
12 models. 820 records carry a verifier label; **none** carries a human rating, and
**none** carries an owner.

## What is missing — the open loop

**One thing, and it is not a missing module.**  Every gap this file used to list
as open is now built.  The gap that is left is the measurement:

1. **The A/B on the holdout has not been run.**  The ruler is built — `src/learn/eval.py` draws the holdout and the dev split, pins the
   baseline, and `assert_outside_agent_surface()` keeps the whole apparatus out of
   the agent's writable surface.  The loops are built.  What is missing is the run
   that says whether they help: the same tasks, `ONIT_LEARN=off` against
   `ONIT_LEARN=adapt`, scored against the pinned baseline, written to `RESULTS.md`
   whether the numbers are good or not.

   PAST-Bench's finding is that the null result is the common one for memory
   scaffolds, so the expected outcome is a null result, and a null result is a
   *result*: it goes in `RESULTS.md` and the loop does not ship.

**What is left undone inside the built parts.**  Smaller, and each is named here so
that it is not mistaken for something that is finished:

• **The gate does not run the skill's tests.**  The five checks read the source
  and screen it; running the tests is the sandbox's job (I6) and it is not wired.
  Until it is, a skill can pass the gate on inspection alone.

• **GAIA is not wired as the tool-creation benchmark.**  The RFC names it as
  Loop C's evaluation; it is registered as a task but not used to score skills.

• **The stored-prompt-injection surface is only partly treated.**  Tool output
  is still not marked untrusted in the message stream, and a learned artifact
  distilled from a fetched page is a durable instruction the agent later obeys.
  The skill *source* screen blocks the obvious escapes; it does not make the
  injection surface safe by construction.

• **The holdout is drawn after the loops exist.**  I1 is strongest when the
  holdout is frozen before the loops that could influence it.  The RFC says so.
  The honest move is to draw it now and record the draw's provenance, not to
  pretend the ordering was the easy one.

• **Loop D has not started**, and should not until Loops A and B have won.  Six
  weeks of scaffold evolution on top of an unmeasured loop is six weeks of
  measuring the wrong thing.

**A note on how this file reads now.**  This section used to be a numbered list of
missing modules, and each entry was a claim about the tree ("no `recall.py`", "no
`playbook.py`", "no frozen holdout").  Every one of those claims was true when it
was written and every one of them is false now, so the list was replaced rather
than annotated — a gap list whose every row is stale is worse than no gap
list, because it reads as a current assessment.  The reasoning behind each closure
is in `docs/SELF_IMPROVEMENT.md` (§3.0, §5.0, §8), and the
per-file claims are verifiable in the code.

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

**The critical path is now one item long, and it is not a commit.**

1. **Run the A/B.**  Draw the holdout (`onit learn --holdout`), then run the same
   tasks with `ONIT_LEARN=off` and with `ONIT_LEARN=adapt`, score both against the
   pinned baseline, and write the numbers to `RESULTS.md` — including the ones
   that are null.  Everything above this line is built; everything below it is
   downstream of the answer.

**What the path was, and what each item became.**  Kept because the order was the
argument, and the argument is what makes the order right:

| Was | Now |
|---|---|
| 1. Freeze the holdout + pin the baseline + the I1/I7 ownership invariant | **built** — `src/learn/eval.py` |
| 2. Loop A → episodic recall | **built, unmeasured** |
| 3. Implicit signals + a text-UI rating path | **built** |
| 4. Loop B → the playbook | **built, unmeasured** |
| 5. Lifecycle actuator, or delete `lifecycle.py` | **built** |
| 6. Loops C/D | **C built, unmeasured; D not started** |

The most valuable thing in the repo right now is no longer the plan — it is the
ruler, and the most valuable next action is not a commit but a benchmark run.  The
plan's own exit criterion has always been the same, and it is worth repeating
because it is the sentence that keeps the track honest: **ship only if it wins.**  A
null result goes in `RESULTS.md` and the loop does not ship.

## Key references

- [Recursive Self-Improvement](https://www.philschmid.de/recursive-self-improvement) — Philschmid, Aug 21 2026: bounded vs. recursive, verifier ownership, harness-as-extensions
- `docs/SELF_IMPROVEMENT.md` — the RFC: loops A–D, phases, exit criteria, gap list (§3)
- `docs/archive/NOOA_onit_recommendations.md` — Aug 8 2026 NOOA capability analysis; superseded by this file, kept for its per-capability code citations. **Open item absorbed from the retired `docs/TODO.md`:** integrate [baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) for OCR support.
- `src/learn/config.py` — autonomy ladder, `_MAX_IMPLEMENTED = ADAPT`, and `requested_level()` / `capped()` so the cap is loud
- `src/learn/trajectory.py`, `src/learn/report.py` — what is recorded and how it is summarized
- `src/model/serving/chat.py` — `_build_messages()` now takes the learned context; the recall rides in the volatile half and the playbook at the tail of the static half
- `src/onit.py` — `_chat_kwargs()` is now the single builder for every `chat()` caller; Loop A injects the learned context there
- `src/learn/lifecycle.py` — the tool-lifecycle policy, and `src/learn/actuator.py` — the actuator that finally acts on it
- `src/learn/trajectory.py` — `owner_hash()` and an `iter_records()` that filters by owner and model at the point the record is produced
- `benchmarks/` + `RESULTS.md` — existing fitness function; `baselines/full.json` pinned 2026-09-05. Sampled-tier rows (Sep 06 2026): simpleqa 0.940, humaneval 0.970, mbpp 0.990, on `glm-5.3-flash:cloud` with learn off. The holdout is now drawable with `onit learn --holdout` — and the A/B on it has not been run


---

*Amended October 23, 2026 (second pass) — the four blockers recorded above are
now fixed. The holdout no longer leaks into recall or reflection (158/160 → 0/160
on the real corpus), the benchmark no longer writes into the operator's corpus,
the model-based Reflector exists alongside the mechanical one, and the eval
runner exists. Three pre-existing bugs were found on the way — the split file was
written and never read back, the cluster sort counted keys instead of records,
and the runner never awaited its agent factory. `docs/SELF_IMPROVEMENT.md` §5.2
has the details. What remains unrun is the A/B itself.*
