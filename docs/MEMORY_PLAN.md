# Agent memory for OnIt — implementation plan

Based on Anthropic, "Components of a memory system for agents"
(https://www.anthropic.com/research/components-of-agent-memory), mapped onto
the existing OnIt scaffold (`onit/learn/`, `docs/SELF_IMPROVEMENT.md`).

## 1. What the article establishes

- **Memory is a hierarchy.** Top layer = injected context, loaded into every
  session, must stay small (Claude Code loads only the first ~200 lines / 25 KB
  of MEMORY.md as an *index*). Lower layers are read or searched on demand.
- **Memory is a prediction.** Writing a fact down is a bet that a future
  session needs it without re-deriving it. Store what is *not* re-derivable:
  outcomes, rules, corrections — not what can be read off the screen again.
- **Over-detailed memory rots.** Sonnet 3.7 wrote 413 files / 225 folders and
  spent turns combing them instead of acting. Fable 5 wrote 9 files + one
  `global.md` dashboard and did better. Keep the top small; push detail down.
- **Dreaming is the consolidation loop.** Run when the agent is idle. It
  validates memory against the session transcript (ground truth), corrects
  wrong notes, prunes stale ones, and builds **wikis** — linked pages that act
  as a map over memory. Measured: 13% → 48% on 25 questions as dreaming scaled
  2 → 128 sessions; wiki + search beat search alone (61% vs 53%) and saved
  ~20,000 tokens per question.
- **An uncorrected wrong note is worse than no memory.** The Pokémon run with
  the bad note looped forever; the dreaming-corrected run finished.

## 2. How this maps onto OnIt today

| Article concept | OnIt today | Gap |
|---|---|---|
| Injected top layer | Loop A recall lines (`learn/recall.py`), pinned playbook (`onit.py` `_playbook_context`) | No index header; recall is per-episode, not per-topic |
| Searched lower layer | `index_documents` / `local_search` over `documents_path` | Corpus is user documents only; agent's own findings are not in it |
| Session notes | `note_write` / `note_read` / `context_status` | Notes die with the session; nothing survives summarization |
| Dreaming | `learn/llm.py` background loop + `lifecycle.py` idle detection | No memory-consolidation job exists |
| Ground truth for validation | Session transcripts `~/.onit/sessions/*.jsonl` + `learn/trajectory.py` | Not read by any consolidation step |
| Measurement | `learn/runner.py` A/B harness with pinned holdout | Ready to use as-is |

The scaffold is already there — this plan adds a memory layer between the
session notes and the corpus, plus the dreaming job that maintains it.

## 3. Architecture: one sandbox wiki, session wikis below it

**Yes to the two-tier structure, with the roles inverted from what you might
expect.** The durable wiki is per-*topic* at sandbox level; per-session memory
stays ephemeral.

```
~/.onit/memory/
  INDEX.md                 # top layer: <200 lines, loaded every session
  wiki/                    # durable, sandbox-wide, per-topic pages
    <topic-slug>.md        # facts + rules + links, each with provenance
  sessions/<session-id>.md # per-session working memory (auto-pruned)
  dream.log                # what each dreaming run changed, and why
```

- **Sandbox wiki (`wiki/`)** — one page per subject the agent has actually
  touched (a repo, a server, a recurring question class). Pages hold verified
  facts, tested rules, and links to related pages. This is the article's
  "wiki" representation: the map that lets retrieval find things fast.
- **Session memory (`sessions/<id>.md`)** — what the current session wrote
  down (candidate facts, open questions, dead ends). Ephemeral by design:
  dreaming either promotes a fact up to the wiki (with provenance) or lets the
  file expire. This is the answer to "secondary wikis per session": yes, but
  they are *staging areas*, not wikis. Promoting everything per-session is the
  Sonnet 3.7 failure mode — hundreds of stale files that dream-time must comb.
- **`INDEX.md`** — the small injected header: what topics exist, one line each,
  with pointers. Mirrors Claude Code's 200-line MEMORY.md. Loop A injects this
  alongside its episode lines; the agent gets a `wiki_search` tool for the rest.

**Why not one giant wiki:** the article's scaling data shows wiki value grows
with dreaming rounds, but only when pages stay focused and linked. A single
page accretes and goes stale; per-topic pages can be pruned, merged, and
validated independently.

## 4. How facts consolidate during dreaming

Dreaming runs when the agent is idle — `lifecycle.episodes_since(last_used_ts)`
already computes idle episodes; the trigger is "idle > N episodes" (start with
N = 1, i.e. after every quiet period). The job runs through `learn/llm.py`'s
background loop, which already guarantees no conversation leakage.

Each dreaming pass, per session with new activity:

1. **Collect.** Read the session transcript (`trajectory.read_session`) since
   the last dream watermark. This is the ground truth.
2. **Extract candidates.** Ask the background model: what facts, rules, or
   corrections did this session establish? Each candidate must carry a citation
   to transcript lines (the article's "tag salient events as they happen" —
   extraction after the fact is the batch version).
3. **Verify.** Check each candidate against the transcript it came from — the
   Pokémon fix in miniature: the note is wrong, the transcript proves it, the
   note gets rewritten. Candidates that contradict an existing wiki page become
   *corrections*, not additions; the correction records what changed and why.
4. **Reconcile.** Merge into the target wiki page: new fact → append with
   provenance (session id, date); conflict → rewrite the page section and log
   the superseded claim to `dream.log`. Never silently overwrite.
5. **Prune and demote.** Pages untouched for K dreaming passes get their detail
   collapsed into a summary line; the detail moves down (stays searchable in
   the transcript index) and the page keeps the link. This is the article's
   "moving details down the hierarchy and leaving links behind."
6. **Rebuild the index.** Regenerate `INDEX.md` from the wiki's page titles and
   first lines. Hard cap at ~200 lines.
7. **Reindex.** Feed the wiki into the `index_documents` corpus so
   `local_search` (BM25 + dense) can retrieve it.

**Safety properties:** dreaming is append-and-log (nothing is destroyed — same
rule as `corpus.py`'s "nothing is rewritten"); every wiki claim carries
provenance so a wrong fact is traceable and correctable; the agent working on a
task and dreaming never run concurrently (idle trigger only, plus a lock file
for the edge case of a task starting mid-dream).

## 5. How this speeds up answering and verifying

- **Answering:** Loop A already injects past outcomes; the wiki adds the
  *distilled state* — the agent stops re-deriving facts it established days
  ago. The article's ~20k tokens/question saved is the target: fewer search
  round-trips, fewer re-reads of the same documents.
- **Verifying:** the fact-check loop gets a prior. Instead of verifying from
  scratch, it checks the answer against the wiki page (cheap, already
  consolidated) and only falls back to source documents when the wiki is silent
  or the answer contradicts it. A contradiction is also a dreaming input: the
  next idle pass reconciles it.

## 6. Implementation phases

| Phase | Work | Size | Touches | Status |
|---|---|---|---|---|
| 1 | Memory store: `~/.onit/memory/` layout, `INDEX.md` format, provenance schema | ~2 days | new `onit/learn/memory.py` | **done** |
| 2 | Session capture: write `sessions/<id>.md` from `note_write` calls + end-of-session distillation | ~2 days | `harness.py` note tools, `trajectory.py` | **done** (`dream.stage_session_note`, expiry in the pass) |
| 3 | Dreaming job: extract → verify → reconcile → prune → reindex, triggered by idle; **auto-creates wiki pages** (§7) | ~1 week | new `onit/learn/dream.py`, `lifecycle.py`, `llm.py` | **done** (`onit.py` `_schedule_dream` fires it; `onit learn --dream` runs it by hand) |
| 4 | Injection: `INDEX.md` into Loop A recall; `wiki_search` tool for the agent; **wiki-first fast path with provisional answers** (§8) | ~3 days | `recall.py`, `chat.py`, harness tool registry | **done** (`onit.py` `_attach_learned_context` appends the block; `harness.py` carries the tool; `run_state.wiki_hit` forces the deep check) |
| 5 | Verification hook: fact-check consults wiki first, reports contradictions back to the dream queue | ~3 days | fact-check path in harness | **done** (`verify.py` `wiki_evidence`; `chat.py` records contradictions via `memory.record_contradiction`) |
| 6 | Measurement: A/B in `runner.py` — control = no wiki, treatment = wiki; replay the pinned holdout | ~2 days | `runner.py`, `eval.py` | **done** (`onit learn --eval-run --wiki-ab`; `test_learn_gate.py`) |

Phases 1–5 are implemented and tested (`test_learn_memory.py`,
`test_learn_dream.py`, `test_memory_chain.py` — 102 tests).  The wiki is
**default OFF**: it activates only at autonomy `adapt` or higher, and phase 6's
A/B delta is the gate for raising the default.

Phase 6 is not optional: the article's numbers (13% → 48%, 61% vs 53%) came
from measuring, and `runner.py` exists precisely so "the agent is better" is a
delta with an error bar, not a vibe. Gate phase 4's default-on on that delta.

### Phase 6 as built (§6a)

The gate lives in `runner.py` next to the classic A/B, and it is a
*different* experiment: the classic arms differ in autonomy, which moves
recall, playbooks, the wiki and the dream trigger at once, so their delta
cannot say what the wiki alone is worth.  The gate's arms sit at the same
autonomy (`adapt`) and differ only in the wiki's master switch:

- **The switch.**  `learn.memory.enabled` (env `ONIT_MEMORY_ENABLED`),
  default **off**.  Every wiki gate reads it: injection
  (`onit.py` `_attach_learned_context`), the `wiki_search` tool offer
  (`chat.py`), and the dream trigger (`dream.dream_enabled`).  The CLI's
  `--dream` is a forced pass and ignores the switch, so an operator can
  build the store by hand before turning it on.
- **The freeze.**  A replay sets `min_idle_episodes` to 1,000,000: the
  dream trigger cannot fire mid-run, so the wiki cannot grow from the
  items being scored.  Deferred, not disabled — the gate's own seeding
  calls `run_dream(force=True)`, which consults neither the switch nor
  the idle count, but keeps the autonomy gate and the lock.
- **The seeding.**  A fresh store has no episodes, so a treatment seeded
  from itself would have nothing to distil.  The control runs first; its
  recorded trajectories are copied into the treatment's store and one
  forced dream pass distils them into the treatment's wiki — the same
  pass production runs, hallucination check included.  Both arms then
  recall the same corpus; the treatment additionally holds it in
  distilled form.  A win is the wiki beating raw recall on identical
  experience, which is exactly the claim the default has to survive.
- **The isolation.**  Each arm's `learn.path` *and* `learn.memory.path`
  are pinned under the arm's own directory, so an operator config that
  names a memory path cannot point both arms at one wiki.  The control's
  wiki stays empty by configuration, not by luck.
- **The verdict.**  `format_wiki_verdict` prints the delta with its error
  bar and the decision the plan ties to it: a significant non-negative
  delta raises the default; a significant negative one is the poisoning
  signal (keep it off, inspect the pages the treatment served); anything
  else keeps the default off until a bigger run says otherwise.

Run it: `onit learn --eval-run --wiki-ab [--eval-split holdout]
[--eval-limit N] [--eval-out DIR]`.  Spend the holdout only on the run
whose number will be acted on.  Without `--eval-out`, a finished run is
also copied to `tmp/eval-runs/<run-name>` in the checkout, so a report
someone asks about later survives the temp directory's cleanup.

#### The first real run, and what it fixed

The gate's first live replay (20 dev items, control arm first) looked
stuck.  It was not: it was running, minutes per item, and printing
nothing.  Three defects came out of that run, and all three are fixed in
the same commit:

- **The jail and the workdir disagreed.**  The user's config names no
  `data_path`, so `_ensure_mcp_servers` sent the tool servers to the
  default `~/sandbox` while every item ran under `<root>/data` inside
  the throwaway eval dir.  The servers' jail refused every filesystem
  tool call ("data_path must be within the server data directory"), and
  both arms would have scored on how well they described the refusals.
  `run_arm` now resolves the item workdir and stamps it into the arm
  config *before* the factory runs, so `ONIT_DATA_PATH` and the
  per-item `data_path` cannot disagree again.  This fixed the classic
  `--ab` replay too, which shares `run_arm`.
- **Silence read as a hang.**  `run_arm` printed nothing between items,
  and seeding, the grader-unavailable warning and trajectory
  inheritance were `logger.info` at an invisible level.  Progress now
  prints per item, and the seeding and grader notices print as well as
  log.  A skipped seeding -- a treatment that would run with an empty
  wiki -- is announced loudly rather than passed over.
- **The run's scratch dir is not where you look for it.**  `default_root()`
  is `tempfile.mkdtemp()`, so the arm stores live under
  `$TMPDIR/onit-evalrun-*` (macOS: `/var/folders/...`), not under
  `~/.onit` -- by design, to keep benchmark traffic out of the
  operator's corpus.  The "Replaying N items" line now says where.
- **The grader read the wrong config shape, and its failures were
  silent.**  The first completed run graded nothing: `resolve_endpoint`
  read only `serving.host`, but the config's serving block is an
  `endpoints` list, so the grader resolved to the localhost default
  (`127.0.0.1:8001`, nothing listening) and all 39 grading calls died
  on a closed port, swallowed at `debug` level.  The output was
  0.000 vs 0.000 with the verdict "no measurable effect" -- which
  reads as a null result about the wiki and is nothing of the kind.
  Fixed three ways: the resolver takes the listed entry the balancer
  would prefer when the flat `host` is absent; a failed grading call
  logs at warning and records its exception on the item, and the
  summary counts `grading_errors`; an all-ungraded run now prints
  "NOTHING WAS GRADED -- this is not a measurement" and the verdict
  "NOT MEASURED" instead of a tie.  A tie requires grades.

#### The first graded run, and what it fixed (2026-11-17)

The second run graded real items -- and produced a plausible-looking
wrong answer: control 0.20 vs treatment 0.05, delta -0.150, "no
measurable effect."  Kept copy: `tmp/eval-runs/onit-evalrun-_exp1_5k`.
Three defects came out of it, and all three are fixed in the same
commit:

- **The treatment wiki was empty -- the treatment was a placebo.**  The
  dream state said so plainly: `sessions=1 candidates=0 rejected=6
  pages_created=0`, a 0-byte `INDEX.md`, no pages directory.  Two
  causes stacked.  (1) A replay's whole day lands in one session file
  (20 items = one session id, 20 turns), and the dream pass truncated
  the transcript to its production budget of 6,000 characters -- the
  first two items.  The seeding pass now takes `max_session_chars`
  through `run_dream`/`dream_once`/`extract` and the gate passes
  200,000, so it reads the control's entire recorded experience.
  (2) The extraction call shared the grader's 400-token output cap,
  and a reasoning model spends that budget thinking before writing --
  the reply came back empty and `_extract_json` got nothing.  The cap
  is now one constant, `JUDGE_MAX_TOKENS = 4000`, for every branch.
- **Empty grader replies fell out of the comparison silently.**  A
  call that *succeeds* but returns no text is a second failure mode
  the unreachable-grader fix did not cover: 12 of 40 items ungraded
  while `grade-err=0` printed.  `grade()` now retries once and, if the
  reply is still empty, records it as a grading error so the summary
  counts it.
- **The headline delta manufactured a difference from that asymmetry.**
  Accuracy divides by every item, so an item graded in one arm and
  empty in the other counts as a miss in one and nothing in the other.
  The control's two correct answers had no graded treatment
  counterpart.  The paired view -- only items both arms graded -- told
  the real story: 12 pairs, control 2 correct vs treatment 1, one
  discordant pair, i.e. noise.  `run_wiki_ab` now reports `wiki_pages`
  and `paired` stats, and the verdict prints a PLACEBO line when the
  treatment wiki has no pages, the paired line always, and an
  incomplete-ruler note when >20% of items are ungraded.
- **The third placebo: extraction worked and the page policy starved
  it** (2026-11-17 afternoon run, `n9olwlr6`).  The seed pass
  extracted 6 verified facts from the control's whole replay day --
  and created 0 pages, because a replay day is a tour of unrelated
  tasks: clustering put every fact in its own one-fact cluster, and
  every cluster fell below `MIN_FACTS_FOR_PAGE = 3`.  The gate's
  single forced pass cannot accumulate across passes the way
  production dreaming does, so the production threshold is wrong for
  it.  The seed pass now runs with `SEED_MIN_FACTS_FOR_PAGE = 1`
  (`dream.SEED_MIN_FACTS_FOR_PAGE`, threaded through `run_dream` and
  `dream_once`; production keeps the 3-fact policy).  The pass stats
  ride into the comparison as `wiki_seed`, and the verdict names the
  zero-page cause: extraction failure (0 candidates), policy
  starvation (candidates but no pages), or a pass that never ran.
  Two smaller fixes from the same run: `extract` logs a warning when
  the model returns an empty reply (the morning run's 12 silent
  extraction drops), and the verdict flags a graded-count asymmetry
  that could account for the whole headline delta -- that run's +0.05
  was exactly 1/20, manufactured by one control grading error on an
  item the treatment graded correct; the paired line showed the truth
  (6 vs 7 correct, net zero).  "Staged only" in the dream log was
  also a lie -- `dream_once` never wrote staging files -- so the
  message now says "dropped", and `stage_session_note` remains dead
  phase-2 code with no caller.

#### The fourth placebo: the endpoint failed the extractor, once
(2026-11-17 evening run, `9c_agxy2`)

The seed pass got exactly one extraction call, the endpoint returned a
hollow reply -- the same intermittent empty-reply flakiness the grader
absorbed with its retry (the control's one grading error in the same
run reads "grading call returned no text (retried once)") -- and the
pass recorded `candidates=0, rejected=6, pages_created=0`: 0 candidates,
0 pages, and three hours of replay measuring nothing.  The diagnosis
line named extraction failure correctly; what was missing was
resilience.  Reproducing the call offline against the kept run's own
transcript returned 7 verified candidates on the first try -- the
pipeline was never broken, the attempt count was.  Fixed: `extract`
retries an empty reply up to `EXTRACT_ATTEMPTS = 3` (an exception still
aborts -- a network that is down is a different failure than a reply
that came back hollow), `dream_once` carries `empty_replies` in its
stats, and the verdict now splits the zero-candidate cause in two:
"the endpoint failed the extractor" (empty replies) wants a re-run or
a client fix, while "the control's day established no reusable facts"
(a real reply holding no items) is not a failure at all -- but the run
is still a placebo, and the verdict says so either way.

#### The hang: an unbounded request timeout stalled a run for hours
(2026-11-17 16:33 run, `xa7fqj6k`)

The fifth failure was not a placebo but a stall: the run never printed
a verdict because it never finished.  The control arm completed 15 of
20 items at a healthy pace (2-3 minutes each, trajectories clean), then
went silent at 17:15 -- the agent had finished tool call #47 (a bash
call writing a slide-deck proposal) and the next LLM call never came
back.  2h40m of nothing.  The cause was the operator's config:
``timeout: '-1'`` -- wait forever, defensible interactively where a
human sees a hang and interrupts it -- flows through
``base_config_for_run`` into ``OnIt.timeout`` and then
``chat(timeout=None)``, and the OpenAI client's default is no timeout.
A request that never returns never raises, so the balancer never fails
over (failover is error-driven), the arm never advances, and the run
stalls indefinitely.  Fixed: ``base_config_for_run`` now caps the
agent's request timeout at ``REPLAY_TIMEOUT_S = 600`` -- a replay has
no human to interrupt a stalled call, and its observed calls run under
20 seconds, so a 10-minute ceiling is generous against real work and a
floor against the hang; a shorter operator timeout is kept, a longer
or unbounded one is capped.  The same ceiling now also bounds the
grader and extractor calls.

#### The fifth placebo: the extractor's reply kept nothing, and nothing said why
(2026-11-17 21:46 run, `slmxd2lv`)

The sixth run finally measured the arms honestly -- real grades, the
paired line, the tripwire -- and was still a placebo: the treatment
wiki had 0 pages again.  The seed pass read the whole control day
(200k-char budget held), got one hollow reply, retried, and the retry
came back with a reply that parsed to **zero candidates**.  The
verdict's diagnosis said "the control's day established no reusable
facts" -- which is false: the control answered 20 items.  The stats
could not distinguish the two causes that want different fixes: the
reply held no items (format drift -- the model answered in prose), or
it held items whose citations failed the transcript check (the
cite-cleaning dropped them all).  The reply itself was never logged,
so the diagnosis was blind by construction.  Fixed: ``extract()``
returns ``parsed_items`` (what the reply held before cleaning) as a
third stat, logs the reply head when a real reply keeps nothing, and
the verdict gains a fourth diagnosis branch -- "the extractor held N
item(s) but the cite-cleaning dropped all of them" -- checked before
the empty-reply branch, because a run with one hollow reply and
dropped items is a cleaning failure, not an endpoint one.  (The run's
wall clock also spanned the Mac's overnight sleep: the treatment arm
took 10.7h for what the control did in 1h47m, with two multi-hour
gaps that match the sleep log.  A gate run should go under
``caffeinate``.)

#### The first real treatment, and what it still could not reach (2026-11-18 run, `o__cy_oz`)

The seventh run cleared every earlier failure at once: 3 pages, 5
facts, 6/6 candidates verified, no empty replies, 24.6 s of seeding --
the first treatment arm that carried wiki content.  The delta was
+0.000 (7 vs 7, 20 pairs), and the audit of the kept copy found the
measurement still not about the wiki, for two reasons the run's own
files confess:

**The treatment was never offered its own tool.**  The offer gate in
``chat.py`` called ``list_pages()`` with no argument -- the operator's
default store, empty by definition inside a gate run -- instead of the
arm's pinned store, which held the three seeded pages.  The treatment
trajectory's ``tools_available`` lists 15 tools; ``wiki_search`` is not
one of them.  The injected index digest *was* attached (that path
passes ``config_data`` correctly), so the treatment ran on the digest
alone.  Fixed: the gate is now ``_wiki_tool_available(config_data)``,
which resolves the store through the run's config, with tests pinning
the pinned-store, switched-off and low-autonomy branches.

**The arms shared one filesystem.**  Neither arm passed a
``data_path``, so both defaulted to ``<root>/data`` -- a 399 MB
workspace the control built first: its ``agents/`` clone, its pushed
commits, its ``.onit/notes``.  The treatment's answers on the three
wiki-matched items speak of the control's work as already done
("``CONFIGURATION.md`` was already created and pushed in an earlier
session"); all three scored I in *both* arms, and all six discordant
items were wiki-irrelevant.  Fixed: ``run_wiki_ab`` gives each arm
``<root>/data/<arm>`` (a caller's ``data_path`` becomes the parent),
and ``run_ab`` gets the same isolation.

**The paired line counted letter churn.**  ``discordant`` was any
grade-letter change: this run's 6 were 4 correctness flips (2 each
way) plus 2 I-to-N moves on items neither arm answered correctly.  The
gate acts on flips; ``grade_changes`` now keeps the raw movement and
the verdict prints a note when the two diverge.

#### The sixth placebo: the watermark burned the failed day (2026-11-18 run, `wzz_cnsi`)

The eighth run's seed pass got 3 hollow replies in one 76-second
window (04:07:41-04:08:57) and 0 candidates; the grader's one failure
(item `#89`, two hollow replies back-to-back) sat in the same
degradation ~5 minutes earlier.  The diagnosis named the endpoint
correctly, and the tool-offer fix proved itself: 0 of 41 treatment
records were offered `wiki_search` -- correctly, because the store
really was empty.  But the audit found a second bug under the first:
**the watermark advanced over the day the pass never processed.**  The
comment promised "the watermark only advances over what was actually
applied", yet step 7 advanced over everything in ``todo`` -- so
``state.json`` read ``watermark_ts: 04:07:39``, the day's last record,
and any later pass would see zero new records and skip the day
forever.  A failed seed did not just waste a pass; it burned the day.

Fixed in two layers:

- **`extract()` returns `failed`** -- a fourth return value marking a
  session whose extraction never got a usable reply (hollow replies
  through every attempt, or a call that raised).  ``dream_once`` only
  advances the watermark over sessions that were processed; a failed
  session is logged ("left for the next pass") and stays unread.
- **The gate retries a failed seed once.**  ``run_wiki_arm`` re-runs
  ``seed_wiki`` when the first pass reports 0 candidates with empty
  replies -- safe now that the watermark did not advance, and the
  2026-11-18 04:07 placebo is exactly what a second pass would have
  prevented.

## 7. Building the wiki automatically

The wiki is never written by the agent during a task and never written by a
human. **The dream pass is the only writer.** One writer means no merge
conflicts, no concurrent edits, and every change traceable to one idle pass.

**Creation policy** (the rule that prevents the Sonnet 3.7 rot):

- Extraction (dream step 2) clusters its candidates by topic. A cluster with
  **>= 3 verified facts** and no existing page creates one, seeded from the
  cluster, each fact with provenance. Fewer than 3 verified facts: the
  candidates stay in the session page until later passes accumulate more.
  A page is something a topic *earns*, not something a session spawns.
- **Splitting rule:** a page that grows past ~200 lines splits. The detail
  demotes into `wiki/<topic>/<sub>.md`; the parent keeps a summary line and
  the link. Depth appears exactly where a topic earned it, nowhere else.
- `INDEX.md` is regenerated every pass: page title, one-line summary,
  last-verified date. Hard cap ~200 lines.

## 8. Fast answers from memory (wiki-first serving)

Second requirement: when the fact is already on the wiki, answer now, unless
circumstances changed -- and treat that answer as temporary until the
rigorous path finishes behind it.

**The serving path on task arrival:**

1. **Lookup.** `local_search` over the indexed wiki corpus -- no model call,
   milliseconds. A hit requires all three: score above threshold, the page's
   last-verified stamp inside its TTL, and no open contradiction on the page.
2. **Scope check.** Every wiki fact carries its validity scope (observed
   date, environment, versions). Scope matches current context -> fast path
   fires. Scope mismatch ("true as of X, the env has since changed") -> the
   hit is downgraded to a *hint* the agent may see, but the fast path does
   not fire and the full path runs in the foreground.
3. **Provisional answer.** Fast-path answers go out immediately, marked
   "from memory (as of date) -- verifying", while the normal rigorous path
   (tools + verification) runs behind on the existing `background_verify`
   slot. `_FactChecker.schedule_deep_check` already delivers late
   corrections; a fast-path answer rides exactly that delivery.
4. **Confirm or correct.** Rigorous pass agrees -> the page's provenance is
   stamped re-confirmed (the fact gets stronger). Disagrees -> correction is
   delivered to the user *and* the contradiction enters the dream queue --
   phase 5's feedback loop becomes load-bearing here.

**Retrieval order within the corpus.** `wiki/*.md` and `sessions/*.md` are
searched in one `local_search` pass -- no separate general-then-session
queries. On conflict, the **session page wins**: it is fresher and may hold
a fact from earlier in this same session that dreaming has not yet
consolidated, possibly superseding the topic wiki. The conflict itself is
logged as a signal for the next dream pass to reconcile. Verified-but-older
loses to fresh-but-unconsolidated at answer time; dreaming then decides
which claim survives.

**Non-negotiables:**

- **A memory answer is a claim with provenance, never a verified answer.**
  The deep check is mandatory on every fast-path answer; the fast path buys
  latency, never skips verification.
- **Wiki text is untrusted data.** Injected with the same marker discipline
  as `recall.py`'s `UNTRUSTED_MARKER`: a poisoned page must not be able to
  steer the agent. The fast path serves facts, never page instructions.
- **Default OFF until measured.** Gated on autonomy >= ADAPT and on the
  phase 6 A/B delta, like every other loop.

## 9. Risks

- **Wrong-fact poisoning.** The article's worst outcome. Mitigations:
  provenance on every claim, transcript-grounded verification, corrections
  logged rather than silently applied, and the fact-check loop feeding
  contradictions back into dreaming.
- **Staleness.** Per-topic pages + prune-and-demote in step 5; the sandbox wiki
  holds only what is not re-derivable cheaply.
- **Token budget.** `INDEX.md` hard-capped; wiki pages searched, not loaded;
  recall injection stays distilled lines.
- **Concurrency.** Dreaming only when idle, plus a lock; a task that starts
  mid-dream waits or proceeds without the update (memory is an optimization,
  never a dependency).
- **Fast-path wrongness.** A memory answer reaches the user before
  verification. Mitigations: scope check + TTL gate before firing, the
  "from memory, verifying" banner on every provisional answer, the deep
  check mandatory behind it, and a same-session correction delivered
  through the existing late-correction channel.
- **Answer churn.** A user who gets a fast answer then a correction may
  trust neither. Mitigation: corrections state what changed and why
  (provenance), and the measured A/B gate exists precisely to kill the
  fast path if corrections outweigh the latency win.

## 10. What landed, and where to look

| Piece | File | Entry points |
|---|---|---|
| Store, gates, fast path | `onit/learn/memory.py` | `build_context` (injection), `lookup` (the three gates), `evidence_for` (checker), `apply_deltas`/`prune_page` (dream writes) |
| Dreaming pass | `onit/learn/dream.py` | `run_dream` (trigger + lock + pass), `should_dream`/`idle_episodes` (trigger), `extract` (model step), `stage_session_note` (capture) |
| Idle trigger | `onit/onit.py` | `_schedule_dream` / `_run_dream`, fired from `process_task` when the safety queue is empty |
| Injection | `onit/onit.py` | `_attach_learned_context` appends the wiki block to `run_state.learned_context`; sets `run_state.wiki_hit` |
| `wiki_search` tool | `onit/model/serving/harness.py` | `wiki_search` method, `WIKI_SEARCH_ITEM` spec, `wiki_available` gate |
| Fast-path deep check | `onit/model/serving/chat.py` | `_FactChecker.fact_check` reads `harness.wiki_hit` and runs the check even with no gathered evidence |
| Checker consult | `onit/model/serving/verify.py` | `verify_answer(wiki_evidence=...)`; wiki facts join the evidence pool, never `TRUSTED_TOOLS` |
| Contradiction loop | `chat.py` → `memory.record_contradiction` | a finding on a wiki page closes the page's fast path until the next dream pass resolves it |
| CLI | `onit/cli.py` | `onit learn --dream` (run a pass now), `onit learn --memory` (show the wiki), `onit learn --eval-run --wiki-ab` (the gate) |
| Config | `onit/configs/default.yaml` | `learn.memory.{path,enabled,ttl_days,prune_after_days,min_idle_episodes}` |
| Wiki gate (phase 6) | `onit/learn/runner.py` | `run_wiki_ab`, `run_wiki_arm`, `seed_wiki`, `format_wiki_verdict`; switch in `memory.enabled` |

Tests: `onit/test/test_learn_memory.py` (31), `onit/test/test_learn_dream.py` (19),
`onit/test/test_memory_chain.py` (8), `onit/test/test_learn_gate.py` (24) —
the chain test runs dream → wiki → inject → fast path → contradiction →
resolve end to end; the gate test runs the A/B itself with stub agents.
