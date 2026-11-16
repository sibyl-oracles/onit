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
| 6 | Measurement: A/B in `runner.py` — control = no wiki, treatment = wiki; replay the pinned holdout | ~2 days | `runner.py`, `eval.py` | not started — the gate before default-on |

Phases 1–5 are implemented and tested (`test_learn_memory.py`,
`test_learn_dream.py`, `test_memory_chain.py` — 102 tests).  The wiki is
**default OFF**: it activates only at autonomy `adapt` or higher, and phase 6's
A/B delta is the gate for raising the default.

Phase 6 is not optional: the article's numbers (13% → 48%, 61% vs 53%) came
from measuring, and `runner.py` exists precisely so "the agent is better" is a
delta with an error bar, not a vibe. Gate phase 4's default-on on that delta.

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
| CLI | `onit/cli.py` | `onit learn --dream` (run a pass now), `onit learn --memory` (show the wiki) |
| Config | `onit/configs/default.yaml` | `learn.memory.{path,ttl_days,prune_after_days,min_idle_episodes}` |

Tests: `onit/test/test_learn_memory.py` (31), `onit/test/test_learn_dream.py` (19),
`onit/test/test_memory_chain.py` (8) — the chain test runs dream → wiki →
inject → fast path → contradiction → resolve end to end.
