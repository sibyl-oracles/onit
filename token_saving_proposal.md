# Speed & Token Plan — `onit` codebase

**Date:** September 18, 2026 (supersedes the 2026-09-17 token-only review)
**Status: EXECUTED 2026-09-18** — S1–S8, Tier 5.1–5.3, and §7 A1–A3/B1–B3 are
implemented and tested (see "Executed" markers per section; S9 deliberately
deferred, it needs an A/B). Suite: 2255 passed, 4 skipped, 1 pre-existing env
failure (`test_host2_enables_load_balancing`, fails on a clean tree too).
New tests: `src/test/test_speed_plan.py` (33 tests).
**Goal:** speed up task completion (wall time per task) **and** keep optimizing token usage. The two share most of the same levers: fewer turns, smaller prompts, cheaper compaction, and a server that prefills less.
**Scope:** `src/` (agent loop, result store, harness, prompt builder, MCP servers, `onit.py`), `benchmarks/` (measurement), `docs/RUN_A_MODEL_SERVER.md` (serving flags).
**Method:** code-traced, like the 2026-09-17 review. Wall-time model: a run costs
`Σ_turns(prefill + decode) + Σ tool time + compaction LLM calls + verify fast pass + instruction build`.
Every proposal below names which term it attacks.

---

## 0. Status of the 2026-09-17 plan (commit `4d9100f`)

Tiers 1–4 are **executed and verified** (see `.onit/notes/token_review_findings.md`):

| What | Before → after | Effect |
|---|---|---|
| Tool descriptions | 25.8k → 20.1k chars | ~−1.4k tok/request |
| `DEFAULT_MAX_TOKENS` | 131072 → 16384 | compaction threshold 0.50 → **0.89**; ~half the compaction LLM calls |
| `parameters` echo in tool msgs | 7 sites → 1 (empty-buffer fallback only) | −200–800c per historical call per turn |
| `indent=2` in tool-facing dumps | ~30 sites | −~1k tok on JSON-heavy calls |
| Decay dials | KEEP_FULL 3→2, DECAY 6000→4000, STORE_THRESHOLD 8000→5000, HISTORY_DECAY 1200→800 | −~1k tok mid-run |
| Compaction transcript | unbounded → 60k chars / 150 msgs / 30 mentions | compaction can no longer fail because context is full |
| Producer caps | bash 100k→32k, fetch 50k→24k, grep lines 300c, log tail 8k | −1–2k tok on the turns that hit them |
| `think_tool_turns` | default True → **False** | thinking only on turn 1 |
| Revision budget | uncapped → min(max_tokens, 8192) | bounded verify rewrite |

Standing payload: ~9k → ~6.5k tok/request. Mid-run saving ~4.5–6.5k tok/request (~30–40% of the non-schema prompt). Tests: 2234 passed, 4 skipped.

**One regression found in this review — the Tier-3 history cut is not actually live:**

### BUG: `src/configs/default.yaml:183` still ships `history_turns: 10`

The code default was cut 10 → 6 (`onit.py:889` Field, `onit.py:1350` fallback), but the shipped
default config file still says `history_turns: 10`, and a config value **overrides** the Field
default. Anyone running with the shipped config gets the old depth. One-line fix:

```yaml
history_turns: 6
```

Cost: ~1k tok/request on every session with ≥6 prior turns. Do this first — it is the cheapest
remaining token win in the repo.

---

## 1. Where the wall time goes (measured from the code)

| Term | What drives it | Where it lives |
|---|---|---|
| Prefill | prompt size × cache hit rate | every turn; `TurnMetrics.prefill_s` |
| Decode | output tokens ÷ server decode rate | `TurnMetrics.decode_s`; reasoning models emit full reasoning before every tool-call JSON |
| Tool time | network round trips, serial per batch unless read-only | `_execute_tools_in_parallel` covers 13 read-only tools only |
| Compaction | one extra LLM call per compaction event | `_compact_context`; now fires at 89% |
| Verify fast pass | one verdict call, 512 tok budget, 2 s timeout | `_fact_check`; runs on every ≥80-char answer |
| Instruction build | ~0 in-process (`prompt_in_process: true`), timed as `instruction_s` | `onit.py:1515` |
| Session history load | reads the **entire** session JSONL on every task | `load_session_history`, `onit.py:1390` |

The decode term dominates for reasoning models; the prefill term dominates when prefix caching
misses; the tool term dominates for research tasks with many serial calls.

---

## 2. Remaining token work (Tier 5, unchanged from the 09-17 proposal)

> **Executed 2026-09-18.** 5.1 as `TOOL_RESPONSE_BUDGETS` + `_truncate_tool_response(response, tool)`
> (chat.py); 5.2 as the prefix-cache contract comment in `_build_messages` plus
> `TestRequestPrefixByteStable`; 5.3 as URL extraction in `ResultStore.stored()`
> (fetch-shaped results carry their URL into the `context_status` list).

**5.1 Per-tool result budgets.** `MAX_TOOL_RESPONSE` (16,000) is one number for all tools. A
`{tool: max_chars}` table in the truncate/store path lets cheap tools stay cheap.

**5.2 Prefix-cache contract, documented and tested.** The static instruction half
(`INSTRUCTION_SPLIT`, `split_instruction`) and the tool payload are byte-stable by design;
nothing verifies it. Add a test that the first N bytes of the request are identical across two
turns of the same run, and a comment in `_build_messages` warning that reordering keys or
blocks breaks the server's prefix cache.

**5.3 `fetch_content` URL → handle dedup.** Surface the URL→handle mapping in
`context_status`'s stored list so a re-fetch of the same URL is visibly wasteful; one nudge
line in `result_block`.

---

## 3. Speed work, ranked by impact ÷ effort

> **Executed 2026-09-18**, except S9 (needs an A/B; deferred). Per-item notes:
> S1 — provider publishes per-sample metadata (`wall_s`, `model_s`,
> `prefill_s`, `decode_s`, `tool_s`, `ttft_s`, `prompt_tokens_max`,
> `completion_tokens`, `cached_tokens`, `compactions`, `turn_count`,
> `retries`); `report.py` aggregates means + TTFT p50/p95 + token-weighted
> `cache_hit_pct` and renders the speed table when logs carry it.
> S2 — `default.yaml` now ships 6. S3 — `TurnMetrics.end_api(cached_tokens=...)`
> via `_cached_tokens_of` (`usage.prompt_tokens_details.cached_tokens`; None,
> never 0, when the path does not report it); `summarize_metrics` prints the
> run's hit rate. S4 — incremental compaction: `RunState.compaction_summary` /
> `compaction_summarized_upto`; `_compact_context` summarizes only the delta
> and merges into the prior summary; the cursor rides the compacted message as
> a `_compaction` marker that `_strip_internal_keys` pops before every request;
> `merge()` resets it per run. S5 — doc updated. S6 — `has_tool_evidence`
> gates the fast pass. S7 — `load_session_history` reads 64KB chunks from the
> end. S8 — the four harness read tools joined `_READ_ONLY_TOOLS`.

### S1 — Measure first: wall-time and token columns in the benchmark report *(hours)*

`benchmarks/report.py` collects **accuracy only**. No proposal below can be judged without
wall time and tokens next to it. Extend `collect()`/`to_markdown()` to read each sample's
`stats["metrics"]` (already produced by `process_task`: `instruction_s`, `model_s`, `prefill_s`,
`decode_s`, `tool_s`, `prompt_tokens_max`, `completion_tokens`, `compactions`) and emit:

| Benchmark | Accuracy | Wall s/sample | Prompt tok (peak) | Out tok | Compactions |
|---|---|---|---|---|---|

Also record `ttft_s` p50/p95 — it is the user-perceived latency and it separates prefill
problems from decode problems. This is the plan's gate: every S-item lands only if it moves
these columns on the `sampled` tier without moving accuracy beyond the 0.05 gate.

### S2 — Fix the `history_turns` default *(minutes — see the bug above)*

Fewer replayed pairs is both a token and a prefill win on every multi-task session.

### S3 — Track prefix-cache hit rate; keep the request byte-stable *(a day)*

Nothing measures whether the cacheable prefix is actually being cached. vLLM reports
`prompt_tokens_details.cached_tokens` on every response; `TurnMetrics.end_api` drops it.

- Capture `cached_tokens` per turn alongside `prompt_tokens`; add `cache_hit_pct` to
  `summarize_metrics` and to the S1 report columns.
- Assert in a test that the serialized request prefix (system message + tool payload) is
  byte-identical across turns — this is the property the whole instruction-split design buys.
- If hit rate is low on a real run, the usual culprits are: a volatile block that drifted into
  the static half, a tool payload rebuilt in a different order, or a provider that ignores
  `chat_template_kwargs` ordering. Measure before touching anything.

Why this is speed work: at ~6.5k tok standing payload, a warm cache turns most of the prefill
into a no-op; a cold cache re-pays it on every turn of every task.

### S4 — Cheaper compaction: incremental summary instead of full re-summarize *(1–2 days)*

`_compact_context` re-summarizes the whole transcript every time it fires. At the 89%
threshold that is less frequent but still one full LLM call over up to 60k chars. Keep a
running summary in `RunState` (it already persists via `state.py`) and compact by
summarizing only the *new* messages since the last compaction, merging into the prior
summary. Cuts the compaction call's input by ~k× and its latency with it. The transcript
caps (60k/150/30) stay as the bound on the incremental step.

### S5 — Speculative decoding on the vLLM path *(config + doc, a day)*

The decode term is the largest single term for reasoning models. vLLM supports speculative
decoding (`--speculative-model` / ngram or draft-model based); on the Qwen3.8-27B serving
path in `docs/RUN_A_MODEL_SERVER.md` this is a pure latency win at unchanged output. Add the
flag pair to the doc next to `--enable-prefix-caching`, with the caveat that it needs a draft
model or ngram spec and helps most on the repetitive tool-call JSON that dominates tool turns.

### S6 — Skip the verify fast pass when there is nothing to check *(hours)*

`_fact_check` runs on every answer ≥80 chars. When the run gathered **no** evidence
(`evidence_digest` returns the "nothing was gathered" stub) and the answer is short of
claims, the verdict call is a pure latency add on the critical path — the user waits ~1–2 s
for a check that can only re-report the model's own knowledge. Gate the fast pass on
evidence actually existing; keep the background deep check as is. This removes a
serial LLM call from the tail of every knowledge-answer turn.

### S7 — Tail-read the session file *(hours)*

`load_session_history` parses the whole JSONL to return the last 6 pairs. Sessions grow
unboundedly; a task late in a long session pays a growing file read before the first token.
Read the file backwards (chunked from the end) or keep a byte offset of the last N records
in the session state. Pure startup latency, no token change.

### S8 — Widen the parallel batch to the harness tools *(hours)*

`_READ_ONLY_TOOLS` covers the MCP read tools but not `context_status`, `note_read`,
`result_read`, `result_grep` — all local, all safe to run concurrently. Add them so a model
that batches a read with a status check does not serialize on them. Tiny, but free.

### S9 — Continuation budgets: cap `think` on tool turns at the provider, not the prompt *(a day, needs A/B)*

`think_tool_turns: false` already stops reasoning on later turns for models that honor
`enable_thinking: false` (the `_no_think` path at chat.py:4220). For hosts that cannot switch
thinking off (`_template_kwargs_unsupported`), every tool turn still pays full reasoning.
For those hosts, cap `CONTINUATION_MAX_TOKENS` at a verdict-sized budget on tool-only turns
and measure accuracy on `sampled` before/after. If accuracy holds, this is the biggest
decode-term cut available on those deployments.

---

## 4. What NOT to change

- **`_trim_history` keeping tasks whole** — tasks are one line each; they are what makes follow-ups intelligible.
- **`think` on the opening turn** — the plan is decided there; that is the highest-value thinking in the run.
- **The compaction threshold at 0.89** — the 09-17 analysis holds: fewer, bigger lossy compactions beat more, smaller ones (survival odds pᵏ, and k halves).
- **`RESULT_PREVIEW_CHARS = 6000`** and the documents block — evidence caps stay.
- **`MAX_STDOUT_CHARS = 400k`** in the interpreter — `run_code` output goes through the store's handle path; the cap is a ceiling, not a cost.
- **Instruction-split / prefix-cache structure** — S3 measures it; nothing restructures it.

---

## 5. Expected effect

| Change | Wall time | Tokens | Effort |
|---|---|---|---|
| S1 measurement columns | — (enables the rest) | — | hours |
| S2 history_turns default fix | small (prefill) | ~−1k tok/request | minutes |
| S3 cache-hit tracking + stability test | up to ~large on cold-cache deployments (prefill) | 0 (measurement) | a day |
| S4 incremental compaction | −compaction call latency (seconds per event) | −input tokens of the compaction call | 1–2 days |
| S5 speculative decoding | −decode time, potentially 1.5–2× on tool-heavy runs | unchanged | a day |
| S6 skip verify fast pass w/o evidence | −1–2 s per knowledge-answer turn | −1 verdict call | hours |
| S7 tail-read session file | −startup on long sessions | 0 | hours |
| S8 parallel harness tools | −serialization on batched turns | 0 | hours |
| Tier 5.1–5.3 | small | −1–3k tok on the turns that hit them | a day |

Combined realistic target on the `sampled` tier: **−20–40% wall time per sample at flat
accuracy**, on top of the ~30–40% prompt-token reduction already landed in `4d9100f`.

---

## 6. Acceptance checks

1. `pytest src/test/test_chat_metrics.py src/test/test_result_store.py src/test/test_chat.py` still green after S2/S6/S8.
2. S1: `python -m benchmarks.report --log-dir benchmarks/logs/sampled` prints the new columns; a `sampled`-tier run before/after each speed change lands in `RESULTS.md` with wall s/sample and prompt tokens.
3. S3: a test asserts byte-identical request prefix across two turns; a real run shows `cache_hit_pct` > 0 on vLLM.
4. S4: compaction test asserts the incremental path produces a summary bounded by the same transcript caps and preserves the verbatim instruction restatement.
5. S6: a no-evidence answer returns without a verdict call (metrics `verify_s == 0`), and a run with evidence still verifies.
6. The regression gate (`report.py --baseline`, tolerance 0.05) must not trip on any of these.
---

## 7. Timeouts and "I am sorry" bail-outs (added 2026-09-18)

> **Executed 2026-09-18.** A1 — `_build_client_timeout(timeout, stream, prompt_tokens)`:
> the first read gets `max(300, min(600, prompt_tokens/300))` seconds (queue +
> prefill), per-chunk gaps stay at 300; `chat()` sizes it from the standing
> payload (`chars/4 + tools×48`). A2 — timeout retries capped at 1
> (`_MAX_TIMEOUT_RETRIES`); the retry goes out with older tool results decayed
> to their heads (`_decay_old_tool_results(messages, keep_full=1)`, which now
> returns the trim count). A3 — `ONIT_BENCH_TIMEOUT` default 300→600.
> B1/B3 — the repeated-call guard now counts a consecutive streak: at streak ≥3
> (or lifetime ≥ `max_repeated`) a steering notice is appended to the tool
> message and the loop continues; the turn only ends at streak ≥5 or lifetime
> ≥ 2×max_repeated, and the bail message names the loop and the way out
> instead of the apology. B2 — the notice points at `result_read`/`result_grep`
> for trimmed content. C — nothing was reverted: 16384, the 0.89 threshold and
> the decay budgets all stand.

Post-`4d9100f` runs time out more often, and exhausted API retries surface as
`"I am sorry. Could you try to rephrase or provide additional details?"`
(`chat.py:1898` — actually the repeated-tool-call bail; the API-retry path
returns `None`). Two independent mechanisms, both fixable without giving back
the token savings.

### A. Timeout side

- **A1 — Scale the first-read budget with prompt size.** `_build_client_timeout`
  (`chat.py:95-110`) applies `STREAM_STALL_TIMEOUT=300` to the *first* read,
  which includes queue + prefill. Prompts now grow to ~0.89 of the window before
  compaction, so queue+prefill alone can cross 300s on a loaded server. Fix:
  `read = max(STREAM_STALL_TIMEOUT, prompt_tokens / 300)` seconds (≈300 tok/s
  conservative prefill), or a flat 600s first-read allowance. Inter-chunk gaps
  stay at 300s.
- **A2 — Never retry a timeout with the identical prompt.** The
  `APITimeoutError` handler (`chat.py:4932-4946`) only drops
  `tool_choice="required"`; the retry re-sends the same oversized prompt into
  the same loaded server up to `MAX_API_RETRIES=3` times (3×300s of stall).
  Fix: cap timeout-caused retries at 1, and before the retry apply an emergency
  trim (oldest tool results → heads) so the retry is *smaller*, not identical.
- **A3 — Scale `ONIT_BENCH_TIMEOUT`** (`benchmarks/config.py:252-261`, 300s
  total per request) with prompt size or raise to 600s for large-window models;
  the sample limit is 4× this (`benchmarks/run.py:172-175`).

### B. Bail-out side

- **B1 — Make the repeated-call trip a steering message, not a turn
  terminator.** `_execute_tool` (`chat.py:1895-1901`) returns the apology and
  the caller ends the turn (`chat.py:5448-5451`). The model never sees it, so
  it cannot correct. Fix: append a tool message instead — "this exact call was
  made N times and returned the same result; do not call it again, use the
  result you have or change approach" — and continue the loop. Keep the hard
  bail as a second-level guard at 2× the limit.
- **B2 — Break the decay↔re-call tie.** The decay trailer instructs "call the
  tool again for the rest" (`chat.py:1226-1237`, marker at `:1300`), while the
  repeated-call guard punishes exactly that. Handle-bearing results must point
  at `result_read` (local, free) instead of a re-call; after the 2nd identical
  call, inject the handle line into the tool message.
- **B3 — Count streaks, not lifetime totals.** `tool_call_history` persists
  across tasks (`state.py:186-188`, window 200) and `count(call_key) >= 30`
  is over that whole window. A legitimate repeated read pattern can trip it.
  Count *consecutive* identical calls (streak ≥ 5 = loop) instead.

### C. What not to undo

Keep `DEFAULT_MAX_TOKENS=16384` and the 0.89 threshold (reverting to 131072
re-creates the 0.50 clamp and adds a window-overflow retry loop on 128k–256k
windows — the clamp's 8k growth buffer is smaller than a 16k-char tool
result). Keep the decay dials; fix the trailer semantics (B2), not the caps.
The timeouts are a timeout-architecture and retry-policy problem, not a token
budget problem.

### D. Order of attack (impact ÷ effort)

1. B1 steering message — hours, removes most apologies at the source.
2. A2 timeout retry cap + shrink — hours, converts 3×300s stalls into ≤1.
3. A1 first-read budget — hours–day, kills the tail-timeout spike.
4. B3 streak counter — hours.
5. B2 trailer semantics — a day, touches decay tests.
6. A3 bench timeout scaling — minutes.

Gate: S1 columns (§3) before/after on `sampled`; timeouts/sample and
apology-bails/sample both to zero, prompt tokens flat vs `4d9100f`.
