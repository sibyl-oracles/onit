# Self-Improvement Quick Start

**How to show that OnIt actually gets better, starting from nothing.**

**Date**: October 24, 2026 · **Status**: the machinery is built and the
measurement works. The loop has **not** been shown to win, and this
guide is written so that it cannot be mistaken for a win.

> **The headline, measured.** A full run of this guide on this machine
> produced baseline 6/8, treatment 8/8, delta +0.250 (stderr 0.153, which
> the code called significant) — and then the controls destroyed it. The
> null arm (same injection, empty store) scored **8/8**. The isolated
> null arm (empty store, nothing injected at all) scored **8/8**. The
> same arm, repeated three times, scored **6, 7, 7**. The agent's date
> arithmetic is not deterministic, and a two-item difference on eight
> items is **sampling noise**. The treatment arm's own leak check
> reported **8 of 8 items contaminated**. And the 0.153 error bar that
> made the delta look significant was the *baseline's* error alone — the
> treatment arm's was zero, because it scored 8/8. See §10 for the full
> accounting.

**Companions**: [`SELF_IMPROVEMENT.md`](SELF_IMPROVEMENT.md) — the design and
the standing gap assessment. [`SELF_IMPROVEMENT_GAPS.md`](SELF_IMPROVEMENT_GAPS.md)
— what is built and what is not.

---

## 0. The claim, stated so it can fail

"OnIt improves over time" is **three** separate claims. A demo that shows
only the first two proves nothing, and most demos stop at two.

| # | Claim | What proves it | What it does *not* prove |
|---|---|---|---|
| 1 | The run was **recorded** | a JSONL line appears with the task and answer | anything about quality |
| 2 | The next run was **shown** the record | `learned_context.episodes_used` is non-empty | that being shown helped |
| 3 | Being shown it **changed the outcome** on tasks the loop never saw | a delta on a frozen holdout, with a control | — |

This guide walks all three. Act 6 is the only one that can carry the
claim, and it is built so that Act 6 can come out **zero**.

### What is true right now, measured

| Fact | Number | Where it came from |
|---|---:|---|
| Autonomy level the code implements | `adapt` (2) | `src/learn/config.py:41-62` |
| Recorded task records on this machine | 1,022 | `onit learn --json` |
| Records carrying `episodes_used` | **0** | every `learned_context` in the store |
| Records carrying a `playbook_version` | **0** | same |
| Corpus questions that are duplicated | 894 distinct over 2,044 records | `eval.task_key()` |
| Holdout items whose own question sits in the recall pool | 160 / 160 | `eval.holdout_exclusions()` |

The zeros are the point of this guide. The loops are wired and reachable;
what has never happened is a run that was *shown* anything, and a
measurement that said whether that helped.

---

## 1. Setup — a store of its own

Never measure on your working corpus. The demo gets its own store, its own
data path, and its own config.

```yaml
# ~/demo/config.yaml
serving:
  host: http://<your-endpoint>/v1     # a HOST, not an endpoints list
  model: <your-model>
learn:
  autonomy: adapt                    # observe records but never injects
  path: ~/demo/store                 # this demo's own store
data_path: <repo>/tmp/onit/data/demo # MUST be inside the bash server's DATA_PATH
session_path: ~/demo/sessions
```

Three of those lines are load-bearing, and each one has bitten:

| Line | Why it is load-bearing |
|---|---|
| `serving.host` | the grader and the Reflector read `serving.host` **only** (`src/learn/llm.py:55`). A config with an `endpoints` list resolves to `127.0.0.1:8001` and every background call dies silently. |
| `learn.autonomy: adapt` | at the default `observe`, the agent records and **never injects** (`recording_enabled()` is `autonomy >= OBSERVE`; the injection paths gate on `>= ADAPT`). |
| `data_path` | the bash tool refuses any `data_path` outside the server's `DATA_PATH` (`src/mcp/servers/tasks/os/bash/mcp_server.py:186-206`). A path outside the jail means **every tool call is refused** and the agent measures refusals. |

Check the jail before you start:

```bash
python3 - <<'PY'
import sys; sys.path.insert(0, ".")
from src.mcp.servers.tasks.os.bash.mcp_server import _session_base, DATA_PATH
print("server jail:", DATA_PATH)
try:
    print("OK   ->", _session_base("<repo>/tmp/onit/data/demo"))
except ValueError as e:
    print("REFUSED ->", e)
PY
```

---

## 2. The task family — ground truth, not a judge

**Do not grade with a model.** Use tasks whose answers are checkable by
string match, so the measurement has one model in it (the agent) and not
two, and no judge endpoint to be down.

**Family**: `Compute the weekday of the following date: <YYYY-MM-DD>`.

Why this family works:

- **Recall has something to transfer.** The first run's answer contains the
  *method* (a worked calculation), not just the answer.
- **The answer is not transferable.** Date B's weekday is not in date A's
  answer, so a win cannot come from copying.
- **No tools.** The bash jail never enters the picture.

### The two phrasings that decide whether the loop can learn

This is the part most demos get wrong, and it is worth a whole section.

**(a) Keep the date out of the first three content words.** The Reflector
groups runs by `task_shape()` — the first three content words of the task
(`src/learn/playbook.py:638-657`). Measured:

| Task text | Shape | Cluster |
|---|---|---|
| `What weekday is 2027-02-14? Answer with the weekday name only.` | `weekday\|2027\|answer` | alone |
| `What weekday is 2028-11-03? Answer with the weekday name only.` | `weekday\|2028\|answer` | alone |
| `Compute the weekday of the following date: 2027-02-14` | `compute\|weekday\|following` | **with the next** |
| `Compute the weekday of the following date: 2028-11-03` | `compute\|weekday\|following` | **with the first** |

The first phrasing puts the date in the third content word, so **every run
is a cluster of one** and no contrast is ever possible. The second puts
the date last, so the whole family shares a shape.

**(b) A contrast needs two sessions.** A run is labelled:

| The run is… | Label | Source |
|---|---|---|
| followed by a **rephrase** of it in the same session | `bad` | `signals.label_pair()` |
| the **last** run of its session | `good` | `signals.label_session()` — "nobody objected" |
| neither | `unknown` | — |

So a **single** session cannot hold both sides: every run but the last is
labelled by its follower, and only the last is `good`. The shape needs
two sessions:

```
session 1:  [date A]  →  ["no, I meant the day of the week, not the date"]
               A is labelled bad (its follower is a rephrase)

session 2:  [date B]
               B is labelled good (session end, nobody objected)

same shape, opposite labels: a contrast
```

Measured on the real code: `is_rephrase(A, B)` is `False` (similarity
0.571, below the threshold) and `is_rephrase(B, "no, I meant…")` is `True`
(the "no" cue). So the pushback has to be a **rephrase of the previous
task**, and a second date is *not* a rephrase — it is a new task, and it
leaves the previous run `unknown`.

---

## 3. The holdout — authored, not drawn

`onit learn --holdout` draws the holdout **from the recorded corpus**. That
is the right instrument for a corpus that already exists, and the wrong
one for a first measurement: on day 0 there is no corpus, so there is
nothing to draw from, and a holdout drawn from the very runs the agent is
about to make measures nothing.

Author the holdout instead. It was never in the store, so it has no leak
to fix.

```jsonl
{"task": "Compute the weekday of the following date: 2027-02-14", "response": "Sunday"}
{"task": "Compute the weekday of the following date: 2028-11-03", "response": "Friday"}
```

`response` is the reference. Alternatives separated by `|` are all
accepted. An item with no reference is recorded as `ungraded` rather than
counted wrong — a missing measurement that counted as a failure would make
the arms look different for reasons that have nothing to do with the agent.

**Split, authored before anything runs:**

| File | Items | Used for |
|---|---|---|
| `~/demo/loop.txt` | 20 dates | the agent's ordinary work — this is what it learns from |
| `~/demo/holdout.jsonl` | 40 dates | measured, never recorded, never reflected on |

**Never let the holdout's questions into the loop's questions.** The
holdout is measured; the loop is what learns.

---

## 4. Act 0 — freeze the baseline, before the agent has ever run

```bash
python3 scripts/measure.py --config ~/demo/config.yaml \
    --tasks ~/demo/holdout.jsonl --autonomy off --out ~/demo/baseline.json
```

`--autonomy off` is a hermetic arm: `recording_enabled()` is
`autonomy >= OBSERVE`, so `off` neither injects nor records. The baseline
run leaves **no trace in the store** — verified: after a 5-item baseline
the store's `trajectories/` directory does not exist.

Measured on this machine with 5 items:

```
  [  1/  5] correct     What weekday is 2027-02-14? ...
  [  2/  5] correct     What weekday is 2028-11-03? ...
  [  3/  5] incorrect   What weekday is 2030-06-09? ...
  [  4/  5] correct     What weekday is 2029-12-25? ...
  [  5/  5] correct     What weekday is 2031-07-04? ...

  n=5  graded=5  correct=4  wrong=1  not-attempted=0  ungraded=0  errors=0  accuracy=0.800
```

That `0.800` is the only baseline you are allowed to compare against
later. Write it down. **Do not re-run the baseline later and use the new
number** — that is how a null result becomes a win.

---

## 5. Act 1–2 — a cold run, then recall fires

```bash
onit --config ~/demo/config.yaml
```

```
> Compute the weekday of the following date: 2027-02-14
```

Then look at what was written, **not** at the answer:

```bash
python3 - <<'PY'
import json, glob, os
for f in sorted(glob.glob(os.path.expanduser("~/demo/store/trajectories/*.jsonl"))):
    for line in open(f):
        r = json.loads(line)
        if r.get("kind") not in (None, "task"):
            continue
        lc = r.get("learned_context") or {}
        print("episodes_used=%-24s playbook_version=%s | %s"
              % (lc.get("episodes_used"), lc.get("playbook_version"),
                 str(r.get("task"))[:50]))
PY
```

```
episodes_used=[]                     playbook_version=0 | Compute the weekday of the following date: 2027-02-14
```

`episodes_used: []` is the point. **Claim 1 demonstrated, claim 2 not.**

Now ask a *second* date in the same family:

```
> Compute the weekday of the following date: 2028-11-03
```

```
episodes_used=[]                     playbook_version=0 | Compute the weekday of the following date: 2027-02-14
episodes_used=['7387c19d-3301-4fe6…'] playbook_version=0 | Compute the weekday of the following date: 2028-11-03
```

The second run was handed the first run's worked calculation, ahead of the
question, marked as untrusted history. **Claim 2 demonstrated.** Still
nothing about quality.

> The recall block is capped: `top_k=3` episodes, an 800-token budget
> (`src/learn/recall.py:71,78`), and each episode is one line — the task,
> a 220-character excerpt of the answer, and the outcome
> (`_summarize_episode()`). The outcome is the part that transfers.

---

## 6. Act 3 — manufacture a contrast

The Reflector learns from a *contrast*: one run that went well, one that
did not. The reward is **not** a score you type in — it is the **next
turn's behaviour**, read by `signals.label_records()`.

| Turn 2 looks like | Label |
|---|---|
| a rephrase of the same question ("no, I meant…") | **bad** |
| a new task, then the session ends | **good** |

So the demo does what a real user does — it reacts. And, per §2(b), it
reacts **in a second session**, so both sides of the contrast exist:

```
session 1                          session 2
> Compute the weekday of …: 2027-02-14
                                  > Compute the weekday of …: 2028-11-03
> no, I meant the day of the
  week, not the date
```

Check the labels before you reflect:

```bash
onit --config ~/demo/config.yaml learn --signals --write-signals
```

```
Implicit signals
  records      8 across 3 session(s)
  labeled      5 (62.5% of the corpus)
  good / bad   3 / 2
  unlabeled    3
```

`--write-signals` is **not optional** here, and the reason is a real gap:
`onit learn --reflect` calls `label_corpus()`, which is documented as
"pure with respect to the store: nothing is written", and then re-reads
the records from disk (`src/cli.py:409-412`). The labels it computed are
thrown away, and the records the Reflector sees carry no
`signals.implicit`.

Measured, with `labels.jsonl` moved aside and put back:

| | Records read from disk | Carrying an implicit label | Contrast visible |
|---|---:|---:|---|
| without `labels.jsonl` | 4 | 0 | **none — no cluster has both sides** |
| after `--write-signals` | 12 | 8 | `compute\|weekday\|following` good=3 bad=2 |

So without the `--write-signals` step first, **the contrast is invisible
to the Reflector**, and `--reflect` reports "no contrast worth a bullet"
— a conclusion, not an error.

> A second consequence of `--write-signals`: it writes `labels.jsonl` into
> the trajectory directory, and `iter_records()` walks every `*.jsonl`
> there. The label rows have no `response`, so they are recalled as
> episodes with an empty answer. Measured on the real store: 1,022 of the
> 2,044 records read back carry no `response` — exactly the label file.
> On this machine the store grew from 1,022 to 2,044 task records after
> one `--write-signals`.

---

## 7. Act 4 — reflect, offline

```bash
onit --config ~/demo/config.yaml learn --reflect --dry-run   # see, change nothing
onit --config ~/demo/config.yaml learn --reflect
onit --config ~/demo/config.yaml learn --playbook
```

```
Playbook v1 -- 1 bullet(s)

   Apply a deterministic weekday algorithm such as Zeller's congruence, then sanity-check against a known nearby date.
      trigger: weekday computation   used 0, helped 0, hurt 0
```

Two reflectors run, and the difference matters:

- the **mechanical** one contrasts the *tools* the good runs used against
  the tools the bad runs used. A task answered with no tools has nothing
  to contrast, so it produces **no bullet** — by construction.
- the **model-based** one reads the two runs and names what the failing
  one did differently. This is the half that can say something about a
  no-tool task.

Use `--mechanical-only` / `--model-only` to see which one produced a
bullet.

### A defect worth knowing about before you trust a silent Reflector

`src/learn/llm.py:126` sets `num_predict: JUDGE_MAX_TOKENS` (64) for
Ollama and `max_tokens: 400` for OpenAI-shaped hosts. On a **thinking**
model, 400 tokens can be entirely consumed by the reasoning, and the reply
comes back with `finish_reason: "length"` and **`content: None`**.

Measured on this machine, with the Reflector's own prompt:

| Call | finish_reason | completion_tokens | content |
|---|---|---:|---|
| plain | `length` | 400 | `None` |
| `enable_thinking: false` | `stop` | 7 | `{"bullets": []}` |

`_extract_json("")` returns `[]`, so the Reflector returns **no deltas and
no error** — it looks exactly like "the two runs had nothing to teach".
On the real corpus the same call with thinking disabled produced **24
bullets**; through `make_ask`, **2**.

If your Reflector reports "no contrast worth a bullet" on a run you
believe has one, check the token budget before you check the contrast.

---

## 8. Act 5 — restart, and the bullet rides in

**The playbook is pinned per `OnIt` instance** so the prompt prefix stays
cache-stable (`src/onit.py:1465-1490`). A long-running session will not
see a bullet written after it started. This is a deployment shape, not a
bug: *reflect offline, then start the agent.*

```bash
# exit the agent, then start it again
onit --config ~/demo/config.yaml
> Compute the weekday of the following date: 2030-06-09
```

```
episodes_used=['7387c19d-3301-4fe6…'] playbook_version=1 | Compute the weekday of the following date: 2030-06-09
```

**Claim 2 is now demonstrated for both loops.** Everything so far is
machinery.

> `scripts/demo_self_improve.py` automates Acts 1–5 end to end, from a
> store it deletes on every run. Useful as a smoke test, useless as
> evidence — it plants its own contrast and grades nothing.

---

## 9. Act 6 — the only act that can carry the claim

Re-measure **the same holdout**, with the store as the only variable:

```bash
cp -r ~/demo/store ~/demo/store.frozen          # freeze the loop's experience

python3 scripts/measure.py --config ~/demo/config.yaml \
    --tasks ~/demo/holdout.jsonl --autonomy adapt \
    --store ~/demo/store.frozen --out ~/demo/treatment.json
```

| Arm | Store | Injection | Holdout |
|---|---|---|---|
| baseline | empty | none (`off`) | *your number* |
| treatment | `store.frozen` | recall + playbook (`adapt`) | *your number* |

A win is `treatment − baseline`, and it must clear an error bar. The bar
this codebase uses is **1.96 × the standard error of the difference**,
`1.96 × sqrt(se_baseline² + se_treatment²)` (`src/learn/runner.py`,
`compare()`): one arm's error at n=40 and p≈0.45 is ~0.079, so the bar
is **~0.218 — about 9 items out of 40**. A delta smaller than that is
noise, and must be reported as such.

That bar is wide, and it is the bar you should want. A two-item delta on
eight items (the run in §10) cleared a 0.153 bar *only* because the
treatment arm was perfect, which makes its error zero. One sloppy item
and the same delta would not have cleared it.

*This number changed on October 28, 2026.*  The bar used to be the **sum**
of the two standard errors — `2s`, which is `1.414σ` of the difference,
not 1.96.  That made the test wrong 15.7% of the time under the null
instead of 5%: a placebo arm scored as a win roughly one time in six.
The bar is now ~1.39× wider than the old one, so **every threshold in
this document that was computed against the old bar is ~39% too small**.
The numbers below have been recomputed.

### The controls that decide whether the win is real

**Control C — repeat the baseline. Do this first.**

**The agent is not deterministic.** The same arm, the same items, the
same empty store, three times:

```
trial 0: 6/8 correct   wrong: 2026-01-08, 2028-04-02
trial 1: 7/8 correct   wrong: 2028-04-02
trial 2: 7/8 correct   wrong: 2028-04-02
```

A one-item swing on the same arm. So **a one-item difference between two
arms is not evidence of anything**, and a two-item difference on eight
items is not evidence of anything either.

Run the baseline three times before you run the treatment once, and take
the baseline's **worst** score as the number you compare against. If the
arms differ by less than the baseline's own spread, you have measured
the model, not the loop.

```bash
for i in 1 2 3; do
  python3 scripts/measure.py --config ~/demo/config.yaml \
      --tasks ~/demo/holdout.jsonl --autonomy off \
      --store ~/demo/empty-$i --out ~/demo/base-$i.json
done
```

**Control A — the null arm.** Re-run the baseline with `--autonomy adapt`
but an **empty** store. If this scores above the baseline, the gain is
from the prompt shape, not from learning.

```bash
python3 scripts/measure.py --config ~/demo/config.yaml \
    --tasks ~/demo/holdout.jsonl --autonomy adapt \
    --store ~/demo/store.empty --out ~/demo/null.json
```

**Control B — shuffled recall.** Re-run the treatment with recall
returning a *random* episode from the store instead of the top-scored
one. If random injection helps as much as recall, `recall()` is not
doing the work — the agent is just being reminded that it is being
watched.

```bash
python3 scripts/measure.py --config ~/demo/config.yaml \
    --tasks ~/demo/holdout.jsonl --autonomy adapt \
    --store ~/demo/store.frozen --shuffle-recall --seed 0 \
    --out ~/demo/shuffled.json
```

`--shuffle-recall` patches the *ranker*, not the store: the same
episodes, the same `episodes_used` bookkeeping, the same block in the
prompt. The only thing destroyed is which episode wins the cut, which
is the only thing this control is allowed to destroy. `--seed` makes it
reproducible.

**Control D — isolated items.** Give every item its own fresh copy of
the store, so item *k* cannot be shown item *k−1*'s run. Without it the
arms are not independent, and a holdout whose questions resemble the
loop's own questions leaks into itself.

```bash
python3 scripts/measure.py --config ~/demo/config.yaml \
    --tasks ~/demo/holdout.jsonl --autonomy adapt \
    --store ~/demo/store.frozen --isolate --out ~/demo/isolated.json
```

`--isolate` writes `<store>.iso/item-NNNN/` — one copy of the store per
item, each holding the seed store plus only its own run. The playbook
is pinned on the `OnIt` instance, so the pin is reset per item too;
otherwise item 2 would still be carrying item 1's pinned playbook.

A result worth publishing looks like:

```
baseline   18/40
null arm   18/40
shuffled   19/40
isolated   19/40
treatment  26/40
```

A result that is **not** worth publishing looks like `18, 24, 25, 26` —
which is what a prompt-shape effect looks like, and it is the most likely
outcome.

### If you have a corpus already: the built-in A/B

```bash
onit learn --holdout
onit learn --eval-run --ab --eval-split dev --eval-limit 40
```

```
A/B on the dev split:
  control  (off): 0.000  n=1  correct=0  wrong=0  not-attempted=1  ungraded=0  errors=1
  treatment(adapt): 0.000  n=1  correct=0  wrong=0  not-attempted=1  ungraded=0  errors=1
  delta: +0.000  (stderr 0.000)  -> not significant
  A delta smaller than its error bar is not a result. Ship nothing on this evidence.
```

**This command does not work today.** The per-item error is
`'NoneType' object has no attribute 'empty'`: `src/learn/runner.py:238`
passes `safety_queue=None` to `process_task`, and `process_task` calls
`effective_safety_queue.empty()` (`src/onit.py:1616-1618`). The queue is
only created in `OnIt.run()`. Every item errors, both arms score 0.000,
and the comparison is between two error streams.

Two more defects sit behind it:

| # | Defect | Effect | Workaround |
|---|---|---|---|
| 1 | `runner.py:238` — `safety_queue=None` | every item errors | `scripts/measure.py` supplies a real `asyncio.Queue` |
| 2 | `runner.py:237` — `data_path` is `<root>/store/<arm>/data`, outside the server's `DATA_PATH` | every tool call refused | `measure.py` uses the config's `data_path` |

Once those are fixed, the built-in A/B collapses into one command. Until
then, **use `scripts/measure.py`**.

### What the built-in split does and does not protect

The holdout is excluded from recall and from reflection **by question**,
not by record id — because the corpus holds the same question many times
("weather today" 21 times, "marikina river water level latest" 6), and
excluding one id leaves its twenty siblings in the pool carrying the
answer. `eval.task_key()` normalises a task to its identity as a
question, and `eval.is_excluded()` is the one predicate every loop goes
through.

Measured on the real split:

| Split | Items | A twin earlier in the same split | Protected by the holdout rule | **Recallable** |
|---|---:|---:|---:|---:|
| holdout | 160 | 6 | 6 | **0** |
| dev | 240 | 4 | 1 | **3** |

The dev split is the *iteration* set, so its questions are not excluded —
3 of 240 dev items can recall an earlier twin's answer. The holdout is
clean.

*October 28, 2026 — the split itself leaked.*  Excluding the holdout from
recall was only half the boundary. The *draw* shuffled records and sliced,
so a question asked in two sessions could land in the holdout on one
record and in dev on the other: the ids were disjoint and the questions
were not. The split drawn on 2026-10-23 had **7 questions on both sides**,
including "what is the current SOTA in TTS" and "marikina river water level
latest". `build_split()` now draws **one record per question**, keeps the
rest out of the measurement (they stay in the store, and the loops still
learn from them), and reports how many siblings it dropped
(`counts.duplicate_questions`, 48 of 400 on the real corpus). The holdout
and dev question sets are now disjoint by construction, and
`TestTheSplitIsDisjointByQuestion` asserts it.

**A split drawn before this fix is still leaking.**  Re-draw it:
`onit learn --holdout` rewrites `holdout.jsonl`, `dev.jsonl` and
`split.json`.

*Done, same day.*  The frozen split in `benchmarks/eval/` was re-drawn:
160 holdout, 240 dev, 48 duplicate questions dropped, **0 questions on both
sides** (it was 7).  The old files are in `tmp/eval-backup-20261023/`, and
the new `split.json` records `counts.duplicate_questions` so the drop is
visible rather than implied.  The seed did not change; the draw did.

---

## 10. What actually happened when this guide was run

This is the worked example, and it is the reason the guide is shaped the
way it is. Everything below is measured, on this machine, with
`Qwen/Qwen3.8-27B` on `http://202.92.159.240:8001/v1`.

**The setup.** The loop worked on three dates (`2027-02-14`, `2028-11-03`,
`2030-06-09`) in the order `[A, pushback, B, C]`. The holdout asked about
**eight other dates**. The store was frozen and copied.

**Act 1–5 all worked as designed.** Recall fired (`episodes_used` non-empty
from the second run on), the contrast was found (the cluster
`compute|weekday|following` held good=1 bad=1), and the labels were
materialised.

**Act 6 produced this:**

| Arm | Store | Injected | Holdout | Contaminated |
|---|---|---|---|---|
| baseline | empty | nothing | 6/8 = 0.750 | 0 |
| null | empty | *self-populated* | **8/8 = 1.000** | 0 |
| treatment | the loop's | *self-populated* | 8/8 = 1.000 | 0 |
| isolated null | empty, fresh per item | **nothing at all** | **8/8** | 0 |
| isolated treatment | the loop's, fresh per item | the loop's runs | 7/8 | **8 of 8** |
| shuffled | the loop's | *self-populated*, **random** episode | *not run* | — |

The shuffled arm was not run in this pass, and that is a gap in the
evidence, not a result: `--shuffle-recall` is a control that only
answers its question once the holdout is clean. With the answer key in
the store, "does the ranking matter" is unanswerable — every arm that
injects anything is reading the key. Run it on a holdout the loop never
touched.

A delta of **+0.250** (stderr 0.153) that cleared its own error bar. On
paper, a win. On inspection, the error bar was the baseline's alone, and
the arms that could have told the difference all failed — see below.

### Why it is not a win

**1. The null arm tied.** The null arm had an empty store, and scored
8/8 — the same as the treatment. So the gain was not the loop's
experience.

**2. The isolated null arm — which injected *nothing at all* — also
scored 8/8.** An empty store means no recall block and no playbook. The
prompt was byte-identical to the baseline arm's. The only difference
between "8/8 with nothing injected" and "6/8 with nothing injected" is
**the model**.

**3. The agent is not deterministic.** The same arm, the same items, the
same empty store, three times: **6, 7, 7**. The spread is one item.
The treatment's advantage was two items on eight.

**4. The isolated treatment arm was 8-of-8 contaminated.** With one fresh
store per item, the loop's store was copied into every item's store — and
the leak check fired on every item. The isolated treatment arm was
*reading the answer key* for all eight items, and still scored 7/8. The
one it missed, it missed *despite* being handed the answer.

**5. The non-isolated treatment arm was also contaminated**, in the sense
that matters: its second item was shown the first item's run, its third
the second's. With the holdout and the loop's dates both drawn from the
same small pool of weekday arithmetic, the arms were not independent.

**6. The error bar it "cleared" was a bar of its own making.** The
reported stderr is 0.153, and the delta is 0.250, so the code called it
significant. But that stderr is the *sum* of two arms' standard errors
(`src/learn/runner.py:294`), and the treatment arm's contribution was
**zero** — because it scored 8/8, and a proportion of 1.0 has a
standard error of 0. So the bar was 0.153 ≈ the baseline arm's error
alone. One wrong item in the treatment arm and the same result would
have been 7/8 against 6/8: a delta of 0.125 against a bar of 0.270,
reported as **not significant**.

That is worth stating plainly, because it is the most likely way a
self-improvement demo lies to itself: **a perfect arm makes its own
error bar look small.** The bar has to be computed from the arms you
*expect*, not from the arms you happened to get. The 0.153 that made
this result "significant" was the baseline's error, and nothing else.

### What this run actually established

| Claim | Verdict |
|---|---|
| 1 — the run was recorded | **yes** |
| 2 — the next run was shown the record | **yes**, `episodes_used` non-empty from the second run |
| 3 — being shown it changed the outcome | **not established** — the delta is within the arm's own run-to-run spread, the null arm ties, and the isolated treatment arm was reading the answer key |

The machinery works. The measurement is now honest enough to say so. What
it says is: **not yet**. The shuffled arm was never run, so the guide
cannot yet say whether `recall()`'s *ranking* does any work either.

### What a real measurement needs, in order

| # | Requirement | Why |
|---|---|---|
| 1 | **n ≥ 40**, and the arm's own spread measured by repeating it | a 2-item delta on 8 items is noise; the spread was 1 item on 8 |
| 2 | **The null arm must be run and must tie** | here it tied at 8/8, which alone kills the claim |
| 3 | **The isolated arm** — a fresh store per item (`--isolate`) | otherwise item *k* is shown item *k-1*'s answer, and the arms are not independent |
| 4 | **The holdout's questions must be absent from the store** | the leak check caught 8 of 8 |
| 5 | **The shuffled arm** (`--shuffle-recall`) | if a *random* episode helps as much as the top-scored one, `recall()` is not doing the work |
| 6 | **The task must not be one the model is bad at** | weekday arithmetic is right ~80% of the time, so the ceiling is low and the noise is high |

**And the error bar is the code's, not yours.** `compare()` uses
`1.96 × sqrt(se_b² + se_t²)`, so at n=40 and p≈0.45 the bar is
**~0.218 — about 9 items**, not the ~3 a single-arm standard error
would suggest. The §10 delta of 2 items cleared a 0.153 bar only
because the treatment arm was perfect. A win that only clears the bar
when one arm is flawless is not a win yet.

**Choose a task family where the baseline is near 50%.** If the agent
already scores 8/8, there is no room to improve and any delta is noise.
If it scores 0/8, there is no room to degrade. The weekday family fails
this test: the baseline was already 6/8.

---

## 11. The one-sentence version

**Start from an empty store, author a holdout the agent will never see,
let it work 20 tasks and react to two of them in two sessions, reflect
offline, restart, then re-measure the frozen holdout — three times, with
a null arm, a shuffled-recall arm and a per-item isolated arm beside it,
and report the delta next to the arm's own run-to-run spread.** If the
delta survives all of that, the agent improved. If it does not, the demo
has still shown you something true: the loops are wired, and wiring is
not learning.

---

## Appendix A — every command in this guide

| Command | What it does |
|---|---|
| `onit --config C` | the agent, with tools |
| `onit --config C learn --json` | the store's summary |
| `onit --config C learn --signals` | how many records carry a label |
| `onit --config C learn --signals --write-signals` | materialise the labels so the Reflector can see them |
| `onit --config C learn --reflect --dry-run` | what the Reflector *would* write |
| `onit --config C learn --reflect` | write the bullets |
| `onit --config C learn --playbook` | what is in the playbook and what would be injected |
| `onit --config C learn --rollback` | restore the previous version |
| `onit --config C learn --eval` | the state of the ruler |
| `onit --config C learn --holdout` | draw and freeze a holdout from the corpus |
| `onit --config C learn --eval-run --ab` | the built-in A/B (broken today) |
| `python3 scripts/measure.py …` | replay a holdout you authored, one arm |
| `… --shuffle-recall --seed N` | Control B: a random episode instead of the top-scored one |
| `… --isolate` | Control D: a fresh copy of the store per item |
| `python3 scripts/demo_self_improve.py` | the mechanism demo, Acts 1–5, automated |

## Appendix B — where each piece lives

| Thing | Where |
|---|---|
| Autonomy ladder (`off`/`observe`/`adapt`) | `src/learn/config.py:41-62`, `:153` |
| Recording a run | `src/learn/trajectory.py:204` (`build_record`), `:272` (`record_task`) |
| Recall | `src/learn/recall.py:261` (`recall`), `:380` (`build_context`) |
| The untrusted-data marker | `src/learn/recall.py:87-91` |
| The playbook | `src/learn/playbook.py:546` (`build_context`), `:798` (`reflect`) |
| The model-based Reflector | `src/learn/reflector.py:213`, `:250` |
| The implicit signals | `src/learn/signals.py:396` (`label_pair`), `:411` (`label_session`), `:440` (`label_records`) |
| The cluster key | `src/learn/playbook.py:638` (`task_shape`) |
| The holdout exclusion | `src/learn/eval.py:526` (`task_key`), `:549` (`holdout_exclusions`), `:593` (`is_excluded`) |
| The A/B | `src/learn/ab.py:73` (`run_ab`), `src/learn/runner.py:189` (`run_arm`), `:294` (`compare`) |
| The endpoint the loops use | `src/learn/llm.py:45` (`resolve_endpoint`) |
| The bash jail | `src/mcp/servers/tasks/os/bash/mcp_server.py:135`, `:186` |
