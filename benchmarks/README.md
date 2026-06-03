# OnIt Capability Benchmark Suite

Drives the **real OnIt agent** (`OnIt.process_task`) through standard public
benchmarks using [Inspect AI](https://inspect.aisi.org.uk/) as the backbone, and
scores the results. This measures end-to-end agent capability — prompt
engineering, tool use, and the tool loop — not just the underlying LLM.

This suite is intentionally **outside `src/test/`** and excluded from the default
`pytest` run (it is slow, networked, and costs tokens).

See [RESULTS.md](RESULTS.md) for the curated table of `full`-tier benchmark scores.

## Quick start

```bash
pip install -e ".[bench]"          # inspect-ai + dataset deps

# Eval target (defaults to Ollama cloud):
export ONIT_BENCH_MODEL=glm-5.1:cloud
export OLLAMA_API_KEY=...           # or OPENROUTER_API_KEY for OpenRouter

make -C benchmarks bench-smoke      # a few samples per benchmark
make -C benchmarks view             # browse traces in the Inspect viewer
make -C benchmarks report           # write summary.md / summary.json
```

Run specific tasks or categories:

```bash
python -m benchmarks.run --tier smoke   --tasks gsm8k
python -m benchmarks.run --tier sampled --tasks reasoning coding
```

### Resuming an interrupted run

Long runs (especially `--tier full`) persist results incrementally, so a run
that dies part-way — out of cloud credits, killed, crashed — can pick up where
it stopped instead of repeating completed work:

```bash
# Resume the most recent run for this tier (re-runs only the unfinished samples).
python -m benchmarks.run --tier full --resume

# Or point at a specific .eval log.
python -m benchmarks.run --tier full --resume benchmarks/logs/full/<log>.eval
```

Inspect writes each sample to the `.eval` log as it completes; `--resume` calls
`eval_retry` on that log, which re-runs only the incomplete/errored samples and
keeps the rest. (See [SWE-bench](#swe-bench) for resuming that runner.)

## Eval target (environment)

| Variable | Purpose | Default |
|---|---|---|
| `ONIT_BENCH_HOST` | LLM host URL (falls back to `ONIT_HOST`) | `https://api.ollama.com` (Ollama cloud) |
| `ONIT_BENCH_MODEL` | Model id (required for Ollama cloud / OpenRouter) | auto-detected for vLLM |
| `ONIT_BENCH_HOST_KEY` | Explicit API key | else `OLLAMA_API_KEY` / `OPENROUTER_API_KEY` / keychain |
| `ONIT_BENCH_THINK` | Enable thinking mode | `false` |

OpenRouter / local vLLM are selected just by changing `ONIT_BENCH_HOST`.

## Benchmark aliases

Each benchmark has a short **alias** — the name you pass to `--tasks` and the
label shown in Inspect logs and the report table. List them any time with
`python -m benchmarks.run --list`.

| Alias | Benchmark | Category | How it runs |
|---|---|---|---|
| `gsm8k` | GSM8K | reasoning | provider (numeric match) |
| `humaneval` | HumanEval | coding | provider + Docker sandbox |
| `mbpp` | MBPP | coding | provider + Docker sandbox |
| `bigcodebench` | BigCodeBench | coding | provider + Docker (`inspect_evals`) |
| `livecodebench` | LiveCodeBench Pro | coding | provider + Docker (`inspect_evals`) |
| `swe_bench` | SWE-bench | coding | dedicated runner — see [SWE-bench](#swe-bench) |

Categories (`--tasks <category>`): `reasoning`, `coding`, `all`.

## Tiers

| Tier | Samples/benchmark | Concurrency | Use |
|---|---|---|---|
| `smoke` | 5 | 2 | CI gate, catch breakage |
| `sampled` | 100 (seeded) | 4 | nightly tracked signal |
| `full` | all | 8 | leaderboard-comparable, on demand |

## Architecture

| File | Role |
|---|---|
| `onit_provider.py` | Inspect `ModelAPI` (`onit/<label>`) driving `OnIt.process_task` — full-stack mode |
| `onit_agent.py` | Native-tools mode for tool-calling fidelity (BFCL/tau) — *Phase 4 scaffold* |
| `config.py` | Tier presets + eval-target resolution |
| `tasks/` | `@task` definitions by capability |
| `scorers/onit_judge.py` | LLM-as-judge for open-ended factuality |
| `run.py` | CLI wrapper over `inspect_ai.eval` |
| `report.py` | Aggregate logs → markdown/JSON + baseline regression gate |
| `baselines/` | Committed `summary.json` snapshots for regression gating |
| `test_provider.py` | Offline harness tests (stub agent, no model/network) |

## Benchmark coverage (rollout)

| Category | Benchmarks | Status |
|---|---|---|
| Coding | HumanEval, MBPP | **wired** (native, Docker sandbox) |
| | BigCodeBench, LiveCodeBench Pro | **wired** (via `inspect_evals`, Docker) |
| | SWE-bench | **wired** via dedicated runner — see [SWE-bench](#swe-bench) below |
| Reasoning | GSM8K | **wired (smoke)** |
| | GPQA-Diamond, MMLU-Pro, MATH/AIME, BBH, DROP | Phase 3 |
| Factuality | SimpleQA, TruthfulQA, FRAMES | Phase 3 scaffold (`tasks/factuality.py`) |
| Agentic | GAIA, BFCL, tau-bench | Phase 4 scaffold (`tasks/agentic.py`, `onit_agent.py`) |

## SWE-bench

SWE-bench is a **repo-editing agent** benchmark: given a real GitHub issue, the
agent must edit a real codebase so the project's hidden tests pass. It does *not*
fit the final-answer provider path the other coding tasks use, so it has its own
runner: [`benchmarks/swe_bench_runner.py`](swe_bench_runner.py).

### How it works (and where OnIt's sandbox fits)

OnIt roots all of its file tools (`read_file`, `edit_file`, `write_file`,
`grep`, `bash`) at its `data_path`. The runner exploits this in three stages:

1. **Edit — OnIt.** For each instance the repo is cloned at its `base_commit`
   into a per-instance workspace, and OnIt's `data_path` is pointed at it. OnIt
   then reads the issue and edits the source *inside that repo*. With
   `--onit-sandbox`, OnIt executes code through its **MCP sandbox provider**
   (`sandbox_run_code`, etc.) instead of the host — so it can compile/run the
   project safely while iterating. (You can additionally run the whole runner
   inside `onit --container` for process-level isolation.)
2. **Capture.** The model patch is the workspace `git diff`, with test-file
   edits stripped out (the grader supplies the gold test patch).
3. **Grade — official harness.** Patches are written to `predictions.jsonl` and
   scored by the official `swebench` Docker harness, which applies each patch to
   the canonical per-instance image and runs the tests. This is exactly the
   grader behind the public leaderboard, so scores are comparable.

### Prerequisites

```bash
pip install -e ".[bench]"      # inspect-ai, inspect-evals, datasets
pip install swebench           # the official grading harness
# Docker daemon running; ~120 GB free disk for the full image set
# (Lite/Verified subsets pull far fewer images).
export HF_TOKEN=...            # recommended for dataset/image pulls
# Eval target, as for all benchmarks:
export ONIT_BENCH_MODEL=glm-5.1:cloud
export OLLAMA_API_KEY=...
```

If you use `--onit-sandbox`, also configure an MCP sandbox provider (a server
exposing `sandbox_run_code` / `sandbox_install_packages` / `sandbox_stop`) and
set `sandbox: true` works automatically via the flag. Without it, OnIt edits and
runs on the host — fine inside a container or VM, riskier on a workstation.

### Run it

```bash
# Smoke: 5 instances of SWE-bench Lite, host execution, then grade.
python -m benchmarks.swe_bench_runner --dataset lite --tier smoke

# Sampled: 100 Lite instances, OnIt sandboxed, 4 parallel graders.
python -m benchmarks.swe_bench_runner --dataset lite --tier sampled \
    --onit-sandbox --run-id onit-lite --max-workers 4

# Full SWE-bench Verified (500), leaderboard-comparable.
python -m benchmarks.swe_bench_runner --dataset verified --run-id onit-verified \
    --max-workers 8

# Generate patches only (e.g. to grade later / on another machine):
python -m benchmarks.swe_bench_runner --dataset lite --tier smoke --no-grade
```

**Resuming:** the runner is resumable by default. Each prediction is written to
`predictions_<run-id>.jsonl` as soon as the instance finishes, so a run that
dies part-way (out of credits, killed) can be re-invoked with the *same*
`--run-id` and `--data-root`: instances that already succeeded are skipped, and
instances that errored last time are re-attempted (from a freshly reset
workspace). Pass `--fresh` to ignore prior predictions and start over.

```bash
# First run dies at instance 70/100 (e.g. credits run out)...
python -m benchmarks.swe_bench_runner --dataset lite --tier sampled --run-id onit-lite
# ...re-run the exact same command once credits are back; it resumes from 70.
python -m benchmarks.swe_bench_runner --dataset lite --tier sampled --run-id onit-lite
```

| Flag | Purpose | Default |
|---|---|---|
| `--dataset` | `lite` (300) / `verified` (500) / `full` (2294) | `lite` |
| `--tier` | sets `--limit` from the tier preset (smoke=5, sampled=100, full=all) | — |
| `--limit` | explicit instance cap (overrides `--tier`) | — |
| `--onit-sandbox` | OnIt executes code via its MCP sandbox provider | off |
| `--run-id` | label for predictions + harness report | `onit` |
| `--max-workers` | parallel grading containers | 4 |
| `--data-root` | where workspaces + `predictions.jsonl` live | `$TMPDIR/onit-swebench` |
| `--no-grade` | generate patches only, skip Docker grading | off |
| `--fresh` | ignore existing predictions and start over (default: resume) | off |

### Output

The runner prints a resolve-rate table and the harness writes a per-instance
report:

```
# SWE-bench summary

| Dataset | Model        | Instances | Resolved | Resolve rate |
|---------|--------------|-----------|----------|--------------|
| lite    | glm-5.1:cloud| 5         | 2        | 0.400        |
```

Per-instance pass/fail and logs are in the harness's `logs/run_evaluation/<run_id>/`.

### Notes & caveats

- **Two sandboxes, by design:** OnIt's sandbox isolates the *editing/iteration*
  step; the official harness's Docker images are the *graded* environment. They
  are separate on purpose — grading must use the canonical image to be valid.
- This runner is **separate from the Inspect pipeline** (`run.py`/`report.py`)
  because grading is orchestrated by the SWE-bench Docker harness, not an Inspect
  scorer.
- The stock `inspect_evals` SWE-bench task is still registered as `swe_bench` in
  `run.py`, but it benchmarks Inspect's *own* tool-calling agent, **not OnIt** —
  use it only for comparison.
- First run is slow: cloning repos and pulling SWE-bench images dominates. Reuse
  `--data-root` across runs to keep cloned workspaces.

## Notes

- **Code execution** (HumanEval and other coding tasks) runs inside an Inspect
  Docker sandbox (`sandbox="docker"`), mirroring OnIt's `--container`/`--sandbox`
  posture. A Docker daemon must be available for coding tasks.
- **Gated datasets** (GPQA, GAIA) need `HF_TOKEN`.
- **Judge bias:** `onit_judge` defaults to the model under test; pass a stronger
  judge for `full` factuality runs.
- The full suite plan lives at the repo planning doc referenced in the PR.
