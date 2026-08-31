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

# Eval target: the agent's own vLLM setup (http://localhost:8000/v1).
# Start the server for the pinned model — see docs/RUN_A_MODEL_SERVER.md:
vllm serve Qwen/Qwen3.8-27B --port 8000 \
  --max-model-len 262144 --enable-auto-tool-choice --tool-call-parser hermes \
  --reasoning-parser qwen3 --chat-template-content-format string \
  --enable-prefix-caching

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

Runs **resume by default**. Inspect writes each sample to the `.eval` log as it
completes, so a run that dies part-way — out of cloud credits, killed, crashed —
leaves the finished samples on disk. Re-running the *same command* detects the
most recent incomplete log for each task (matched by task + model + tier) and
re-runs only the unfinished samples, keeping the rest:

```bash
# First run is cancelled at 78/1319 samples (e.g. credits run out)...
python -m benchmarks.run --tier full --tasks bigcodebench
# ...just run it again; it picks up from sample 79.
python -m benchmarks.run --tier full --tasks bigcodebench

# Force a clean run, ignoring any prior logs:
python -m benchmarks.run --tier full --tasks bigcodebench --fresh
```

A task whose most recent log already finished (`success`), used a different
model, or doesn't exist yet is run fresh. (See [SWE-bench](#swe-bench) for
resuming that runner.)

## Eval target (environment)

The default target is the agent's own vLLM serving setup
([RUN_A_MODEL_SERVER.md](../docs/RUN_A_MODEL_SERVER.md)): host
`http://localhost:8000/v1`, model auto-detected from `/v1/models` unless pinned,
no API key unless the server was started with `--api-key` (then `VLLM_API_KEY`).
`run.py` probes a local endpoint before the first sample and fails fast if it
is unreachable or does not serve the pinned model.

| Variable | Purpose | Default |
|---|---|---|
| `ONIT_BENCH_HOST` | LLM host URL (falls back to `ONIT_HOST`) | `http://localhost:8000/v1` (local vLLM) |
| `ONIT_BENCH_MODEL` | Model id (defaults to the pinned `Qwen/Qwen3.8-27B`; set to `""` to auto-detect from the endpoint) | pinned model |
| `ONIT_BENCH_HOST_KEY` | Explicit API key | else `VLLM_API_KEY` / `OLLAMA_API_KEY` / `OPENROUTER_API_KEY` / keychain |
| `ONIT_BENCH_THINK` | Enable thinking mode | `false` |

Ollama cloud / OpenRouter are selected just by changing `ONIT_BENCH_HOST`
(and `ONIT_BENCH_MODEL`, required there).

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
| `metr` | METR Time Horizon | long-horizon | METR bridge + Docker — see [METR time horizon](#metr-time-horizon) |

Categories (`--tasks <category>`): `reasoning`, `coding`, `baseline`, `all`.
`baseline` is the pinned four-task set (bigcodebench, gsm8k, humaneval, mbpp)
that the regression gate tracks.

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
| | BigCodeBench | **wired** (via `inspect_evals`, Docker) |
| Reasoning | GSM8K | **wired (smoke)** |
| | GPQA-Diamond, MMLU-Pro, MATH/AIME, BBH, DROP | Phase 3 |
| Factuality | SimpleQA, TruthfulQA, FRAMES | Phase 3 scaffold (`tasks/factuality.py`) |
| Agentic | GAIA, BFCL, tau-bench | Phase 4 scaffold (`tasks/agentic.py`, `onit_agent.py`) |

## METR time horizon

Long-horizon capability is measured with the [METR time-horizon
methodology](https://arxiv.org/abs/2503.14499): run the agent on tasks with
known human-expert completion times, fit P(success) vs log2(human-minutes),
and report the **50%-time-horizon** — the task duration the agent succeeds at
half the time.

What is in the repo:

* `tasks/metr.py` — the logistic fit (`horizon_summary`, `time_horizon_50`)
  and the `metr` Inspect task; `test_metr.py` covers the fit offline.
* `data/metr_human_minutes.json` — the 228-task human-time label map from
  METR's Time Horizon v1.1 release (HCAST + RE-Bench + SWAA; 0.02–1800 min).

Running a live task family (requires the METR bridge, Docker, and task
images — see [METR/inspect-metr-task-bridge](https://github.com/METR/inspect-metr-task-bridge)):

```bash
pip install git+https://github.com/METR/inspect-metr-task-bridge
python -m benchmarks.run --tier sampled --tasks metr \
    -T image_tag=password_check-1.0.13
```

Samples are tagged with their family's median human time, so the task-level
`time_horizon_50` metric lands in `summary.json` next to the accuracy metrics
and flows through the regression gate like any other task. 14 of the labeled
families (password_check, sadservers, make_web_server, symbolic_regression, …)
are runnable today via the bridge; the rest of the labeled set exists only as
recorded outcomes in METR's release.

## Notes

- **Code execution** (HumanEval and other coding tasks) runs inside an Inspect
  Docker sandbox (`sandbox="docker"`), mirroring OnIt's `--container`/MCP-sandbox
  posture. A Docker daemon must be available for coding tasks.
- **Gated datasets** (GPQA, GAIA) need `HF_TOKEN`.
- **Judge bias:** `onit_judge` defaults to the model under test; pass a stronger
  judge for `full` factuality runs.
- The full suite plan lives at the repo planning doc referenced in the PR.
