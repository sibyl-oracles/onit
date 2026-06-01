# OnIt Capability Benchmark Suite

Drives the **real OnIt agent** (`OnIt.process_task`) through standard public
benchmarks using [Inspect AI](https://inspect.aisi.org.uk/) as the backbone, and
scores the results. This measures end-to-end agent capability — prompt
engineering, tool use, and the tool loop — not just the underlying LLM.

This suite is intentionally **outside `src/test/`** and excluded from the default
`pytest` run (it is slow, networked, and costs tokens).

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

## Eval target (environment)

| Variable | Purpose | Default |
|---|---|---|
| `ONIT_BENCH_HOST` | LLM host URL (falls back to `ONIT_HOST`) | `https://api.ollama.com` (Ollama cloud) |
| `ONIT_BENCH_MODEL` | Model id (required for Ollama cloud / OpenRouter) | auto-detected for vLLM |
| `ONIT_BENCH_HOST_KEY` | Explicit API key | else `OLLAMA_API_KEY` / `OPENROUTER_API_KEY` / keychain |
| `ONIT_BENCH_THINK` | Enable thinking mode | `false` |

OpenRouter / local vLLM are selected just by changing `ONIT_BENCH_HOST`.

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
| Coding | HumanEval | **wired (smoke)** |
| | MBPP, LiveCodeBench, SWE-bench | Phase 2 |
| Reasoning | GSM8K | **wired (smoke)** |
| | GPQA-Diamond, MMLU-Pro, MATH/AIME, BBH, DROP | Phase 3 |
| Factuality | SimpleQA, TruthfulQA, FRAMES | Phase 3 scaffold (`tasks/factuality.py`) |
| Agentic | GAIA, BFCL, tau-bench | Phase 4 scaffold (`tasks/agentic.py`, `onit_agent.py`) |

## Notes

- **Code execution** (HumanEval and other coding tasks) runs inside an Inspect
  Docker sandbox (`sandbox="docker"`), mirroring OnIt's `--container`/`--sandbox`
  posture. A Docker daemon must be available for coding tasks.
- **Gated datasets** (GPQA, GAIA) need `HF_TOKEN`.
- **Judge bias:** `onit_judge` defaults to the model under test; pass a stronger
  judge for `full` factuality runs.
- The full suite plan lives at the repo planning doc referenced in the PR.
