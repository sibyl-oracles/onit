# OnIt benchmark suite. Run from the repo root: `make -C benchmarks bench-smoke`.
#
# Eval target is resolved from the environment (defaults to the agent's own
# vLLM serving setup, http://localhost:8000/v1 — docs/RUN_A_MODEL_SERVER.md):
#   ONIT_BENCH_HOST   (default http://localhost:8000/v1)
#   ONIT_BENCH_MODEL  (default pinned Qwen/Qwen3.8-27B; required for
#                      Ollama cloud / OpenRouter)
#   VLLM_API_KEY      (only if the server was started with --api-key)

PY ?= python
TASKS ?= all

.PHONY: bench-smoke bench-sampled bench-full bench-baseline bench-test view report

# CI gate: a few samples per benchmark.
bench-smoke:
	$(PY) -m benchmarks.run --tier smoke --tasks $(TASKS)

# Nightly: seeded subset for a tracked capability signal.
bench-sampled:
	$(PY) -m benchmarks.run --tier sampled --tasks $(TASKS)

# On-demand: full datasets, leaderboard-comparable.
bench-full:
	$(PY) -m benchmarks.run --tier full --tasks $(TASKS)

# Pinned baseline set (bigcodebench, gsm8k, humaneval, mbpp) on the pinned
# model. Produces logs/<tier>/summary.json for benchmarks/baselines/.
bench-baseline:
	$(PY) -m benchmarks.run --tier sampled --tasks baseline

# Provider/harness unit tests (no model or network needed).
bench-test:
	$(PY) -m pytest benchmarks/test_provider.py -v

# Open the Inspect log viewer.
view:
	inspect view --log-dir benchmarks/logs

# Aggregate the latest logs into a markdown + JSON report.
report:
	$(PY) -m benchmarks.report --log-dir benchmarks/logs
