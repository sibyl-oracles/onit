# OnIt benchmark suite. Run from the repo root: `make -C benchmarks bench-smoke`.
#
# Eval target is resolved like the agent's own: benchmark env vars, then the
# agent's ~/.onit/config.yaml (preferred endpoint + model, key from the OS
# keychain), then ONIT_HOST, then the local-vLLM default:
#   ONIT_BENCH_HOST   (overrides the agent's configured endpoint)
#   ONIT_BENCH_MODEL  (overrides the agent's; default pinned Qwen/Qwen3.8-27B,
#                      required for Ollama cloud / OpenRouter)
#   ONIT_BENCH_HOST_KEY (else the agent's keychain key for the endpoint)

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
