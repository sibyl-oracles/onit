# OnIt benchmark suite. Run from the repo root: `make -C benchmarks bench-smoke`.
#
# Eval target is resolved from the environment (defaults to Ollama cloud):
#   ONIT_BENCH_HOST   (default https://api.ollama.com)
#   ONIT_BENCH_MODEL  (required for Ollama cloud / OpenRouter)
#   OLLAMA_API_KEY / OPENROUTER_API_KEY  (as appropriate)

PY ?= python
TASKS ?= all

.PHONY: bench-smoke bench-sampled bench-full bench-test view report

# CI gate: a few samples per benchmark.
bench-smoke:
	$(PY) -m benchmarks.run --tier smoke --tasks $(TASKS)

# Nightly: seeded subset for a tracked capability signal.
bench-sampled:
	$(PY) -m benchmarks.run --tier sampled --tasks $(TASKS)

# On-demand: full datasets, leaderboard-comparable.
bench-full:
	$(PY) -m benchmarks.run --tier full --tasks $(TASKS)

# Provider/harness unit tests (no model or network needed).
bench-test:
	$(PY) -m pytest benchmarks/test_provider.py -v

# Open the Inspect log viewer.
view:
	inspect view --log-dir benchmarks/logs

# Aggregate the latest logs into a markdown + JSON report.
report:
	$(PY) -m benchmarks.report --log-dir benchmarks/logs
