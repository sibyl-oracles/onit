# OnIt benchmark suite.
#
# Targets run from the repo root regardless of how make is invoked:
#   make -C benchmarks bench-smoke     (cds into benchmarks/)
#   make -f benchmarks/Makefile bench-smoke   (stays at the repo root)
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

# Repo root, derived from this Makefile's own location, so the recipes work
# from any invocation directory (see header). The benchmarks package and the
# test paths are repo-root-relative.
ROOT := $(abspath $(dir $(firstword $(MAKEFILE_LIST))))/..

.PHONY: bench-smoke bench-sampled bench-full bench-baseline bench-test view report

# CI gate: a few samples per benchmark.
bench-smoke:
	cd $(ROOT) && $(PY) -m benchmarks.run --tier smoke --tasks $(TASKS)

# Nightly: seeded subset for a tracked capability signal.
bench-sampled:
	cd $(ROOT) && $(PY) -m benchmarks.run --tier sampled --tasks $(TASKS)

# On-demand: full datasets, leaderboard-comparable.
bench-full:
	cd $(ROOT) && $(PY) -m benchmarks.run --tier full --tasks $(TASKS)

# Pinned baseline set (bigcodebench, gsm8k, humaneval, mbpp) on the pinned
# model. Produces logs/<tier>/summary.json for benchmarks/baselines/.
bench-baseline:
	cd $(ROOT) && $(PY) -m benchmarks.run --tier sampled --tasks baseline

# Provider/harness unit tests (no model or network needed).
bench-test:
	cd $(ROOT) && $(PY) -m pytest benchmarks/test_provider.py benchmarks/test_coding.py -v

# Open the Inspect log viewer.
view:
	cd $(ROOT) && inspect view --log-dir benchmarks/logs

# Aggregate the latest logs into a markdown + JSON report.
report:
	cd $(ROOT) && $(PY) -m benchmarks.report --log-dir benchmarks/logs