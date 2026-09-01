# OnIt Benchmark Results

Baseline model pin: **`Qwen/Qwen3.8-27B`** (set as `benchmarks/config.py`
`DEFAULT_MODEL`, 2026-08-31). Pinned runs target the agent's own model serving
setup: the preferred endpoint from `~/.onit/config.yaml` (the benchmark
inherits the agent's serving config — host IP, model, and the keychain key —
falling back to `http://localhost:8000/v1` only when no agent config exists;
see docs/RUN_A_MODEL_SERVER.md). The table below is the model-selection
evidence from earlier runs (Qwen3.**6**-27B on vLLM among them); the first
pinned-model run will add its `summary.json` under `baselines/` and a row here.

Tier note: rows marked **full** ran the complete dataset (leaderboard-
comparable); rows marked **sampled** ran the fixed seeded 100-sample subset
(nightly tracked signal, used for the regression gate). Unmarked rows are from
the earlier model-selection runs.

Scope note (2026-08-31): SWE-bench and LiveCodeBench have been **removed** from
the benchmark suite (runner, task wrappers, `swe_bench` extra, docs). The
tracked set is bigcodebench, gsm8k, humaneval, mbpp, plus the METR
time-horizon layer for long-horizon capability.

**Correction (2026-09-01):** the earlier MBPP full row (0.856) was produced while the benchmark's MCP servers failed to start on macOS (a `multiprocessing.Pool`-from-daemon-thread bug plus a legacy `ToolsMCPServer` config entry with no module), so the agent discovered **0 tools** and every sample scored as a tool-intent miss. That number was a harness failure, not a model measurement. The 0.899 above is the first MBPP run with the agent's full tool set (14 tools) live; see commit `946ee30`.

| Benchmark | Host | Model | Accuracy | Stderr |
|---|---|---|---|---|
| bigcodebench | vLLM (private endpoint) | Qwen/Qwen3.6-27B | 0.518 | 0.015 |
| gsm8k | https://api.ollama.com | glm-5.1:cloud | 0.978 | 0.004 |
| gsm8k | https://api.ollama.com | gemma4:31b-cloud | 0.978 | 0.004 |
| gsm8k | vLLM (private endpoint) | Qwen/Qwen3.6-27B | 0.953 | 0.006 |
| gsm8k (full, 1,319) | vLLM (agent endpoint) | Qwen/Qwen3.8-27B | **0.977** | 0.004 |
| gsm8k (sampled, 100) | vLLM (agent endpoint) | Qwen/Qwen3.8-27B | 0.980 | 0.014 |
| gsm8k | https://api.ollama.com | minimax-m3:cloud | 0.937 | 0.007 |
| gsm8k | https://api.ollama.com | rnj-1:8b | 0.927 | 0.007 |
| humaneval (full, 164) | vLLM (agent endpoint) | Qwen/Qwen3.8-27B | **0.927** | 0.020 |
| humaneval | https://api.ollama.com | minimax-m3:cloud | 0.921 | 0.021 |
| humaneval | vLLM (private endpoint) | Qwen/Qwen3.6-27B | 0.915 | 0.022 |
| humaneval | https://api.ollama.com | gemma4:31b-cloud | 0.909 | 0.023 |
| humaneval | https://api.ollama.com | glm-5.1:cloud | 0.890 | 0.031 |
| humaneval | https://api.ollama.com | rnj-1:8b | 0.848 | 0.028 |
| mbpp (full, 257) | vLLM (agent endpoint) | Qwen/Qwen3.8-27B | **0.899** | 0.019 |
| mbpp | https://api.ollama.com | glm-5.1:cloud | 0.957 | 0.013 |
| mbpp | https://api.ollama.com | minimax-m3:cloud | 0.942 | 0.015 |
| mbpp | vLLM (private endpoint) | Qwen/Qwen3.6-27B | 0.922 | 0.017 |
| mbpp | https://api.ollama.com | gemma4:31b-cloud | 0.895 | 0.019 |
| mbpp | https://api.ollama.com | rnj-1:8b | 0.895 | 0.019 |
