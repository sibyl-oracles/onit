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

**Correction (2026-09-01):** the earlier MBPP (0.856), HumanEval (0.927), and BigCodeBench (0.121) full rows were all produced while the benchmark's MCP servers failed to start on macOS (a `multiprocessing.Pool`-from-daemon-thread bug plus a legacy `ToolsMCPServer` config entry with no module), so the agent discovered **0 tools** and every sample scored as a tool-intent miss. Those numbers were harness failures, not model measurements. The MBPP (0.899) and HumanEval (0.902) rows above are the first runs with the agent's full tool set (14 tools) live; see commit `946ee30`. A second fix (commit below) repaired a context-compaction crash (`AttributeError: 'ChatCompletionMessage' object has no attribute 'get'` in `_compact_context`) that killed the HumanEval run at sample 112 once a sample's context grew large enough to trigger compaction. BigCodeBench was re-run with both fixes in place (14 tools live, compaction fix): the result is **0.025** (28/1,140 correct), down from the 0-tool 0.121. The drop is not a regression — 96.8% of the 1,140 samples have no code block in the model's output (the agent's system prompt encourages tool use, so the model responds with tool-intent text instead of writing code). The 0.025 is a real measurement of the agent's behavior on BigCodeBench, not a harness failure. The dataset also loaded 1,140 problems (the full test split) rather than the 612 from the earlier run.

**Correction (2026-09-03):** the full-tier BigCodeBench run of 2026-09-02 (log
`logs/full/2026-09-02T12-09-39-00-00_bigcodebench_8qUhiif6wHGLjCuUVWRjBA.eval`;
`run_meta.json`: tier full, model `onit/Qwen/Qwen3.8-27B`, learn off) was
interrupted at sample 486 of 1,140. Over the 486 recorded samples the tally is
274 correct / 212 incorrect = **0.564** (stderr 0.022; the 0.56 mean shown by
the log viewer). This supersedes the 0.025 row for the pinned model. The
completions confirm the earlier diagnosis and its repair: 450/486 (92.6%)
contain a fenced code block (vs 17.6% in the 0.121-era log), average completion
length is 2,128 chars (vs 834), and final answers reference writing
`task_func.py` and verifying it — i.e. the agent's tool loop completed, where
the 0.025-era runs returned raw tool-intent prose with no code block. Commit
`55b7d0e` (pre-import the PyPI `mcp` SDK before the local `src/mcp` shadow)
landed immediately before this run started and is the likely repair. Caveat:
the recorded sample ids are the low-id prefix (0–487), so 0.564 is the score
over recorded samples, not a full-dataset estimate; a completed 1,140-sample
run (~7 h at this run's pace) is needed for a leaderboard-comparable number.

| Benchmark | Host | Model | Accuracy | Stderr |
|---|---|---|---|---|
| bigcodebench (full, 486/1,140 recorded) | vLLM (agent endpoint) | Qwen/Qwen3.8-27B | **0.564** | 0.022 |
| bigcodebench (full, 1,140) | vLLM (agent endpoint) | Qwen/Qwen3.8-27B | ~~0.025~~ superseded 2026-09-03 | — |
| bigcodebench | vLLM (private endpoint) | Qwen/Qwen3.6-27B | 0.518 | 0.015 |
| gsm8k | https://api.ollama.com | glm-5.1:cloud | 0.978 | 0.004 |
| gsm8k | https://api.ollama.com | gemma4:31b-cloud | 0.978 | 0.004 |
| gsm8k | vLLM (private endpoint) | Qwen/Qwen3.6-27B | 0.953 | 0.006 |
| gsm8k (full, 1,319) | vLLM (agent endpoint) | Qwen/Qwen3.8-27B | **0.977** | 0.004 |
| gsm8k (sampled, 100) | vLLM (agent endpoint) | Qwen/Qwen3.8-27B | 0.980 | 0.014 |
| gsm8k | https://api.ollama.com | minimax-m3:cloud | 0.937 | 0.007 |
| gsm8k | https://api.ollama.com | rnj-1:8b | 0.927 | 0.007 |
| humaneval (full, 164) | vLLM (agent endpoint) | Qwen/Qwen3.8-27B | **0.902** | 0.023 |
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
