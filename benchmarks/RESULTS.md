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

**Correction (2026-09-05):** the SimpleQA smoke run (5 samples,
`logs/smoke/2026-09-05T12-33-04-…simpleqa_PVy4Ynunzbijvc67rJ6qJU.eval`) is the
first run of the task registered in §8 item 2: **1.000** accuracy
(model-graded C on all 5), 14 tools live, 2 m 43 s. The judge path
(`onit_judge` → `model_graded_qa`) is validated; a judge-rotation regrade of
the same 5 submissions with `glm-5.1:cloud` agreed (5×C), so self-judging bias
is not visible at this sample size. The same session also fixed the MCP
spawn-child crash-loop (`sys.path` shadow in `multiprocessing` spawn children;
the `55b7d0e` fix pinned the SDK in the parent only): Prompts/ToolsNet/
VLMTools servers exited code 1 every 10 s for the whole first smoke attempt.
GAIA remains unrun: the dataset is gated (`GatedRepoError 401` without a
token) and its task wrapper passed `trust=True`, which datasets 5.x rejects —
wrapper fixed (no `trust` flag; the hub repo is Parquet-backed since
October 2025), but a run still needs `HF_TOKEN` after accepting the dataset's
terms. The baseline pin (`baselines/full.json`) now exists with the two
verifiable full-tier rows (gsm8k 0.977/1,319; bigcodebench 0.564/486);
humaneval/mbpp are excluded until re-run because their only surviving logs
are the 0-tool harness-failure era (0.927/0.856) and the runs that superseded
them (0.902/0.899) left no logs.

**Correction (2026-09-06):** first **sampled-tier** runs of the newly
registered tasks, all on the agent's own preferred endpoint
(`onit/glm-5.3-flash:cloud`, 14 tools live, learn off, `run_meta.json` in
`logs/sampled/`):

* **simpleqa (sampled, 100): 0.940** (94 C / 6 I / 0 not-attempted; stderr
  0.024; log `logs/sampled/2026-09-06T00-33-31-…JridUjXJGZ6jWVB2Y28e4a.eval`).
  The agent never abstained — every sample was graded C or I, so the
  not-attempted bucket the SimpleQA metric normally reports is empty here.
  Two earlier partial attempts of the same run (resumed by `eval_retry`)
  finished at 0.93/0.95 before the final 0.940 — run-to-run noise at n=100 is
  ±0.02–0.03, consistent with the stderr.
* **humaneval (sampled, 100): 0.970** (stderr 0.017; log
  `2026-09-06T00-44-21-…66anSTB6REe3hSRZVb26NY.eval`) and **mbpp (sampled,
  100): 0.990** (stderr 0.010; log `2026-09-06T00-51-09-…iJzq65qhSwQ9toB2Grv7tm.eval`).
  These are the first verifiable verbatim-dataset rows for both tasks (the
  full-tier 0.902/0.899 rows below predate the surviving-log problem and stay
  excluded from the pin). A duplicate-log quirk (two `.eval` files per task,
  agreeing within 0.01) is noted for the runner to consolidate.
* GAIA: the dataset was cached locally on 2026-09-06 (validation 165 rows +
  test 83 rows, parquet snapshots under `~/.cache/huggingface/`), so the task
  now loads **offline** (`HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1`) without a
  token — the gated-repo blocker is bypassed for cached configs. Caveat: the
  task wrapper passes only `Question`/`Final answer` and ignores
  `file_path`, so GAIA runs are **text-only by construction** (attachment
  tasks are unanswerable as specified) — a known scope limit to fix in a
  later pass.

| Benchmark | Host | Model | Accuracy | Stderr |
|---|---|---|---|---|
| bigcodebench (full, 486/1,140 recorded) | vLLM (agent endpoint) | Qwen/Qwen3.8-27B | **0.564** | 0.022 |
| bigcodebench (full, 1,140) | vLLM (agent endpoint) | Qwen/Qwen3.8-27B | ~~0.025~~ superseded 2026-09-03 | — |
| simpleqa (smoke, 5) | https://api.ollama.com | glm-5.3-flash:cloud | 1.000 | 0.000 |
| simpleqa (sampled, 100) | https://api.ollama.com | glm-5.3-flash:cloud | **0.940** | 0.024 |
| humaneval (sampled, 100) | https://api.ollama.com | glm-5.3-flash:cloud | **0.970** | 0.017 |
| mbpp (sampled, 100) | https://api.ollama.com | glm-5.3-flash:cloud | **0.990** | 0.010 |
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
