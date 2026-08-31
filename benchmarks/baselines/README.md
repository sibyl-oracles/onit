# Baselines

Committed `summary.json` snapshots used as regression gates by
`benchmarks/report.py --baseline`. Generate a baseline from a known-good run:

```bash
python -m benchmarks.run --tier sampled --tasks baseline   # pinned model + pinned task set
python -m benchmarks.report --log-dir benchmarks/logs/sampled
cp benchmarks/logs/sampled/summary.json benchmarks/baselines/sampled.json
```

The pinned task set is the `baseline` category (bigcodebench, gsm8k, humaneval,
mbpp) on the pinned model (`Qwen/Qwen3.8-27B` unless `ONIT_BENCH_MODEL`
overrides it). A baseline must state the model it was produced under — rows
from different models are not comparable.

Update a baseline deliberately (with justification in the commit message) when a
model/prompt change legitimately moves a metric. CI fails the smoke gate when a
metric drops more than the tolerance (default 0.05) below its baseline.
