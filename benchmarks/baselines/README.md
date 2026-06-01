# Baselines

Committed `summary.json` snapshots used as regression gates by
`benchmarks/report.py --baseline`. Generate a baseline from a known-good run:

```bash
python -m benchmarks.run --tier smoke --tasks all
python -m benchmarks.report --log-dir benchmarks/logs/smoke
cp benchmarks/logs/smoke/summary.json benchmarks/baselines/smoke.json
```

Update a baseline deliberately (with justification in the commit message) when a
model/prompt change legitimately moves a metric. CI fails the smoke gate when a
metric drops more than the tolerance (default 0.05) below its baseline.
