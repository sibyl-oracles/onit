"""
# Copyright 2025 Rowel Atienza. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

Serving-endpoint diagnostics.

Run against a live vLLM host, from the repo root:

    python -m src.model.serving.diagnostics [host]

``host`` defaults to $ONIT_HOST, then http://localhost:8000/v1.

Reports whether automatic prefix caching is on and how often it hits.  Inside
one task the conversation only ever grows at the end, so every turn after the
first re-sends a prompt the server has already seen: with caching on, that
prefix is not prefilled again.  Without it, each extra turn re-processes the
whole conversation, which is invisible in a tokens/sec figure measured over
the final answer alone.
"""

import asyncio
import os
import sys
from typing import Optional

import httpx

# Metric names vLLM exposes for automatic prefix caching.  V1 reports the two
# counters; V0 reported a pre-computed rate.  Either one answers the question.
_QUERY_METRICS = ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries")
_HIT_METRICS = ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits")
_RATE_METRICS = ("vllm:gpu_prefix_cache_hit_rate",)


def _metrics_url(host: str) -> str:
    """vLLM serves /metrics at the server root, not under the /v1 API prefix."""
    base = host.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return f"{base}/metrics"


def parse_prometheus(text: str, names: tuple) -> Optional[float]:
    """Sum the samples of the first metric in ``names`` that appears.

    A metric is reported once per model and per engine, so a host serving one
    model still emits several lines; summing keeps the ratio right either way.
    Returns None when none of the names are present at all — which is itself
    the answer, since a build without prefix caching never emits them.
    """
    total = None
    for name in names:
        for line in text.splitlines():
            if not line or line.startswith("#") or not line.startswith(name):
                continue
            # "name{labels} value" or "name value" — the value is the last field
            head, _, value = line.rpartition(" ")
            if not head.split("{", 1)[0].strip() == name:
                continue
            try:
                total = (total or 0.0) + float(value)
            except ValueError:
                continue
        if total is not None:
            return total
    return None


async def prefix_cache_report(host: str, api_key: str = "EMPTY",
                              timeout: float = 10.0) -> dict:
    """Ask a vLLM host whether automatic prefix caching is on and working.

    Returns {"reachable", "enabled", "queries", "hits", "hit_rate", "detail"}.
    ``enabled`` is None when the host answered but said nothing about prefix
    caching — an older build, a non-vLLM server, or metrics turned off.
    """
    url = _metrics_url(host)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, headers={"Authorization": f"Bearer {api_key}"})
    except Exception as e:
        return {"reachable": False, "enabled": None, "queries": None, "hits": None,
                "hit_rate": None, "detail": f"could not reach {url}: {e}"}

    if resp.status_code != 200:
        return {"reachable": False, "enabled": None, "queries": None, "hits": None,
                "hit_rate": None,
                "detail": f"{url} returned HTTP {resp.status_code}"}

    text = resp.text
    queries = parse_prometheus(text, _QUERY_METRICS)
    hits = parse_prometheus(text, _HIT_METRICS)
    rate = parse_prometheus(text, _RATE_METRICS)

    if queries is None and rate is None:
        return {"reachable": True, "enabled": None, "queries": None, "hits": None,
                "hit_rate": None,
                "detail": "host exposes no prefix-cache metrics; it is either "
                          "not vLLM, predates them, or was started with "
                          "--no-enable-prefix-caching"}

    if rate is not None and queries is None:
        hit_rate = rate
    elif queries:
        hit_rate = (hits or 0.0) / queries
    else:
        hit_rate = None

    # Metrics present means the feature is compiled in and reporting; queries
    # accumulating means requests are actually being looked up in the cache.
    enabled = True if (queries or rate is not None) else None
    return {"reachable": True, "enabled": enabled, "queries": queries,
            "hits": hits, "hit_rate": hit_rate,
            "detail": "prefix caching is reporting"}


def format_report(host: str, report: dict) -> str:
    lines = [f"vLLM prefix cache — {host}"]
    if not report["reachable"]:
        lines.append(f"  unreachable: {report['detail']}")
        return "\n".join(lines)
    if report["enabled"] is None:
        lines.append(f"  status:   unknown — {report['detail']}")
        lines.append("  fix:      start vLLM with --enable-prefix-caching "
                     "(V1 enables it by default) and leave /metrics on")
        return "\n".join(lines)

    lines.append("  status:   enabled")
    if report["queries"] is not None:
        lines.append(f"  queries:  {report['queries']:,.0f}")
        lines.append(f"  hits:     {(report['hits'] or 0):,.0f}")
    if report["hit_rate"] is not None:
        lines.append(f"  hit rate: {report['hit_rate']:.1%}")
        if report["hit_rate"] < 0.5:
            # Every agent turn re-sends the whole conversation plus one new
            # tool result, so the prefix is shared by construction.  A low rate
            # means something is breaking it — a prompt whose head changes per
            # request, or eviction under memory pressure.
            lines.append("  note:     low for an agent workload; the prompt "
                         "prefix is probably changing between turns")
    return "\n".join(lines)


def main(argv: list) -> int:
    host = (argv[1] if len(argv) > 1
            else os.environ.get("ONIT_HOST") or "http://localhost:8000/v1")
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
    report = asyncio.run(prefix_cache_report(host, api_key))
    print(format_report(host, report))
    return 0 if report["reachable"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
