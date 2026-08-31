"""Tier definitions and OnIt agent configuration for benchmarks.

A *tier* is just a preset of (sample limit, concurrency). The benchmark task
code is tier-agnostic; the tier is applied at run time by ``run.py``.

The eval target (model/host) is resolved from environment variables so the same
task code can run against a local vLLM endpoint (the default — the agent's own
serving setup), Ollama cloud, or OpenRouter without edits.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Tier:
    """A benchmark run preset.

    Attributes:
        name: Tier identifier (``smoke`` / ``sampled`` / ``full``).
        limit: Max samples per benchmark. ``None`` means the whole dataset.
        max_connections: Concurrent samples in flight (agent calls are serial
            per sample but multiple samples run in parallel).
    """

    name: str
    limit: int | None
    max_connections: int


TIERS: dict[str, Tier] = {
    # CI gate: a handful of samples per benchmark, just enough to catch breakage.
    "smoke": Tier(name="smoke", limit=5, max_connections=2),
    # Nightly: a fixed, seeded subset that gives a tracked capability signal.
    "sampled": Tier(name="sampled", limit=100, max_connections=4),
    # On-demand: the complete dataset, leaderboard-comparable.
    "full": Tier(name="full", limit=None, max_connections=8),
}

DEFAULT_TIER = "smoke"

# Default eval target: the agent's own vLLM serving setup (see
# docs/RUN_A_MODEL_SERVER.md). Override for hosted endpoints.
DEFAULT_HOST = "http://localhost:8000/v1"

# Pinned benchmark model. RESULTS.md model-selection evidence: Qwen3.8-27B
# (released Aug 2026) is the strongest open-weight ~27B coder available and
# keeps every benchmark row on one comparable model. Override per run with
# ONIT_BENCH_MODEL; the pin exists so unannotated runs land on a fixed target
# and their rows stay comparable to the committed baseline. An empty value
# ("") defers to the agent's auto-detection (first entry of /v1/models) —
# use it when the server was launched with a different model than the pin.
DEFAULT_MODEL = "Qwen/Qwen3.8-27B"

# Alias (CLI / Inspect task name) -> canonical benchmark display name + category.
# The alias is what you pass to ``--tasks`` and what Inspect logs/reports show.
BENCHMARKS: dict[str, tuple[str, str]] = {
    "gsm8k": ("GSM8K", "reasoning"),
    "humaneval": ("HumanEval", "coding"),
    "mbpp": ("MBPP", "coding"),
    "bigcodebench": ("BigCodeBench", "coding"),
    "metr": ("METR Time Horizon", "long-horizon"),
}


def display_name(alias: str) -> str:
    """Return the canonical benchmark name for an alias (or the alias itself)."""
    entry = BENCHMARKS.get(alias)
    return entry[0] if entry else alias


def resolve_serving() -> dict:
    """Build OnIt's ``serving`` config block from environment variables.

    Precedence mirrors OnIt itself: explicit env wins, else the local-vLLM
    default host. ``ONIT_BENCH_MODEL`` pins the model (required for Ollama
    cloud and OpenRouter; for vLLM it is verified against the endpoint by
    ``run.py``'s preflight, and auto-detection is the fallback when unset).
    """
    host = os.environ.get("ONIT_BENCH_HOST") or os.environ.get("ONIT_HOST") or DEFAULT_HOST
    serving: dict = {"host": host, "think": _env_bool("ONIT_BENCH_THINK", False)}

    model = os.environ.get("ONIT_BENCH_MODEL")
    if model is None:
        model = DEFAULT_MODEL
    if model:  # empty string = defer to the agent's /v1/models auto-detection
        serving["model"] = model

    # host_key is optional; OnIt also reads OPENROUTER_API_KEY / OLLAMA_API_KEY
    # from env or keychain. Pass it through only when explicitly set.
    host_key = os.environ.get("ONIT_BENCH_HOST_KEY")
    if host_key:
        serving["host_key"] = host_key

    return serving


def model_label() -> str:
    """Human-readable label for the model under test, for logs and reports."""
    serving = resolve_serving()
    return serving.get("model") or serving["host"]


def learn_level() -> str:
    """Autonomy level the agent under test runs at.

    Defaults to ``off``, not to whatever the operator's config file says. A
    benchmark row is only comparable to another row if the two agents were
    allowed to change themselves by the same amount, and the useful default is
    the one that measures the scaffold as shipped. Set ``ONIT_LEARN`` to
    measure a learning level on purpose — that is the A/B this exists for.
    """
    return os.environ.get("ONIT_LEARN") or "off"


def resolve_learn() -> dict:
    """OnIt's ``learn`` config block for a benchmark run."""
    return {"autonomy": learn_level()}


def bench_timeout() -> int:
    """Per-request timeout (seconds) for the agent under test.

    Bounded by default so a stalled endpoint fails the sample instead of hanging
    the whole run forever. Override with ``ONIT_BENCH_TIMEOUT`` (``-1`` disables,
    use only when you know the endpoint is reliable).
    """
    raw = os.environ.get("ONIT_BENCH_TIMEOUT", "300")
    try:
        return int(raw)
    except ValueError:
        return 300


def bench_data_root() -> Path:
    """Server-wide jail root for benchmark runs.

    This is what OnIt's ``data_path`` config is set to, which the MCP servers
    read as their ``DATA_PATH`` (via ``ONIT_DATA_PATH``). Every per-sample
    ``data_path`` handed to ``process_task`` **must** be this directory or a
    descendant of it, or ``_session_base`` rejects the call with
    ``"data_path must be within the server data directory"`` and every file tool
    fails. Callers that mint their own working directories (per-sample dirs, the
    SWE-bench per-instance workspaces) build them under this root.
    """
    return Path(tempfile.gettempdir()) / "onit-bench-data"


def preflight_endpoint(timeout: float = 10.0) -> None:
    """Fail fast when the eval endpoint is unreachable or serves the wrong model.

    A benchmark run is long; discovering after an hour that the vLLM server was
    never started (or was redeployed with a different model than the pin) wastes
    all of it. This checks the endpoint's ``/v1/models`` before any samples run:

    * connection refused / timeout -> exit with the launch command from
      docs/RUN_A_MODEL_SERVER.md,
    * pinned model not served -> exit naming what the endpoint *does* serve,
      with the override to proceed anyway,
    * no model configured (empty ``ONIT_BENCH_MODEL``) -> adopt the endpoint's
      first model, mirroring the agent's own auto-detection.

    Skipped entirely for hosted endpoints (anything not on localhost), where a
    probe would spend a paid API call and 401 semantics differ per provider.
    """
    import urllib.error
    import urllib.request

    serving = resolve_serving()
    host = serving["host"]
    if "localhost" not in host and "127.0.0.1" not in host:
        return

    url = host.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    url += "/models"

    request = urllib.request.Request(url)
    api_key = os.environ.get("VLLM_API_KEY") or serving.get("host_key") or "EMPTY"
    request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        sys.exit(
            f"[bench] eval endpoint {host} is unreachable: {e}\n"
            f"[bench] start the vLLM server first (docs/RUN_A_MODEL_SERVER.md):\n"
            f"[bench]   vllm serve {DEFAULT_MODEL} --port 8000 \\\n"
            f"[bench]     --max-model-len 262144 --enable-auto-tool-choice \\\n"
            f"[bench]     --tool-call-parser hermes --reasoning-parser qwen3 \\\n"
            f"[bench]     --chat-template-content-format string \\\n"
            f"[bench]     --enable-prefix-caching\n"
            f"[bench] or point ONIT_BENCH_HOST at a running endpoint."
        )

    served = sorted(
        str(m.get("id")) for m in payload.get("data", []) if m.get("id")
    )
    if not served:
        sys.exit(f"[bench] endpoint {host} answered but serves no models.")

    model = serving.get("model") or ""
    if not model:
        # Same rule as the agent: a single-model server needs no explicit name.
        print(f"[bench] no model set; auto-detected {served[0]!r} from {host}")
        os.environ["ONIT_BENCH_MODEL"] = served[0]
        return
    if model not in served:
        sys.exit(
            f"[bench] pinned model {model!r} is not served by {host}.\n"
            f"[bench] served: {', '.join(served)}\n"
            f"[bench] run the server for the pin, or override with "
            f"ONIT_BENCH_MODEL (set it to \"\" to auto-detect)."
        )


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")
