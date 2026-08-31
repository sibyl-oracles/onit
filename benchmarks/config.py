"""Tier definitions and OnIt agent configuration for benchmarks.

A *tier* is just a preset of (sample limit, concurrency). The benchmark task
code is tier-agnostic; the tier is applied at run time by ``run.py``.

The eval target (model/host) is resolved the same way the agent resolves its
own: the benchmark env vars win, then the agent's ``~/.onit/config.yaml``
(written by ``onit setup`` — its preferred endpoint, model, and the key it
authenticates with from the OS keychain), then ``ONIT_HOST``, then the
local-vLLM default. The same task code therefore benchmarks whatever endpoint
the agent actually uses — local vLLM, a LAN host, Ollama cloud, or OpenRouter
— without edits.
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
# docs/RUN_A_MODEL_SERVER.md). Used only when neither the benchmark env vars
# nor the agent's ~/.onit/config.yaml name an endpoint — resolve_serving()
# falls back to the agent's config first.
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
    """Build OnIt's ``serving`` config block for the benchmark run.

    Resolution order, mirroring how the agent resolves its own target:

    1. Benchmark env vars — ``ONIT_BENCH_HOST`` (then ``ONIT_HOST``) and
       ``ONIT_BENCH_MODEL``. Explicit env wins over everything, so a
       benchmark can always be pointed at a different endpoint than the one
       the agent chats through.
    2. The agent's ``~/.onit/config.yaml`` (written by ``onit setup``) — its
       preferred endpoint, model, and the key it authenticates with. The key
       itself is *not* read here; the agent's chat layer resolves
       ``host_key: EMPTY`` through the same keychain lookup it uses in
       conversation, so a benchmark run authenticates exactly like the agent
       does. Only the pin (``ONIT_BENCH_MODEL``) overrides the config's model,
       keeping unannotated runs comparable.
    3. The local-vLLM default host, with auto-detection from ``/v1/models``.
    """
    host = (os.environ.get("ONIT_BENCH_HOST")
            or _agent_config_serving().get("host")
            or os.environ.get("ONIT_HOST")
            or DEFAULT_HOST)
    serving: dict = {"host": host, "think": _env_bool("ONIT_BENCH_THINK", False)}

    model = os.environ.get("ONIT_BENCH_MODEL")
    if model is None:
        # No explicit pin: keep the agent's own model choice if it has one,
        # else fall back to the benchmark pin. The pin exists so unannotated
        # runs land on a fixed, comparable target; the agent's config wins
        # because the point of this resolution order is to benchmark the
        # endpoint the agent actually uses, as configured.
        model = _agent_config_serving().get("model") or DEFAULT_MODEL
    if model:  # empty string = defer to the agent's /v1/models auto-detection
        serving["model"] = model

    # host_key is optional; OnIt also reads OPENROUTER_API_KEY / OLLAMA_API_KEY
    # from env or keychain. Pass it through only when explicitly set.
    host_key = os.environ.get("ONIT_BENCH_HOST_KEY")
    if host_key:
        serving["host_key"] = host_key

    return serving


def _agent_config_serving() -> dict:
    """The ``serving`` block of the agent's own config, or {} when absent.

    Reads ``~/.onit/config.yaml`` — the file ``onit setup`` writes — through
    the agent's own loader so path handling and YAML quirks stay in one place.
    When that config lists several endpoints, the preferred one is chosen by
    the agent's real LoadBalancer rule (explicit priority, else non-Ollama
    over fallback-only Ollama, else first listed), so the benchmark cannot
    drift from the endpoint the agent would actually pick. Never raises: a
    missing or unreadable config just means "no agent config to inherit".
    """
    global _AGENT_SERVING_CACHE
    if _AGENT_SERVING_CACHE is not None:
        return _AGENT_SERVING_CACHE
    serving: dict = {}
    try:
        from src.setup import CONFIG_PATH, _load_config

        config = _load_config() if os.path.isfile(CONFIG_PATH) else {}
        block = config.get("serving") or {}
        if isinstance(block, dict):
            serving = block
            entries = block.get("endpoints")
            if isinstance(entries, list) and entries:
                serving = dict(block)
                chosen = _preferred_endpoint(entries, block)
                if chosen:
                    serving["host"] = chosen["host"]
                    # The agent serves the chosen endpoint's own model when
                    # the entry names one; a block-level model (legacy
                    # single-server shape) would not survive the endpoints
                    # list, so the entry is the faithful thing to inherit.
                    if chosen.get("model"):
                        serving["model"] = chosen["model"]
    except Exception:
        # No config, unreadable config, or agent modules unavailable (e.g.
        # benchmarks vendored elsewhere): fall through to env/default target.
        serving = {}
    _AGENT_SERVING_CACHE = serving
    return serving


# Memoized agent-config serving block. The config file changes only when
# `onit setup` runs, and re-reading it per resolve_serving() call would put a
# keychain touch on every model_label(); tests reset this via the fixture.
_AGENT_SERVING_CACHE: dict | None = None


def _preferred_endpoint(entries: list, serving: dict) -> dict | None:
    """The endpoint entry the agent's LoadBalancer would prefer.

    Mirrors LoadBalancer.preferred (src/model/serving/balancer.py): explicit
    priority (lower wins), else the first non-Ollama endpoint while the
    implicit fallback-only rule is in effect, else the first entry. Reuses the
    balancer itself where it is importable so the rule cannot drift; falls
    back to a local copy of the rule only when src is not importable.
    """
    normalized: list[tuple[int, bool, dict]] = []
    for entry in entries:
        if isinstance(entry, str):
            entry = {"host": entry}
        if not isinstance(entry, dict):
            continue
        host = str(entry.get("host") or "").strip()
        if not host:
            continue
        try:
            priority = int(entry.get("priority", 0))
        except (TypeError, ValueError):
            priority = 0
        is_ollama = any(f in host for f in ("ollama.com", "ollama.ai", ":11434"))
        normalized.append((priority, is_ollama, dict(entry, host=host)))
    if not normalized:
        return None
    try:
        from src.model.serving.balancer import LoadBalancer, ServerEndpoint

        endpoints = [ServerEndpoint(host=h, priority=p) for p, _, h in normalized]
        fallback_only = bool(serving.get("ollama_fallback_only", True))
        preferred = LoadBalancer(
            endpoints, serving.get("load_balancer", "sticky"),
            ollama_fallback_only=fallback_only).preferred
        for _, _, entry in normalized:
            if entry["host"] == preferred.host:
                return entry
    except Exception:
        pass
    # Local copy of the rule (src not importable): explicit priority first,
    # else non-Ollama while fallback-only, else first listed.
    if any(p != 0 for p, _, _ in normalized):
        return min(normalized, key=lambda t: t[0])[2]
    if not serving.get("ollama_fallback_only", True) or all(o for _, o, _ in normalized):
        return normalized[0][2]
    for _, is_ollama, entry in normalized:
        if not is_ollama:
            return entry
    return normalized[0][2]


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

    Authentication mirrors the agent's own request path
    (``src/model/serving/chat.py::_resolve_api_key``): the explicit benchmark
    key, then the key stored for this endpoint's URL by ``onit setup`` (OS
    keychain), then the provider-named legacy secret. A local vLLM started
    without ``--api-key`` needs none of these.

    Skipped for paid hosted endpoints (Ollama cloud, OpenRouter), where a
    probe would spend a paid API call and 401 semantics differ per provider.
    Everything else — localhost or the agent's configured remote host — is
    probed, since a probe of your own model server is free.
    """
    import urllib.error
    import urllib.request

    serving = resolve_serving()
    host = serving["host"]
    if any(p in host for p in ("ollama.com", "ollama.ai", "openrouter.ai")):
        return

    url = host.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    url += "/models"

    request = urllib.request.Request(url)
    api_key = _endpoint_api_key(host, serving)
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
        # The agent's config may name a model the endpoint no longer serves
        # (redeployed server); the agent would auto-detect a fallback at
        # request time, but a benchmark needs the target fixed before the run.
        sys.exit(
            f"[bench] model {model!r} is not served by {host}.\n"
            f"[bench] served: {', '.join(served)}\n"
            f"[bench] run the server for the pin, or override with "
            f"ONIT_BENCH_MODEL (set it to \"\" to auto-detect)."
        )


def _endpoint_api_key(host: str, serving: dict) -> str:
    """The key the agent itself would authenticate this endpoint with.

    Mirrors ``src/model/serving/chat.py::_resolve_api_key``: explicit key,
    then the per-endpoint key stored by ``onit setup`` (env-injected slug or
    OS keychain), then the provider-named legacy secret its URL selects.
    Returns "EMPTY" when nothing is stored — correct for a local vLLM started
    without ``--api-key``, which accepts any bearer token.
    """
    explicit = serving.get("host_key") or os.environ.get("ONIT_BENCH_HOST_KEY")
    if explicit:
        return explicit
    try:
        from src.model.serving.chat import _resolve_api_key

        return _resolve_api_key(host, "EMPTY")
    except Exception:
        pass
    # Local copy of the rule (src not importable): per-endpoint key, then the
    # provider-named secret. Order matches LEGACY_ENDPOINT_KEYS.
    try:
        from src.setup import get_endpoint_key, get_secret, legacy_key_for

        key = get_endpoint_key(host)
        if key:
            return key
        keyring_key, env_var, _, _ = legacy_key_for(host)
        return (os.environ.get(env_var) or get_secret(keyring_key) or "EMPTY")
    except Exception:
        return "EMPTY"


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")
