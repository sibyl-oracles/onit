"""Unit tests for the OnIt Inspect provider — no model, network, or Docker.

A stub agent stands in for the real ``OnIt`` so we can verify the provider's
message flattening, the generate() path, and the full Inspect eval+scorer wiring
deterministically.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model
from inspect_ai.scorer import match

from benchmarks import onit_provider
from benchmarks.onit_provider import _messages_to_task
from benchmarks.tasks.coding import _extract_code


class _StubAgent:
    """Fake OnIt whose process_task returns a canned answer."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls: list[str] = []

    async def process_task(self, task: str, **kwargs) -> str:
        self.calls.append(task)
        return self.answer


@pytest.fixture(autouse=True)
def _reset_agent():
    """Ensure each test controls the shared agent singleton and the cap."""
    onit_provider._agent = None
    onit_provider._concurrency_sem = None
    yield
    onit_provider._agent = None
    onit_provider._concurrency_sem = None


def test_messages_to_task_flattens_system_and_user():
    msgs = [ChatMessageSystem(content="Be precise."), ChatMessageUser(content="2+2?")]
    assert _messages_to_task(msgs) == "Be precise.\n\n2+2?"


def test_extract_code_prefers_fenced_block():
    answer = "Here you go:\n```python\ndef f():\n    return 1\n```\nDone."
    assert _extract_code(answer) == "def f():\n    return 1"


def test_extract_code_falls_back_to_raw():
    assert _extract_code("def f():\n    return 1") == "def f():\n    return 1"


@pytest.mark.asyncio
async def test_generate_returns_agent_answer():
    onit_provider._agent = _StubAgent("ANSWER: 42")
    model = get_model("onit/stub")
    out = await model.generate("ignored")
    assert "42" in out.completion


@pytest.mark.asyncio
async def test_generate_caps_concurrency_when_env_set(monkeypatch):
    """ONIT_BENCH_MAX_CONNECTIONS caps agent requests in flight."""
    import asyncio

    monkeypatch.setenv("ONIT_BENCH_MAX_CONNECTIONS", "1")

    class _SlowAgent(_StubAgent):
        def __init__(self):
            super().__init__("ok")
            self.in_flight = 0
            self.max_in_flight = 0

        async def process_task(self, task: str, **kwargs) -> str:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            await asyncio.sleep(0.05)
            self.in_flight -= 1
            return self.answer

    agent = _SlowAgent()
    onit_provider._agent = agent
    model = get_model("onit/stub")
    # Fire 3 concurrent generations; the cap of 1 must serialize them.
    results = await asyncio.gather(model.generate("a"), model.generate("b"), model.generate("c"))
    assert all("ok" in r.completion for r in results)
    assert agent.max_in_flight == 1


@pytest.mark.asyncio
async def test_generate_no_cap_when_env_unset(monkeypatch):
    """Without the env var, generations run fully concurrent."""
    import asyncio

    monkeypatch.delenv("ONIT_BENCH_MAX_CONNECTIONS", raising=False)

    class _SlowAgent(_StubAgent):
        def __init__(self):
            super().__init__("ok")
            self.in_flight = 0
            self.max_in_flight = 0

        async def process_task(self, task: str, **kwargs) -> str:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            await asyncio.sleep(0.05)
            self.in_flight -= 1
            return self.answer

    agent = _SlowAgent()
    onit_provider._agent = agent
    model = get_model("onit/stub")
    results = await asyncio.gather(model.generate("a"), model.generate("b"), model.generate("c"))
    assert all("ok" in r.completion for r in results)
    assert agent.max_in_flight == 3


def test_eval_pipeline_scores_correct_answer():
    """End-to-end: provider + numeric match scorer + Inspect runner."""
    onit_provider._agent = _StubAgent("Working... ANSWER: 42")
    task = Task(
        dataset=MemoryDataset([Sample(input="What is 6 times 7?", target="42")]),
        scorer=match(location="any", numeric=True),
    )
    logs = inspect_eval(task, model="onit/stub", display="none")
    assert logs[0].status == "success"
    accuracy = logs[0].results.scores[0].metrics["accuracy"].value
    assert accuracy == 1.0


# ─── Serving-target resolution: agent config / env / keychain ────────────────


@pytest.fixture()
def _no_agent_config(monkeypatch):
    """Point the agent-config reader at an empty config and reset the cache."""
    import benchmarks.config as bench_config

    monkeypatch.setattr(bench_config, "_AGENT_SERVING_CACHE", None)
    monkeypatch.setattr("onit.setup._load_config", lambda: {})
    monkeypatch.setattr("onit.setup.CONFIG_PATH", "/nonexistent/config.yaml")
    yield
    bench_config._AGENT_SERVING_CACHE = None


def test_env_host_overrides_agent_config(monkeypatch):
    import benchmarks.config as bench_config

    monkeypatch.setattr(
        bench_config,
        "_AGENT_SERVING_CACHE",
        {"host": "http://agent-host:8000/v1", "model": "agent/model"},
    )
    monkeypatch.setenv("ONIT_BENCH_HOST", "http://env-host:8000/v1")
    serving = bench_config.resolve_serving()
    assert serving["host"] == "http://env-host:8000/v1"
    # The agent's model survives unless the benchmark pins its own.
    assert serving["model"] == "agent/model"


def test_agent_config_host_used_when_env_unset(monkeypatch):
    import benchmarks.config as bench_config

    monkeypatch.setattr(
        bench_config,
        "_AGENT_SERVING_CACHE",
        {"host": "http://agent-host:8000/v1", "model": "agent/model"},
    )
    monkeypatch.delenv("ONIT_BENCH_HOST", raising=False)
    monkeypatch.delenv("ONIT_HOST", raising=False)
    assert bench_config.resolve_serving()["host"] == "http://agent-host:8000/v1"


def test_agent_config_preferred_endpoint_rule(monkeypatch):
    """Priority, then non-Ollama-over-fallback-Ollama, then first listed."""
    import benchmarks.config as bench_config

    entries = [
        {"host": "https://api.ollama.com", "model": "glm-5.3-flash:cloud", "priority": 1},
        {"host": "http://10.0.0.2:8000/v1"},
        {"host": "http://10.0.0.1:8001/v1"},
    ]
    # Priority 0 entries: first non-Ollama wins.
    chosen = bench_config._preferred_endpoint(entries, {})
    assert chosen["host"] == "http://10.0.0.2:8000/v1"
    # Explicit priority outranks the implicit rule.
    chosen = bench_config._preferred_endpoint(
        [dict(e, priority=0) if e["host"].startswith("http://10") else e for e in entries], {}
    )
    assert chosen["host"] == "http://10.0.0.2:8000/v1"
    # The Ollama entry's own model is inherited when it is preferred.
    only_ollama = [entries[0]]
    chosen = bench_config._preferred_endpoint(only_ollama, {})
    assert chosen["host"] == "https://api.ollama.com"
    assert chosen["model"] == "glm-5.3-flash:cloud"


def test_agent_config_model_inherited_and_pin_wins(monkeypatch):
    import benchmarks.config as bench_config

    monkeypatch.setattr(
        bench_config,
        "_AGENT_SERVING_CACHE",
        {"host": "http://agent-host:8000/v1", "model": "agent/model"},
    )
    monkeypatch.delenv("ONIT_BENCH_MODEL", raising=False)
    # No pin: the agent's own model is benchmarked.
    assert bench_config.resolve_serving()["model"] == "agent/model"
    # Explicit pin overrides it (comparability of unannotated runs).
    monkeypatch.setenv("ONIT_BENCH_MODEL", "Qwen/Qwen3.8-27B")
    assert bench_config.resolve_serving()["model"] == "Qwen/Qwen3.8-27B"
    # Empty string still means auto-detect from /v1/models.
    monkeypatch.setenv("ONIT_BENCH_MODEL", "")
    assert "model" not in bench_config.resolve_serving()


def test_missing_agent_config_falls_back_to_defaults(_no_agent_config, monkeypatch):
    import benchmarks.config as bench_config

    monkeypatch.delenv("ONIT_BENCH_HOST", raising=False)
    monkeypatch.delenv("ONIT_HOST", raising=False)
    monkeypatch.delenv("ONIT_BENCH_MODEL", raising=False)
    serving = bench_config.resolve_serving()
    assert serving["host"] == bench_config.DEFAULT_HOST
    assert serving["model"] == bench_config.DEFAULT_MODEL


def test_preflight_uses_agent_keychain_key(monkeypatch):
    """The probe authenticates with the key the agent itself would use."""
    import benchmarks.config as bench_config

    captured = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b'{"data": [{"id": "Qwen/Qwen3.8-27B"}]}'

    def fake_urlopen(request, timeout):
        captured["auth"] = request.get_header("Authorization")
        return _Resp()

    monkeypatch.setattr(
        bench_config,
        "_AGENT_SERVING_CACHE",
        {"host": "http://agent-host:8000/v1", "model": "Qwen/Qwen3.8-27B"},
    )
    monkeypatch.setattr(
        "onit.model.serving.chat._resolve_api_key", lambda host, key: "sk-from-keychain"
    )
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    bench_config.preflight_endpoint()
    assert captured["auth"] == "Bearer sk-from-keychain"


def test_preflight_skips_only_paid_hosts(monkeypatch):
    import benchmarks.config as bench_config

    for host in ("https://api.ollama.com", "https://openrouter.ai/api/v1"):
        monkeypatch.setattr(bench_config, "_AGENT_SERVING_CACHE", {"host": host})
        bench_config.preflight_endpoint()  # returns without probing
    # A remote self-hosted IP is probed, not skipped.
    monkeypatch.setattr(
        bench_config, "_AGENT_SERVING_CACHE", {"host": "http://202.0.113.7:8001/v1", "model": "m"}
    )
    with pytest.raises(SystemExit):
        monkeypatch.setattr(
            "urllib.request.urlopen", lambda req, timeout: (_ for _ in ()).throw(OSError("refused"))
        )
        bench_config.preflight_endpoint()


# ── what the provider publishes, and where it lands ────────────────────────
#
# The metrics are the point of the S1 columns in report.py, and two separate
# breaks kept them from ever arriving: the provider passed ``metadata`` to
# ``ModelOutput.from_content``, which takes no such argument, and report.py
# read ``sample.metadata``, where inspect_ai never puts a provider's output
# metadata.  Both are asserted here so neither can come back quietly.


@pytest.mark.asyncio
async def test_generate_publishes_metrics_without_raising():
    """``from_content`` has no ``metadata`` keyword; setting it afterwards is
    how the SDK's own providers do it."""
    onit_provider._agent = _StubAgent("ANSWER: 42")
    model = get_model("onit/stub")
    out = await model.generate("ignored")
    assert out.metadata is not None
    assert out.metadata["wall_s"] == 0.0
    assert "turn_count" in out.metadata


def test_report_reads_the_metadata_inspect_ai_actually_keeps():
    """A provider's metadata lands on ``sample.output.metadata``.  Reading
    ``sample.metadata`` left every speed column empty."""
    from benchmarks import report

    sample = SimpleNamespace(metadata={}, output=SimpleNamespace(metadata={"wall_s": 1.5}))
    assert report._sample_speed_metadata(sample) == {"wall_s": 1.5}


def test_report_falls_back_to_sample_metadata():
    """Logs written before the provider published on the output still render."""
    from benchmarks import report

    sample = SimpleNamespace(metadata={"wall_s": 2.0}, output=None)
    assert report._sample_speed_metadata(sample) == {"wall_s": 2.0}


def test_report_collects_one_entry_per_task():
    """``collect`` assigned ``results`` itself, so every log but the last was
    discarded and the caller got model/samples/metrics instead of task names."""
    from benchmarks import report

    log = SimpleNamespace(
        status="success",
        results=SimpleNamespace(
            total_samples=1,
            scores=[SimpleNamespace(name="accuracy", metrics={"accuracy": _Metric(1.0)})],
        ),
        eval=SimpleNamespace(task="humaneval", model="onit/stub"),
        samples=[],
    )
    with (
        patch.object(report, "list_eval_logs", return_value=["a", "b"]),
        patch.object(report, "read_eval_log", return_value=log),
    ):
        collected = report.collect("/nonexistent")
    assert list(collected) == ["humaneval"]
    assert collected["humaneval"]["metrics"]["accuracy/accuracy"] == 1.0


class _Metric:
    """A stand-in for inspect_ai's Metric, which carries its value."""

    def __init__(self, value: float) -> None:
        self.value = value
