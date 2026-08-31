"""The event log: one file per session, two kinds of loop writing to it.

What is being protected here.  That lifecycle events land in the *same* store
as the trajectories they are about — two logging systems that agree about what
happened is the entire point (SELF_IMPROVEMENT.md §4.4) — and that a new
record kind cannot silently corrupt the task summary the CLI prints.  Plus the
lifecycle policy: thresholds that propose, never act, and a deletion that is
always soft.
"""

import json

import pytest

from learn import config as learn_config
from learn import events as ev
from learn import lifecycle as lc
from learn import trajectory as traj


@pytest.fixture
def learn_config_data(tmp_path):
    """Config with recording on, writing under tmp_path rather than ~/.onit."""
    return {"learn": {"autonomy": "observe", "path": str(tmp_path / "learned")}}


def _task(session_id, turn, ts, tools=None, config=None):
    """One task record with a fixed timestamp and optional tool calls."""
    turns = [{"n": 1, "tools": tools or [], "tool_runs": tools or []}]
    record = traj.build_record(session_id=session_id, turn=turn, task="t",
                               response="r", metrics={"turns": turns})
    record["ts"] = ts
    path = traj.session_file(session_id, config)
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


class TestEventStore:
    """Events live beside trajectories, in the same file, and stay separate."""

    def test_event_lands_in_the_session_file(self, learn_config_data):
        path = ev.record_event(event="tool.loaded", subject="pdf_extract",
                               session_id="s1", config_data=learn_config_data)
        assert path and path.endswith("s1.jsonl")
        with open(path) as f:
            record = json.loads(f.read().strip())
        assert record["kind"] == "event"
        assert record["event"] == "tool.loaded"
        assert record["subject"] == "pdf_extract"

    def test_read_session_ignores_events(self, learn_config_data):
        """The trajectory reader folds ratings in and leaves events alone."""
        ev.record_event(event="tool.loaded", subject="x",
                        session_id="s1", config_data=learn_config_data)
        traj.record_task(session_id="s1", turn=1, task="t", response="r",
                         config_data=learn_config_data)
        records = traj.read_session("s1", learn_config_data)
        assert len(records) == 1
        assert records[0]["kind"] == "task"

    def test_summary_does_not_count_events_as_tasks(self, learn_config_data):
        """A new record kind must not inflate the CLI's task counts."""
        for _ in range(5):
            ev.record_event(event="tool.reloaded", subject="registry",
                            session_id="s1", config_data=learn_config_data)
        traj.record_task(session_id="s1", turn=1, task="t", response="r",
                         config_data=learn_config_data)
        from learn.report import summarize
        s = summarize(learn_config_data)
        assert s["tasks"] == 1

    def test_off_records_nothing(self, tmp_path):
        cfg = {"learn": {"autonomy": "off", "path": str(tmp_path / "l")}}
        assert ev.record_event(event="tool.loaded", subject="x",
                               session_id="s1", config_data=cfg) is None

    def test_orphan_events_get_their_own_file(self, learn_config_data):
        """A gardener pass belongs to no conversation and must not invent one."""
        path = ev.record_event(event="tool.archived", subject="x",
                               session_id=None, config_data=learn_config_data)
        assert path and path.endswith("loop.jsonl")

    def test_iter_events_filters(self, learn_config_data):
        ev.record_event(event="tool.loaded", subject="a",
                        session_id="s1", config_data=learn_config_data)
        ev.record_event(event="tool.archived", subject="a",
                        session_id="s1", config_data=learn_config_data)
        ev.record_event(event="tool.loaded", subject="b",
                        session_id="s2", config_data=learn_config_data)
        loaded = [r for r in ev.iter_events(learn_config_data,
                                            event="tool.loaded")]
        assert len(loaded) == 2
        just_a = [r for r in ev.iter_events(learn_config_data,
                                            event="tool.loaded", subject="a")]
        assert len(just_a) == 1

    def test_tool_timeline_is_ordered_and_complete(self, learn_config_data):
        for event in ("tool.proposed", "tool.loaded", "tool.updated",
                      "tool.archived"):
            ev.record_tool_event(event=event, tool="pdf_merge",
                                 session_id="s1",
                                 config_data=learn_config_data)
        timeline = ev.tool_timeline("pdf_merge", learn_config_data)
        assert [r["event"] for r in timeline] == [
            "tool.proposed", "tool.loaded", "tool.updated", "tool.archived"]

    def test_summarize_events_counts(self, learn_config_data):
        ev.record_event(event="tool.loaded", subject="a",
                        session_id="s1", config_data=learn_config_data)
        ev.record_event(event="tool.loaded", subject="b",
                        session_id="s1", config_data=learn_config_data)
        s = ev.summarize_events(learn_config_data)
        assert s["total"] == 2
        assert s["by_event"] == {"tool.loaded": 2}
        assert s["subjects"] == {"tool.loaded": 2}

    def test_torn_line_loses_one_event_not_the_file(self, learn_config_data):
        path = traj.session_file("s1", learn_config_data)
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write('{"kind": "event", "event": "tool.loaded", "subject": "a"\n')
            f.write('{"kind": "event", "event": "tool.loaded", "subject": "b"}\n')
        assert len(list(ev.iter_events(learn_config_data))) == 1


class TestCreationPolicy:
    """Recurrence across sessions is the only honest creation evidence."""

    def test_below_thresholds_refuses(self):
        v = lc.should_propose_creation(occurrences=2, sessions=1)
        assert not v["propose"]
        assert len(v["reasons"]) == 2

    def test_recurrence_across_sessions_passes(self):
        v = lc.should_propose_creation(occurrences=3, sessions=2)
        assert v["propose"] and not v["reasons"]

    def test_category_cap_blocks_narrow_drift(self):
        """A week of PDF tasks must not bequeath fifteen PDF tools."""
        v = lc.should_propose_creation(occurrences=5, sessions=3,
                                       category_share=0.6)
        assert not v["propose"]
        assert any("category" in r for r in v["reasons"])

    def test_category_cap_leaves_room(self):
        v = lc.should_propose_creation(occurrences=5, sessions=3,
                                       category_share=0.2)
        assert v["propose"]


class TestUpdatePolicy:
    """Measured misbehavior, not a complaint."""

    def test_below_call_floor_is_noise(self):
        v = lc.should_update(calls=2, errors=2)
        assert not v["update"]

    def test_high_failure_rate_triggers(self):
        v = lc.should_update(calls=10, errors=4)
        assert v["update"]

    def test_healthy_tool_is_left_alone(self):
        v = lc.should_update(calls=50, errors=2)
        assert not v["update"]


class TestArchivePolicy:
    """Deletion is non-earnings, and always soft."""

    def test_never_called_tool_is_due(self):
        v = lc.should_archive(tool="x", last_used_ts=None, calls=0, errors=0)
        assert v["archive"] and "never called" in v["reasons"]

    def test_recently_used_tool_stays(self, learn_config_data):
        _task("s1", 1, "2026-08-30T10:00:00+00:00",
              config=learn_config_data)
        v = lc.should_archive(tool="x", last_used_ts="2026-08-30T10:00:00+00:00",
                              calls=10, errors=0,
                              config_data=learn_config_data)
        assert not v["archive"]

    def test_long_idle_tool_is_due(self, learn_config_data):
        """Twenty episodes of tasks have completed since its last call."""
        for i in range(31):
            _task("s1", i + 1, f"2026-08-{1 + i // 2:02d}T10:00:00+00:00",
                  config=learn_config_data)
        v = lc.should_archive(tool="x", last_used_ts="2026-08-01T09:00:00+00:00",
                              calls=10, errors=0,
                              config_data=learn_config_data)
        assert v["archive"]
        assert any("episodes" in r for r in v["reasons"])

    def test_negative_value_retires_early(self):
        v = lc.should_archive(tool="x", last_used_ts=None, calls=10, errors=8)
        assert v["archive"]
        assert any("failure rate" in r for r in v["reasons"])

    def test_episodes_since_counts_tasks(self, learn_config_data):
        for i in range(4):
            _task("s1", i + 1, f"2026-08-0{i + 1}T10:00:00+00:00",
                  config=learn_config_data)
        assert lc.episodes_since("2026-08-02T00:00:00+00:00",
                                 learn_config_data) == 3


class TestPendingProposals:
    """The gardener's batched input, read off the store."""

    def test_failing_tool_surfaces(self, learn_config_data):
        tools = [{"name": "flaky", "ok": False}] * 4 + \
                [{"name": "flaky", "ok": True}] * 6
        _task("s1", 1, "2026-08-30T10:00:00+00:00", tools=tools,
              config=learn_config_data)
        proposals = lc.pending_proposals(learn_config_data)
        assert any(p["tool"] == "flaky" and p["action"] == "update"
                   for p in proposals)

    def test_loaded_but_never_called_surfaces(self, learn_config_data):
        ev.record_tool_event(event="tool.loaded", tool="dead_weight",
                             session_id="s1", config_data=learn_config_data)
        proposals = lc.pending_proposals(learn_config_data)
        assert any(p["tool"] == "dead_weight" and p["action"] == "review"
                   for p in proposals)

    def test_recording_off_yields_nothing(self, tmp_path):
        cfg = {"learn": {"autonomy": "off", "path": str(tmp_path / "l")}}
        assert lc.pending_proposals(cfg) == []


class TestHarnessEmitter:
    """The registry load is the first real event the store receives."""

    def test_registry_load_writes_event(self, learn_config_data, monkeypatch):
        from type.tools import ToolHandler, ToolRegistry

        registry = ToolRegistry()
        handler = ToolHandler(url="stdio://test", tool_item={
            "type": "function",
            "function": {"name": "echo", "description": "d",
                         "parameters": {"type": "object", "properties": {}}}})
        registry.register(handler)

        class FakeOnIt:
            session_id = "sess-1"
            config_data = learn_config_data
            mcp_servers = []

        fake = FakeOnIt()
        fake.tool_registry = registry
        # Re-run just the emission block by calling the real method body via
        # a stubbed discover: the point is the event, not the discovery.
        from src.onit import OnIt  # noqa: F401  (import proves wiring exists)

        async def fake_discover(servers):
            return registry

        monkeypatch.setattr("src.onit.discover_tools", fake_discover)
        monkeypatch.setattr("builtins.print", lambda *a, **k: None)
        OnIt._setup_tool_registry(fake)
        events = list(ev.iter_events(learn_config_data,
                                     event="tool.registry_loaded"))
        assert len(events) == 1
        assert events[0]["detail"]["tools"] == ["echo"]
        assert events[0]["source"] == "harness"

    def test_emitter_failure_never_breaks_setup(self, learn_config_data,
                                                monkeypatch):
        from type.tools import ToolRegistry

        registry = ToolRegistry()

        class FakeOnIt:
            session_id = "sess-1"
            config_data = learn_config_data
            mcp_servers = []

        fake = FakeOnIt()
        fake.tool_registry = registry

        async def fake_discover(servers):
            return registry

        def boom(*a, **k):
            raise RuntimeError("disk on fire")

        monkeypatch.setattr("src.onit.discover_tools", fake_discover)
        monkeypatch.setattr("builtins.print", lambda *a, **k: None)
        monkeypatch.setattr("learn.events.record_event", boom)
        from src.onit import OnIt
        OnIt._setup_tool_registry(fake)  # must not raise