"""Tests for src/ui/commands.py — the session's backslash commands."""

import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.model.serving.balancer import LoadBalancer, ServerEndpoint
from src.ui import commands


@pytest.fixture
def agent():
    """A stand-in for OnIt carrying only what the commands actually read."""
    endpoints = [
        ServerEndpoint(host="http://localhost:8000/v1", model="Qwen3-30B",
                       name="vllm-a"),
        ServerEndpoint(host="http://localhost:8001/v1", name="vllm-b",
                       priority=1),
    ]
    return SimpleNamespace(
        load_balancer=LoadBalancer(endpoints, "sticky"),
        model_serving={"host": "http://localhost:8000/v1",
                       "model": "Qwen3-30B"},
        session_id="sess-1",
        session_path="/tmp/sess.jsonl",
        data_path="/tmp/data",
        chat_ui=SimpleNamespace(model_name="Qwen3-30B"),
    )


def _models(names, error=None):
    """Patch the endpoint model listing with a fixed answer."""
    async def fake(host, host_key="EMPTY", timeout=15.0):
        if error:
            raise error
        return list(names)
    return patch("src.model.serving.chat.list_models", fake)


# ── parsing ─────────────────────────────────────────────────────────────────

class TestParse:
    def test_bare_command(self):
        assert commands.parse("\\help") == ("help", "")

    def test_command_with_argument(self):
        assert commands.parse("\\model Qwen3-30B") == ("model", "Qwen3-30B")

    def test_leading_and_trailing_space_is_ignored(self):
        assert commands.parse("  \\host  http://x/v1  ") == ("host", "http://x/v1")

    def test_case_insensitive(self):
        assert commands.parse("\\HELP") == ("help", "")

    def test_stop_aliases_are_recognised(self):
        assert commands.parse("\\exit") == ("exit", "")

    def test_plain_message_is_not_a_command(self):
        assert commands.parse("what is the weather") is None

    def test_empty_input(self):
        assert commands.parse("") is None

    def test_lone_unknown_word_is_reported_not_sent(self):
        """A typo is far likelier than a question that is one \\word."""
        assert commands.parse("\\hlep") == ("", "hlep")

    def test_latex_reaches_the_model(self):
        """A backslash line that is not a command must pass through: the
        alternative eats every LaTeX fragment a user types."""
        assert commands.parse("\\frac{a}{b} — simplify this") is None

    def test_unknown_word_with_more_text_passes_through(self):
        assert commands.parse("\\emph something something") is None


class TestSuggest:
    def test_near_miss_suggests(self):
        assert commands.suggest("hlep") == "help"

    def test_nothing_close_suggests_nothing(self):
        assert commands.suggest("zzzz") == ""

    def test_unknown_message_names_the_typo_and_the_fix(self):
        msg = commands.unknown("modle")
        assert "\\modle is not a command" in msg
        assert "\\model" in msg
        assert "\\help" in msg


# ── \help ───────────────────────────────────────────────────────────────────

class TestHelp:
    def test_every_command_is_listed(self):
        text = commands.render_help()
        for cmd in commands.COMMANDS:
            assert cmd.usage in text
            assert cmd.summary in text

    def test_says_other_lines_go_to_the_model(self):
        assert "sent to the model" in commands.render_help()


# ── \model ──────────────────────────────────────────────────────────────────

class TestModel:
    @pytest.mark.asyncio
    async def test_bare_lists_what_the_endpoint_serves(self, agent):
        with _models(["Qwen3-30B", "Llama-3-8B"]):
            out = await commands.cmd_model(agent, "")
        assert "Qwen3-30B" in out and "Llama-3-8B" in out
        assert "vllm-a" in out

    @pytest.mark.asyncio
    async def test_bare_reports_an_endpoint_that_cannot_be_reached(self, agent):
        with _models([], error=ConnectionError("refused")):
            out = await commands.cmd_model(agent, "")
        assert "Could not list models" in out
        assert "refused" in out

    @pytest.mark.asyncio
    async def test_long_listing_is_cut_short(self, agent):
        names = [f"m{i:03d}" for i in range(120)]
        with _models(names):
            out = await commands.cmd_model(agent, "")
        assert out.count("\n  m") == commands.MAX_LISTED_MODELS
        assert f"and {120 - commands.MAX_LISTED_MODELS} more" in out

    @pytest.mark.asyncio
    async def test_switch_sets_the_model_on_the_active_endpoint(self, agent):
        with _models(["Qwen3-30B", "Llama-3-8B"]):
            out = await commands.cmd_model(agent, "Llama-3-8B")
        assert agent.load_balancer.preferred.model == "Llama-3-8B"
        assert agent.model_serving["model"] == "Llama-3-8B"
        assert agent.chat_ui.model_name == "Llama-3-8B"
        assert "Llama-3-8B" in out

    @pytest.mark.asyncio
    async def test_unserved_model_is_set_but_flagged(self, agent):
        """The endpoint is the authority, but it may list less than it
        serves — so warn, don't refuse."""
        with _models(["Qwen3-30B"]):
            out = await commands.cmd_model(agent, "does-not-exist")
        assert agent.load_balancer.preferred.model == "does-not-exist"
        assert "Warning" in out

    @pytest.mark.asyncio
    async def test_unreachable_endpoint_does_not_block_the_switch(self, agent):
        with _models([], error=ConnectionError("refused")):
            out = await commands.cmd_model(agent, "Llama-3-8B")
        assert agent.load_balancer.preferred.model == "Llama-3-8B"
        assert "Could not verify" in out

    @pytest.mark.asyncio
    async def test_dash_restores_auto_detect(self, agent):
        from src.model.serving import chat as chat_mod
        chat_mod._MODEL_ID_CACHE["http://localhost:8000/v1"] = "stale"
        out = await commands.cmd_model(agent, "-")
        assert agent.load_balancer.preferred.model is None
        assert "model" not in agent.model_serving
        assert "http://localhost:8000/v1" not in chat_mod._MODEL_ID_CACHE
        assert "auto-detect" in out

    @pytest.mark.asyncio
    async def test_a_sticky_session_edits_its_own_endpoint(self, agent):
        """The change has to land where the next request will go, not on
        whichever endpoint happens to rank first."""
        second = agent.load_balancer.endpoints[1]
        agent.load_balancer._sticky_map["sess-1"] = 1
        with _models(["Llama-3-8B"]):
            out = await commands.cmd_model(agent, "Llama-3-8B")
        assert second.model == "Llama-3-8B"
        assert agent.load_balancer.endpoints[0].model == "Qwen3-30B"
        assert "vllm-b" in out

    @pytest.mark.asyncio
    async def test_editing_a_non_preferred_endpoint_leaves_the_mirror_alone(self, agent):
        """serving.model stands for the preferred host; a change to another
        endpoint must not rewrite it."""
        agent.load_balancer._sticky_map["sess-1"] = 1
        with _models(["Llama-3-8B"]):
            await commands.cmd_model(agent, "Llama-3-8B")
        assert agent.model_serving["model"] == "Qwen3-30B"


# ── \host ───────────────────────────────────────────────────────────────────

class TestHost:
    @pytest.mark.asyncio
    async def test_bare_lists_the_endpoints(self, agent):
        out = await commands.cmd_host(agent, "")
        assert "http://localhost:8000/v1" in out
        assert "http://localhost:8001/v1" in out
        assert "serves this session" in out

    @pytest.mark.asyncio
    async def test_switch_replaces_the_rotation(self, agent):
        out = await commands.cmd_host(agent, "http://gpu-2:8000/v1")
        assert agent.load_balancer.hosts == ["http://gpu-2:8000/v1"]
        assert agent.model_serving["host"] == "http://gpu-2:8000/v1"
        assert "auto-detect" in out

    @pytest.mark.asyncio
    async def test_switch_clears_a_model_that_belonged_to_the_old_host(self, agent):
        await commands.cmd_host(agent, "http://gpu-2:8000/v1")
        assert "model" not in agent.model_serving
        assert agent.chat_ui.model_name == ""

    @pytest.mark.asyncio
    async def test_model_can_be_given_alongside_the_url(self, agent):
        await commands.cmd_host(agent, "https://openrouter.ai/api/v1 google/gemini-2.5-pro")
        assert agent.load_balancer.preferred.model == "google/gemini-2.5-pro"
        assert agent.model_serving["model"] == "google/gemini-2.5-pro"

    @pytest.mark.asyncio
    async def test_dropping_a_rotation_is_stated(self, agent):
        out = await commands.cmd_host(agent, "http://gpu-2:8000/v1")
        assert "2 endpoints" in out
        assert "http://localhost:8001/v1" in out

    @pytest.mark.asyncio
    async def test_says_the_config_file_is_untouched(self, agent):
        out = await commands.cmd_host(agent, "http://gpu-2:8000/v1")
        assert "onit setup" in out

    @pytest.mark.asyncio
    async def test_a_bare_hostname_is_rejected_with_an_example(self, agent):
        out = await commands.cmd_host(agent, "gpu-2:8000")
        assert "http://" in out
        assert agent.load_balancer.hosts == ["http://localhost:8000/v1",
                                             "http://localhost:8001/v1"]

    @pytest.mark.asyncio
    async def test_the_algorithm_survives_the_switch(self, agent):
        await commands.cmd_host(agent, "http://gpu-2:8000/v1")
        assert agent.load_balancer.algorithm == "sticky"

    @pytest.mark.asyncio
    async def test_the_listing_advertises_add_and_rm(self, agent):
        out = await commands.cmd_host(agent, "")
        assert "\\host add" in out and "\\host rm" in out

    @pytest.mark.asyncio
    async def test_a_replacement_points_at_add(self, agent):
        """Dropping a rotation is worth a nudge toward the command that
        would have widened it instead."""
        out = await commands.cmd_host(agent, "http://gpu-2:8000/v1")
        assert "\\host add" in out


class TestHostAdd:
    @pytest.mark.asyncio
    async def test_add_widens_the_rotation(self, agent):
        await commands.cmd_host(agent, "add http://gpu-3:8000/v1")
        assert agent.load_balancer.hosts == ["http://localhost:8000/v1",
                                             "http://localhost:8001/v1",
                                             "http://gpu-3:8000/v1"]

    @pytest.mark.asyncio
    async def test_add_keeps_the_endpoints_that_were_there(self, agent):
        """The bug this fixes: \\host could only replace, so a second server
        could not be brought in without losing the first."""
        before = [ep.model for ep in agent.load_balancer.endpoints]
        await commands.cmd_host(agent, "add http://gpu-3:8000/v1")
        after = [ep.model for ep in agent.load_balancer.endpoints[:2]]
        assert after == before

    @pytest.mark.asyncio
    async def test_a_model_can_be_given_with_the_url(self, agent):
        await commands.cmd_host(agent, "add http://gpu-3:8000/v1 Llama-3-8B")
        assert agent.load_balancer.endpoints[-1].model == "Llama-3-8B"

    @pytest.mark.asyncio
    async def test_it_joins_the_preferred_tier(self, agent):
        """'Also use this one' means share the traffic, not sit in reserve
        behind everything already configured."""
        added = await commands.cmd_host(agent, "add http://gpu-3:8000/v1")
        assert agent.load_balancer.endpoints[-1].priority == 0
        assert "priority 0" in added

    @pytest.mark.asyncio
    async def test_the_new_endpoint_gets_its_own_label(self, agent):
        await commands.cmd_host(agent, "add http://gpu-3:8000/v1")
        await commands.cmd_host(agent, "add http://gpu-4:8000/v1")
        names = [ep.name for ep in agent.load_balancer.endpoints]
        assert len(set(names)) == len(names)

    @pytest.mark.asyncio
    async def test_a_duplicate_is_refused(self, agent):
        out = await commands.cmd_host(agent, "add http://localhost:8000/v1")
        assert "already in the rotation" in out
        assert len(agent.load_balancer.endpoints) == 2

    @pytest.mark.asyncio
    async def test_a_bare_hostname_is_rejected(self, agent):
        out = await commands.cmd_host(agent, "add gpu-3:8000")
        assert "http://" in out
        assert len(agent.load_balancer.endpoints) == 2

    @pytest.mark.asyncio
    async def test_add_with_no_url_says_what_it_needs(self, agent):
        out = await commands.cmd_host(agent, "add")
        assert "needs a URL" in out

    @pytest.mark.asyncio
    async def test_a_sticky_pin_is_dropped_so_the_new_host_is_reachable(self, agent):
        """Sticky holds an index into the endpoint list. Kept across an
        insert it would point at a different server; kept at all, the session
        would never touch the endpoint just added."""
        agent.load_balancer._sticky_map["sess-1"] = 1
        await commands.cmd_host(agent, "add http://gpu-3:8000/v1")
        assert agent.load_balancer._sticky_map == {}

    @pytest.mark.asyncio
    async def test_sticky_re_picking_is_explained(self, agent):
        out = await commands.cmd_host(agent, "add http://gpu-3:8000/v1")
        assert "re-picks" in out

    @pytest.mark.asyncio
    async def test_the_table_is_shown_after_the_change(self, agent):
        out = await commands.cmd_host(agent, "add http://gpu-3:8000/v1")
        assert "PRIO" in out and "http://gpu-3:8000/v1" in out

    @pytest.mark.asyncio
    async def test_it_says_the_config_is_untouched(self, agent):
        assert "onit setup" in await commands.cmd_host(
            agent, "add http://gpu-3:8000/v1")


class TestHostRemove:
    @pytest.mark.asyncio
    async def test_rm_by_row_number(self, agent):
        await commands.cmd_host(agent, "rm 2")
        assert agent.load_balancer.hosts == ["http://localhost:8000/v1"]

    @pytest.mark.asyncio
    async def test_rm_by_url(self, agent):
        await commands.cmd_host(agent, "rm http://localhost:8001/v1")
        assert agent.load_balancer.hosts == ["http://localhost:8000/v1"]

    @pytest.mark.asyncio
    async def test_rm_by_name(self, agent):
        await commands.cmd_host(agent, "rm vllm-b")
        assert agent.load_balancer.hosts == ["http://localhost:8000/v1"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("verb", ["rm", "remove", "delete", "del"])
    async def test_the_spellings_all_work(self, agent, verb):
        await commands.cmd_host(agent, f"{verb} 2")
        assert len(agent.load_balancer.endpoints) == 1

    @pytest.mark.asyncio
    async def test_the_last_endpoint_cannot_be_dropped(self, agent):
        """There would be nothing left to send the next task to."""
        await commands.cmd_host(agent, "rm 2")
        out = await commands.cmd_host(agent, "rm 1")
        assert "only endpoint" in out
        assert len(agent.load_balancer.endpoints) == 1

    @pytest.mark.asyncio
    async def test_an_unknown_target_is_reported(self, agent):
        out = await commands.cmd_host(agent, "rm 9")
        assert "No endpoint matches" in out
        assert len(agent.load_balancer.endpoints) == 2

    @pytest.mark.asyncio
    async def test_rm_with_no_target_says_what_it_needs(self, agent):
        out = await commands.cmd_host(agent, "rm")
        assert "row number or URL" in out

    @pytest.mark.asyncio
    async def test_dropping_the_preferred_host_moves_the_mirror(self, agent):
        """serving.host stands for the preferred endpoint; left naming a
        server no longer in the rotation it misreports the whole session."""
        await commands.cmd_host(agent, "rm 1")
        assert agent.model_serving["host"] == "http://localhost:8001/v1"

    @pytest.mark.asyncio
    async def test_a_stale_model_mirror_is_cleared(self, agent):
        await commands.cmd_host(agent, "rm 1")
        assert "model" not in agent.model_serving


# ── ollama endpoints ────────────────────────────────────────────────────────

class TestOllamaEndpoints:
    """An Ollama host added to a rotation is held back by the implicit
    fallback rule. The routing is right; saying nothing about it was not."""

    @pytest.fixture
    def mixed(self, agent):
        agent.load_balancer = LoadBalancer(
            [ServerEndpoint(host="http://gpu-1:8000/v1", name="vllm-a")],
            "round_robin")
        return agent

    @pytest.mark.asyncio
    async def test_an_added_ollama_host_is_marked_in_the_table(self, mixed):
        out = await commands.cmd_host(mixed, "add https://ollama.com")
        row = [ln for ln in out.splitlines()
               if "ollama.com" in ln and ln.strip()[0].isdigit()][0]
        assert "fallback only" in row

    @pytest.mark.asyncio
    async def test_the_message_says_it_will_not_serve_yet(self, mixed):
        out = await commands.cmd_host(mixed, "add https://ollama.com")
        assert "will not serve traffic yet" in out
        assert "--share" in out

    @pytest.mark.asyncio
    async def test_share_puts_it_in_rotation(self, mixed):
        await commands.cmd_host(mixed, "add https://ollama.com --share")
        lb = mixed.load_balancer
        picks = {lb.acquire().host for _ in range(20)}
        assert picks == {"http://gpu-1:8000/v1", "https://ollama.com"}

    @pytest.mark.asyncio
    async def test_without_share_it_serves_nothing(self, mixed):
        await commands.cmd_host(mixed, "add https://ollama.com")
        lb = mixed.load_balancer
        picks = {lb.acquire().host for _ in range(20)}
        assert picks == {"http://gpu-1:8000/v1"}

    @pytest.mark.asyncio
    async def test_share_is_not_mistaken_for_a_model_name(self, mixed):
        await commands.cmd_host(mixed, "add https://ollama.com --share")
        assert mixed.load_balancer.endpoints[-1].model is None

    @pytest.mark.asyncio
    async def test_a_model_still_works_beside_the_flag(self, mixed):
        await commands.cmd_host(mixed, "add https://ollama.com glm-5.1:cloud --share")
        assert mixed.load_balancer.endpoints[-1].model == "glm-5.1:cloud"

    @pytest.mark.asyncio
    async def test_share_works_on_an_already_added_host(self, mixed):
        """The held-back message tells the user to type this exact line;
        refusing it would send them back to remove-and-re-add."""
        await commands.cmd_host(mixed, "add https://ollama.com")
        out = await commands.cmd_host(mixed, "add https://ollama.com --share")
        assert "equal share" in out
        picks = {mixed.load_balancer.acquire().host for _ in range(20)}
        assert "https://ollama.com" in picks

    @pytest.mark.asyncio
    async def test_re_adding_without_share_is_still_refused(self, mixed):
        await commands.cmd_host(mixed, "add https://ollama.com")
        out = await commands.cmd_host(mixed, "add https://ollama.com")
        assert "already in the rotation" in out
        assert len(mixed.load_balancer.endpoints) == 2

    @pytest.mark.asyncio
    async def test_share_on_a_host_already_sharing_changes_nothing(self, mixed):
        await commands.cmd_host(mixed, "add http://gpu-2:8000/v1")
        out = await commands.cmd_host(mixed, "add http://gpu-2:8000/v1 --share")
        assert "nothing to change" in out
        assert mixed.load_balancer.ollama_fallback_only is True

    @pytest.mark.asyncio
    async def test_share_says_it_changed_a_session_wide_rule(self, mixed):
        """It is not a property of the endpoint added — every Ollama host in
        the list is affected, and the message must not imply otherwise."""
        out = await commands.cmd_host(mixed, "add https://ollama.com --share")
        assert "whole session" in out

    @pytest.mark.asyncio
    async def test_an_ollama_only_rotation_is_not_held_back(self, agent):
        """With nothing else to prefer, the rule must not strand the session
        with no endpoint at all."""
        agent.load_balancer = LoadBalancer(
            [ServerEndpoint(host="https://ollama.com", name="cloud")],
            "round_robin")
        out = await commands.cmd_host(agent, "")
        assert "fallback only" not in out

    def test_a_cooling_down_peer_releases_the_standby(self, mixed):
        """The label tracks the live rule: once the vLLM host fails, the
        Ollama one is what serves."""
        import time
        lb = LoadBalancer([ServerEndpoint(host="http://gpu-1:8000/v1"),
                           ServerEndpoint(host="https://ollama.com")],
                          "round_robin")
        ollama = lb.endpoints[1]
        assert lb.is_fallback_only(ollama) is True
        lb.endpoints[0].failed_at = time.monotonic()
        assert lb.is_fallback_only(ollama) is False


# ── \key ────────────────────────────────────────────────────────────────────

class TestKey:
    @pytest.fixture(autouse=True)
    def _fake_keystore(self, monkeypatch):
        from src import setup as setup_mod
        self.store = {}
        monkeypatch.setattr(setup_mod, "get_secret", lambda k: self.store.get(k))
        monkeypatch.setattr(setup_mod, "store_secret",
                            lambda k, v: self.store.__setitem__(k, v))
        monkeypatch.setattr(setup_mod, "delete_secret",
                            lambda k: self.store.pop(k, None))
        for var in ("OPENROUTER_API_KEY", "OLLAMA_API_KEY", "VLLM_API_KEY"):
            monkeypatch.delenv(var, raising=False)

    @staticmethod
    def _typed(agent, value):
        agent.chat_ui.read_secret = lambda prompt="": value

    @pytest.mark.asyncio
    async def test_the_key_is_read_separately_not_from_the_line(self, agent):
        """Typed as an argument it would be drawn on screen, kept in the
        input history, and left in the chat panel."""
        self._typed(agent, "sk-typed-1234")
        out = await commands.cmd_key(agent, "")
        from src import setup as setup_mod
        assert (setup_mod.get_endpoint_key("http://localhost:8000/v1")
                == "sk-typed-1234")
        assert "sk-typed-1234" not in out

    @pytest.mark.asyncio
    async def test_only_the_last_four_are_echoed_back(self, agent):
        self._typed(agent, "sk-typed-1234")
        assert "••••1234" in await commands.cmd_key(agent, "")

    @pytest.mark.asyncio
    async def test_a_key_on_the_command_line_is_refused(self, agent):
        self._typed(agent, "unused")
        out = await commands.cmd_key(agent, "1 sk-pasted-secret")
        assert "asked for separately" in out
        assert self.store == {}

    @pytest.mark.asyncio
    async def test_it_targets_the_session_endpoint_by_default(self, agent):
        agent.load_balancer._sticky_map["sess-1"] = 1
        self._typed(agent, "sk-b")
        await commands.cmd_key(agent, "")
        from src import setup as setup_mod
        assert setup_mod.get_endpoint_key("http://localhost:8001/v1") == "sk-b"

    @pytest.mark.asyncio
    async def test_an_endpoint_can_be_named(self, agent):
        self._typed(agent, "sk-b")
        await commands.cmd_key(agent, "2")
        from src import setup as setup_mod
        assert setup_mod.get_endpoint_key("http://localhost:8001/v1") == "sk-b"

    @pytest.mark.asyncio
    async def test_an_unknown_endpoint_is_reported(self, agent):
        self._typed(agent, "sk-b")
        out = await commands.cmd_key(agent, "9")
        assert "No endpoint matches" in out
        assert self.store == {}

    @pytest.mark.asyncio
    async def test_an_empty_entry_cancels(self, agent):
        self._typed(agent, "")
        out = await commands.cmd_key(agent, "")
        assert "No key entered" in out
        assert self.store == {}

    @pytest.mark.asyncio
    async def test_rm_forgets_the_stored_key(self, agent):
        from src import setup as setup_mod
        setup_mod.store_endpoint_key("http://localhost:8000/v1", "sk-a")
        out = await commands.cmd_key(agent, "rm 1")
        assert setup_mod.get_endpoint_key("http://localhost:8000/v1") is None
        assert "falls back to" in out

    @pytest.mark.asyncio
    async def test_rm_on_an_endpoint_without_one_says_so(self, agent):
        out = await commands.cmd_key(agent, "rm 1")
        assert "no stored key" in out

    @pytest.mark.asyncio
    async def test_a_config_key_shadowing_the_new_one_is_flagged(self, agent):
        """Otherwise the key is stored, confirmed, and silently never used."""
        agent.load_balancer.endpoints[0].host_key = "sk-in-yaml"
        self._typed(agent, "sk-typed-1234")
        out = await commands.cmd_key(agent, "1")
        assert "read first" in out

    @pytest.mark.asyncio
    async def test_it_says_the_key_persists(self, agent):
        """Every other \\host change dies with the session; this one does not,
        and the asymmetry is worth stating."""
        self._typed(agent, "sk-typed-1234")
        assert "persist" in await commands.cmd_key(agent, "")

    @pytest.mark.asyncio
    async def test_an_interface_that_cannot_mask_says_so(self, agent):
        agent.chat_ui = SimpleNamespace(model_name="")
        out = await commands.cmd_key(agent, "")
        assert "onit setup" in out
        assert self.store == {}


# ── \setup ──────────────────────────────────────────────────────────────────

class TestSetup:
    def test_lists_endpoints_and_session_paths(self, agent, monkeypatch):
        monkeypatch.setattr("src.setup.get_secret", lambda key: None)
        monkeypatch.setattr("src.ui.commands._config_notes", lambda: [])
        out = commands.render_setup(agent)
        assert "http://localhost:8000/v1" in out
        assert "sess-1" in out
        assert "/tmp/data" in out

    def test_marks_the_endpoint_this_session_uses(self, agent, monkeypatch):
        monkeypatch.setattr("src.setup.get_secret", lambda key: None)
        monkeypatch.setattr("src.ui.commands._config_notes", lambda: [])
        agent.load_balancer._sticky_map["sess-1"] = 1
        rows = [ln for ln in commands.render_setup(agent).splitlines()
                if "*" in ln and "8001" in ln]
        assert rows, "the sticky endpoint should carry the marker"

    def test_a_key_is_never_printed_in_full(self, agent, monkeypatch):
        monkeypatch.setattr("src.setup.get_secret",
                            lambda key: "sk-secret-value-1234")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setattr("src.ui.commands._config_notes", lambda: [])
        out = commands.render_setup(agent)
        assert "sk-secret-value-1234" not in out
        assert "••••1234" in out

    def test_an_unauthenticated_endpoint_says_none(self, agent, monkeypatch):
        monkeypatch.setattr("src.setup.get_secret", lambda key: None)
        for var in ("OPENROUTER_API_KEY", "OLLAMA_API_KEY", "VLLM_API_KEY",
                    "ONIT_HOST2_KEY"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr("src.ui.commands._config_notes", lambda: [])
        out = commands.render_setup(agent)
        row = [ln for ln in out.splitlines() if "vllm-a" in ln][0]
        assert "none" in row
        # Nothing is falling back, so the legacy section is not worth a heading.
        assert "Legacy fallback keys" not in out

    def test_a_fallback_key_is_named_in_the_column(self, agent, monkeypatch):
        """'set' and 'none' are both wrong for an endpoint running on a
        provider-named secret — the user needs to know which one."""
        monkeypatch.setattr(
            "src.setup.get_secret",
            lambda key: "sk-old" if key == "vllm_api_key" else None)
        monkeypatch.delenv("VLLM_API_KEY", raising=False)
        monkeypatch.setattr("src.ui.commands._config_notes", lambda: [])
        out = commands.render_setup(agent)
        row = [ln for ln in out.splitlines() if "vllm-a" in ln][0]
        assert "vllm_api_key" in row
        assert "Legacy fallback keys" in out

    def test_a_key_written_into_the_config_is_labelled_as_such(self, agent,
                                                               monkeypatch):
        monkeypatch.setattr("src.setup.get_secret", lambda key: None)
        monkeypatch.setattr("src.ui.commands._config_notes", lambda: [])
        agent.load_balancer.endpoints[0].host_key = "sk-in-yaml"
        out = commands.render_setup(agent)
        row = [ln for ln in out.splitlines() if "vllm-a" in ln][0]
        assert "in config" in row
        assert "sk-in-yaml" not in out

    def test_config_notes_are_included(self, agent, monkeypatch):
        monkeypatch.setattr("src.setup.get_secret", lambda key: None)
        monkeypatch.setattr("src.ui.commands._config_notes",
                            lambda: ["Note: no key for OpenRouter."])
        assert "no key for OpenRouter" in commands.render_setup(agent)

    def test_a_broken_config_file_does_not_break_the_listing(self, monkeypatch):
        monkeypatch.setattr("src.setup._load_config",
                            lambda: (_ for _ in ()).throw(OSError("nope")))
        assert commands._config_notes() == []


# ── dispatch ────────────────────────────────────────────────────────────────

class TestDispatch:
    @pytest.mark.asyncio
    async def test_a_message_is_not_handled(self, agent):
        assert await commands.dispatch(agent, "hello there") is None

    @pytest.mark.asyncio
    async def test_stop_aliases_are_left_to_the_caller(self, agent):
        """OnIt.stop_commands owns them; claiming them here would swallow
        the one command that has to end the loop."""
        for word in ("\\bye", "\\exit", "\\quit", "\\goodbye"):
            assert await commands.dispatch(agent, word) is None

    @pytest.mark.asyncio
    async def test_help_is_handled(self, agent):
        assert "\\setup" in await commands.dispatch(agent, "\\help")

    @pytest.mark.asyncio
    async def test_typo_is_answered_rather_than_sent(self, agent):
        out = await commands.dispatch(agent, "\\hlep")
        assert out is not None and "\\help" in out

    @pytest.mark.asyncio
    async def test_a_failing_handler_costs_a_message_not_the_session(self, agent):
        with patch.object(commands, "cmd_host",
                          side_effect=RuntimeError("boom")):
            out = await commands.dispatch(agent, "\\host http://x/v1")
        assert "\\host failed" in out and "boom" in out


# ── the chat loop ───────────────────────────────────────────────────────────

class _Forwarded(Exception):
    """Raised from the first step of the forward-to-the-model path, so a test
    can tell "sent on" apart from "answered here" without standing up a model,
    queues and an agent session."""


class TestLoopInterception:
    """A command must be answered by the session, never forwarded — a \\setup
    that reached the model would cost a full turn and get a made-up answer."""

    @staticmethod
    def _fake_session(inputs, web=False):
        """The subset of OnIt that client_to_agent touches before a command
        is either answered or passed on to the model."""
        from src.onit import OnIt

        posted = []
        queue = list(inputs)

        async def get_task(loop):
            return queue.pop(0)

        fake = SimpleNamespace(
            web=web,
            stop_commands=["\\bye", "\\exit"],
            load_balancer=LoadBalancer(
                [ServerEndpoint(host="http://localhost:8000/v1",
                                model="Qwen3-30B", name="vllm-a")], "sticky"),
            model_serving={"host": "http://localhost:8000/v1"},
            session_id="sess-1", session_path="/tmp/s.jsonl", data_path="",
            chat_ui=SimpleNamespace(
                console=SimpleNamespace(print=lambda *a, **k: None),
                model_name="Qwen3-30B",
                add_message=lambda role, response, **kw: posted.append(
                    (role, response)),
            ),
            _get_user_task=get_task,
            cancel_background_checks=lambda: None,
            # First call on the path that forwards a message to the model.
            _cancel_background_check=lambda _sid: (_ for _ in ()).throw(
                _Forwarded()),
        )
        fake.run = OnIt.client_to_agent.__get__(fake, type(fake))
        return fake, posted

    @pytest.mark.asyncio
    async def test_a_command_is_answered_without_a_model_turn(self):
        fake, posted = self._fake_session(["\\help", "\\bye"])
        await fake.run()
        assert [role for role, _ in posted] == ["system"]
        assert "\\setup" in posted[0][1]

    @pytest.mark.asyncio
    async def test_a_plain_message_still_reaches_the_model(self):
        fake, _ = self._fake_session(["what is 2+2", "\\bye"])
        with pytest.raises(_Forwarded):
            await fake.run()

    @pytest.mark.asyncio
    async def test_stop_commands_still_end_the_session(self):
        fake, posted = self._fake_session(["\\bye"])
        await fake.run()
        assert posted == []

    @pytest.mark.asyncio
    async def test_the_web_ui_does_not_intercept(self):
        """WebApiUI.add_message is a no-op, so an answer posted there would
        vanish — better the model replies than nothing does."""
        fake, _ = self._fake_session(["\\help", "\\bye"], web=True)
        with pytest.raises(_Forwarded):
            await fake.run()
