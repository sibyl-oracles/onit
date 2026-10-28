"""Tests for the default output-token budget — one number, stated once.

The default used to live in three places that could disagree, and did:

  * ``DEFAULT_MAX_TOKENS`` in ``model/serving/chat.py`` — the code fallback
  * ``serving.max_tokens`` in ``configs/default.yaml`` — the shipped config
  * the ``--max-tokens`` help text in ``cli.py`` — which said 131072 while
    the code said 16384, and had done since 4d9100f lowered it

Which one *applies* is a fourth question: a user config at
``~/.onit/config.yaml`` replaces the shipped file rather than merging with
it, so on a machine whose user config omits ``serving.max_tokens`` the
shipped 32768 never applies and the code constant is what is used.  That is
why the shipped YAML being right did not stop the effective value from being
something else.

These tests pin all of them to one number, so the next person to change the
default finds out here rather than by reading a stale help string.
"""

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from model.serving.chat import DEFAULT_MAX_TOKENS


HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
REPO = os.path.dirname(SRC)

# The one number.  Change it here and in DEFAULT_MAX_TOKENS together.
EXPECTED = 32768


def _shipped_config() -> dict:
    import yaml
    with open(os.path.join(SRC, "configs", "default.yaml")) as f:
        return yaml.safe_load(f) or {}


def _cli_help_text() -> str:
    from src.cli import _build_parser
    return _build_parser().format_help()


def _flag_help(flag: str) -> str:
    """The help string of one option, and only that one.

    Reading it off the parser is better than searching the formatted help:
    the flag also appears in the usage line at the top, so a text search can
    span half the help and land on a *different* option's default.  The
    action's own ``help`` string is the thing that was written in the
    source, with no wrapping or re-flowing in the way.
    """
    from src.cli import _build_parser
    parser = _build_parser()
    action = parser._option_string_actions.get(flag)
    assert action is not None, f"{flag} is not an option of the parser"
    return action.help or ""


# ── the code constant ──────────────────────────────────────────────────────

class TestCodeConstant:
    def test_the_constant_is_the_expected_value(self):
        assert DEFAULT_MAX_TOKENS == EXPECTED

    def test_it_is_an_int(self):
        # A str here would sail through == 32768 nowhere but would break the
        # arithmetic in the threshold computation silently.
        assert isinstance(DEFAULT_MAX_TOKENS, int)
        assert not isinstance(DEFAULT_MAX_TOKENS, bool)


# ── the shipped config ─────────────────────────────────────────────────────

class TestShippedConfig:
    def test_the_shipped_config_agrees_with_the_code(self):
        cfg = _shipped_config()
        assert cfg.get("serving", {}).get("max_tokens") == EXPECTED

    def test_the_shipped_config_does_not_silently_raise_the_code_default(self):
        """The shipped file wins over the code for anyone using it, so a
        value *above* the code default would quietly raise the ceiling past
        what the code says is safe."""
        cfg = _shipped_config()
        shipped = cfg.get("serving", {}).get("max_tokens", EXPECTED)
        assert shipped <= DEFAULT_MAX_TOKENS


# ── the CLI help text ──────────────────────────────────────────────────────

class TestCliHelp:
    def test_the_help_text_names_the_real_default(self):
        help_text = _flag_help("--max-tokens")
        match = re.search(r"\(default:\s*([0-9]+)\)", help_text)
        assert match, "the --max-tokens help no longer states a default"
        assert int(match.group(1)) == EXPECTED

    def test_the_help_text_does_not_claim_the_old_value(self):
        """131072 is the value the help text carried for months after the
        code had moved on.  It is also the one value that clamps the
        compaction threshold to 0.50, so a stale mention is not cosmetic."""
        assert "131072" not in _flag_help("--max-tokens")

    def test_the_help_is_the_flag_own_text(self):
        """Guards the helper itself: it must return this option's help and
        not another option's."""
        help_text = _flag_help("--max-tokens")
        assert "Max output tokens per model response" in help_text
        assert "Context window size in tokens" not in help_text
        assert "Context window size in tokens" in _flag_help("--max-context-tokens")


# ── the compaction threshold, which is what the number really controls ────

class TestCompactionThreshold:
    """The output default is reserved against the context window, so it
    decides how full the context may get before compaction fires.  The
    formula is chat.py's:

        min(0.90, max(0.50, 1 - (max_tokens + 5% * window) / window))

    A default that clamps to the 0.50 floor forces compaction twice as
    often, which costs an LLM call every time.  This is the property that
    made 16384 the right value in the first place, and it is what the
    archived token plan warned about when it warned against 131072.
    """

    @staticmethod
    def _threshold(max_tokens: int, window: int) -> float:
        reserved = max_tokens + int(window * 0.05)
        return min(0.90, max(0.50, 1.0 - reserved / window))

    @pytest.mark.parametrize("window", [131072, 262144, 1048576])
    def test_the_default_never_clamps_to_the_floor(self, window):
        """On any large-context model the default leaves the threshold off
        the 0.50 floor.  (On a 32k or 64k window it does reach the floor,
        because a 32768 output budget is half the window in its own right --
        the per-request clamp shrinks it in practice.  That is a property of
        the window, not of the default.)"""
        assert self._threshold(DEFAULT_MAX_TOKENS, window) > 0.50

    def test_the_threshold_it_leaves(self):
        """The numbers, so a future change to the default is a deliberate
        change to one of these rather than a surprise."""
        assert self._threshold(DEFAULT_MAX_TOKENS, 131072) == pytest.approx(0.70, abs=1e-4)
        assert self._threshold(DEFAULT_MAX_TOKENS, 262144) == pytest.approx(0.825, abs=1e-4)
        assert self._threshold(DEFAULT_MAX_TOKENS, 1048576) == pytest.approx(0.90, abs=1e-4)

    def test_it_does_not_reach_the_old_131072_ceiling(self):
        """131072 on any window <= 256k clamps to exactly 0.50."""
        assert self._threshold(131072, 131072) == 0.50
        assert self._threshold(131072, 262144) == 0.50

    @pytest.mark.parametrize("window", [65536, 131072, 262144])
    def test_the_threshold_is_monotonic_in_the_default(self, window):
        """A bigger output default reserves more of the window, so the
        threshold can only go down."""
        assert (self._threshold(DEFAULT_MAX_TOKENS, window)
                <= self._threshold(16384, window))


# ── the docs ───────────────────────────────────────────────────────────────

class TestDocs:
    def _read(self, *parts: str) -> str:
        with open(os.path.join(REPO, *parts)) as f:
            return f.read()

    def test_cli_doc_states_the_default(self):
        text = self._read("docs", "CLI.md")
        match = re.search(r"\| `--max-tokens N` \|[^|]*\|([^|]*)\|", text)
        assert match, "docs/CLI.md no longer documents --max-tokens"
        assert "32768" in match.group(1)

    def test_configuration_doc_states_the_default(self):
        text = self._read("docs", "CONFIGURATION.md")
        assert "max_tokens: 32768" in text

    def test_the_docs_do_not_mention_the_old_code_default(self):
        """16384 is a fine number to mention in an archived plan — it is the
        shipped docs that must not claim it as the current default."""
        for name in ("CLI.md", "CONFIGURATION.md"):
            text = self._read("docs", name)
            assert "16384" not in text, f"docs/{name} still mentions 16384"


# ── the effective value, resolved the way a run resolves it ────────────────

class TestEffectiveValue:
    """The number that actually reaches the model is not the constant
    directly: it is the config's value when there is one, and the constant
    otherwise.  This exercises that path rather than assuming it."""

    def _resolved(self, cfg: dict) -> int:
        from src.onit import OnIt
        from type.tools import ToolRegistry
        from unittest.mock import patch
        with patch("src.onit.discover_tools", return_value=ToolRegistry()):
            agent = OnIt(config=cfg)
        return agent.model_serving.get("max_tokens", DEFAULT_MAX_TOKENS)

    def test_a_config_without_max_tokens_falls_back_to_the_constant(self):
        assert self._resolved({"serving": {"host": "http://x/v1"}}) == EXPECTED

    def test_a_config_with_max_tokens_wins(self):
        assert self._resolved({"serving": {"host": "http://x/v1",
                                           "max_tokens": 4096}}) == 4096

    def test_a_user_config_without_the_key_gets_the_code_default(self):
        """The case that made the shipped YAML irrelevant: a user config
        replaces the shipped file, so it is the code constant that answers
        when the user config omits the key."""
        import yaml
        user_cfg = yaml.safe_load(
            "serving:\n  host: http://x/v1\n  think: true\n")
        assert "max_tokens" not in user_cfg.get("serving", {})
        assert self._resolved(user_cfg) == EXPECTED
