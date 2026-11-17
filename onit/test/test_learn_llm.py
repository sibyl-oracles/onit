"""
The background loops' model client, tested without a model server.

``llm.py`` is the only place the offline loops talk to a model, and
two of its properties matter enough to be pinned:

1. **It carries no tools.**  The Reflector judges the agent and the
   runner grades the agent.  A judge that could run the agent's tools
   would not be a judge -- it would be a second agent whose output is
   then called a verdict.  The client is built with no registry and no
   harness for exactly this reason.

2. **It talks to the same model the agent does.**  A reflection about
   a model the agent is not running is a reflection about nothing.
   So the endpoint is resolved from the same places the agent resolves
   it from, in the same order.

The network calls themselves are not tested here -- that needs a
server.  What is tested is the resolution, which is where the two
properties above are decided.
"""

from onit.learn import llm


class TestResolveEndpoint:
    """Where the background loops talk to."""

    def test_the_config_serving_block_wins(self):
        config = {"serving": {"host": "http://config/v1", "model": "config-model"}}
        endpoint = llm.resolve_endpoint(config)
        assert endpoint["host"] == "http://config/v1"
        assert endpoint["model"] == "config-model"

    def test_the_environment_is_used_when_the_config_is_silent(self, monkeypatch):
        monkeypatch.setenv("ONIT_HOST", "http://env/v1")
        endpoint = llm.resolve_endpoint({})
        assert endpoint["host"] == "http://env/v1"

    def test_the_config_beats_the_environment(self, monkeypatch):
        monkeypatch.setenv("ONIT_HOST", "http://env/v1")
        endpoint = llm.resolve_endpoint({"serving": {"host": "http://config/v1"}})
        assert endpoint["host"] == "http://config/v1"

    def test_there_is_a_default_host(self):
        assert llm.resolve_endpoint({})["host"]

    def test_no_model_means_auto_detect(self):
        """An unnamed model is detected from the endpoint.

        The alternative -- a hard-coded model name -- would send the
        reflection to a model the agent is not using, and the bullet
        would be about a different agent.
        """
        endpoint = llm.resolve_endpoint({})
        assert endpoint["model"] is None

    def test_the_host_key_is_carried(self):
        endpoint = llm.resolve_endpoint({"serving": {"host": "http://x/v1", "host_key": "secret"}})
        assert endpoint["host_key"] == "secret"

    def test_the_default_host_key_is_empty(self):
        assert llm.resolve_endpoint({})["host_key"] == "EMPTY"


class TestTheEndpointsListShape:
    """The config's serving block can be an ``endpoints`` list.

    The resolver used to read only ``serving.host``, a key the list
    shape never fills, so every background loop on such a config
    resolved to the localhost default -- a host nothing listens on --
    and every grading call died silently.  The list's preferred entry
    (lowest priority number) must resolve instead.
    """

    def test_the_preferred_listed_endpoint_wins(self):
        config = {
            "serving": {
                "endpoints": [
                    {"host": "http://b/v1", "model": "b-model", "priority": 1},
                    {"host": "http://a/v1", "model": "a-model", "priority": 0},
                ]
            }
        }
        endpoint = llm.resolve_endpoint(config)
        assert endpoint["host"] == "http://a/v1"
        assert endpoint["model"] == "a-model"

    def test_list_ties_break_by_order(self):
        config = {
            "serving": {
                "endpoints": [
                    {"host": "http://first/v1", "model": "first", "priority": 0},
                    {"host": "http://second/v1", "model": "second", "priority": 0},
                ]
            }
        }
        endpoint = llm.resolve_endpoint(config)
        assert endpoint["host"] == "http://first/v1"

    def test_a_flat_host_still_beats_the_list(self):
        """An operator pinning one endpoint keeps the pin."""
        config = {
            "serving": {
                "host": "http://pinned/v1",
                "endpoints": [{"host": "http://listed/v1", "priority": 0}],
            }
        }
        assert llm.resolve_endpoint(config)["host"] == "http://pinned/v1"

    def test_the_listed_key_is_carried(self):
        config = {
            "serving": {"endpoints": [{"host": "http://a/v1", "host_key": "secret", "priority": 0}]}
        }
        assert llm.resolve_endpoint(config)["host_key"] == "secret"

    def test_an_empty_or_malformed_list_falls_through(self):
        config = {"serving": {"endpoints": []}}
        assert llm.resolve_endpoint(config)["host"]
        config = {"serving": {"endpoints": "not-a-list"}}
        assert llm.resolve_endpoint(config)["host"]
        config = {"serving": {"endpoints": [None, "string"]}}
        assert llm.resolve_endpoint(config)["host"]


class TestTheClientCarriesNoTools:
    """The judge has no hands."""

    def test_the_builder_takes_no_registry(self):
        """The signature is the contract.

        ``_build_client`` takes a host, a key and a timeout.  There is
        nowhere to pass a tool registry, so there is nowhere for one to
        get in.
        """
        import inspect

        params = list(inspect.signature(llm._build_client).parameters)
        assert params == ["host", "host_key", "timeout"]

    def test_the_ask_callable_sends_no_tools(self):
        """The request carries messages and nothing else.

        This is the check that would fail if someone added tools to the
        judge: the request would grow a "tools" key.
        """
        import inspect

        source = inspect.getsource(llm.make_ask)
        assert '"tools"' not in source
        assert "'tools'" not in source
        assert "tool_registry" not in source
        assert "tool_registry" not in inspect.getsource(llm._build_client)


class TestTheClientTupleOrder:
    """The three values ``_build_client`` returns land in the right slots.

    ``make_ask`` unpacks the tuple as ``(client, ollama_client, is_ollama)``
    and then calls ``ollama_client.chat(...)`` on an Ollama host.  The
    Ollama client was returned in the *first* slot, so ``ollama_client``
    was ``None`` and every background call on an Ollama host raised
    ``'NoneType' object has no attribute 'chat'``.

    Nothing noticed, because both callers swallow the exception: the
    Reflector returned no deltas (indistinguishable from "the two runs had
    nothing to teach") and the grader returned an empty grade (counted as
    "ungraded", not as an error).  An A/B on an Ollama host therefore
    measured 0.000 in both arms and reported it as a clean null result.

    The other tests in this file inspect the signature and the source text,
    which is exactly why they all passed while the client was unusable.
    """

    def test_the_ollama_client_is_in_the_second_slot(self, monkeypatch):
        from onit.learn import llm as llm_module

        from onit.model.serving import chat as chat_module

        # An Ollama host will not build without a key, and asking for one is
        # not what this test is about.  The keychain is empty here (the
        # suite's ``_no_real_keychain``), so the key has to be named -- and
        # named on the seam ``_resolve_api_key`` reads, or the test passes on
        # a machine that has an ``ollama_api_key`` stored and fails on one
        # that does not.
        import onit.setup as setup_mod

        monkeypatch.setattr(
            setup_mod, "get_secret", lambda key: "sk-test" if key == "ollama_api_key" else None
        )
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)

        sentinel = object()
        real = chat_module._create_ollama_client
        # ``_build_client`` imports the name from chat at call time, so the
        # patch has to land there.
        chat_module._create_ollama_client = lambda *a, **k: sentinel
        try:
            client, ollama_client, is_ollama = llm_module._build_client(
                "https://ollama.com", "EMPTY", 30.0
            )
        finally:
            chat_module._create_ollama_client = real

        assert is_ollama is True
        assert ollama_client is sentinel, (
            "the Ollama client must be the second value; make_ask calls .chat() on that slot"
        )
        assert client is None

    def test_an_openai_host_puts_the_client_in_the_first_slot(self):
        from onit.learn import llm as llm_module

        client, ollama_client, is_ollama = llm_module._build_client(
            "http://127.0.0.1:9999/v1", "EMPTY", 5.0
        )

        assert is_ollama is False
        assert client is not None
        assert ollama_client is None


class TestTheOutputBudget:
    """The budget must cover a reasoning model's thinking, not just its answer.

    At 400 tokens a reasoning model spent the whole budget thinking and
    returned empty text: 12 of 40 items in the 2026-11-17 gate run came
    back ungraded with ``grading_errors`` reporting 0, and the seeding
    pass extracted nothing from a 20-item session.
    """

    def test_the_budget_is_not_a_verdict_sized_number(self):
        assert llm.JUDGE_MAX_TOKENS >= 4000

    def test_the_budget_is_the_same_everywhere(self):
        import inspect

        source = inspect.getsource(llm.make_ask)
        assert "JUDGE_MAX_TOKENS if len(messages)" not in source
        assert source.count("JUDGE_MAX_TOKENS") >= 2
        assert "400" not in source.replace("JUDGE_MAX_TOKENS", "")
