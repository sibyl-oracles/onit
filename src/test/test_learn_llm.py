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



from src.learn import llm


class TestResolveEndpoint:
    """Where the background loops talk to."""

    def test_the_config_serving_block_wins(self):
        config = {"serving": {"host": "http://config/v1",
                              "model": "config-model"}}
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
        endpoint = llm.resolve_endpoint(
            {"serving": {"host": "http://x/v1", "host_key": "secret"}})
        assert endpoint["host_key"] == "secret"

    def test_the_default_host_key_is_empty(self):
        assert llm.resolve_endpoint({})["host_key"] == "EMPTY"


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

    def test_the_ollama_client_is_in_the_second_slot(self):
        from src.learn import llm as llm_module

        from src.model.serving import chat as chat_module

        sentinel = object()
        real = chat_module._create_ollama_client
        # ``_build_client`` imports the name from chat at call time, so the
        # patch has to land there.
        chat_module._create_ollama_client = lambda *a, **k: sentinel
        try:
            client, ollama_client, is_ollama = llm_module._build_client(
                "https://ollama.com", "EMPTY", 30.0)
        finally:
            chat_module._create_ollama_client = real

        assert is_ollama is True
        assert ollama_client is sentinel, (
            "the Ollama client must be the second value; make_ask calls "
            ".chat() on that slot")
        assert client is None

    def test_an_openai_host_puts_the_client_in_the_first_slot(self):
        from src.learn import llm as llm_module

        client, ollama_client, is_ollama = llm_module._build_client(
            "http://127.0.0.1:9999/v1", "EMPTY", 5.0)

        assert is_ollama is False
        assert client is not None
        assert ollama_client is None
