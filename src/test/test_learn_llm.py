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

import os

import pytest

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
