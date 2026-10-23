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

A model client for the offline loops, and only for the offline loops.

The Reflector and the evaluation runner both need to *ask a model a
question* outside the agent's own conversation: the Reflector to compare
two runs, the runner to grade an answer.  Neither of them may use the
agent's tool registry, because both are judging the agent -- a judge
that can run the agent's tools is not a judge, it is a second agent
whose output is then called a verdict.

So this module builds a *plain* client: an OpenAI-compatible or Ollama
client with no tools, no harness, and no registry.  It reuses the
endpoint resolution the agent already uses -- the configured host, the
key from the keychain, the model name -- so that the model that
reflects and the model that answers are the same one, and a reflection
is about the model that will act on it.

The one thing that is deliberately *not* shared is the conversation:
these calls carry their own messages and nothing else.
"""

import logging
import os

logger = logging.getLogger(__name__)

# The output budget for a grading call.  A verdict is one letter.
JUDGE_MAX_TOKENS = 64


def resolve_endpoint(config_data: dict | None = None) -> dict:
    """The host, model and key a background loop should talk to.

    Read from the same places the agent reads them, in the same order:
    the ``serving:`` block, then the environment, then the agent's own
    config file.  Resolving it in one place means the loop and the agent
    cannot end up pointed at different models, which would make a
    reflection about a model the agent is not using.
    """
    serving = dict((config_data or {}).get("serving") or {})
    host = (serving.get("host")
            or os.environ.get("ONIT_HOST")
            or "http://127.0.0.1:8001/v1")
    model = serving.get("model") or os.environ.get("ONIT_MODEL") or None
    host_key = serving.get("host_key") or "EMPTY"
    return {"host": host, "model": model, "host_key": host_key}


def _build_client(host: str, host_key: str, timeout: float | None):
    """A plain client for ``host``, with the key the agent would use.

    Returns ``(client, ollama_client, is_ollama)``.  The client is
    built here rather than borrowed so that no tool registry, no
    harness and no conversation can leak into a background call.
    """
    from src.model.serving.chat import (_build_client_timeout,
                                        _create_ollama_client,
                                        _is_ollama_host,
                                        _normalize_openai_base_url,
                                        _resolve_api_key)

    api_key = _resolve_api_key(host, host_key)
    host = _normalize_openai_base_url(host)
    is_ollama = _is_ollama_host(host)
    if is_ollama:
        return _create_ollama_client(host, api_key, timeout, stream=False), \
            None, True
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=host, api_key=api_key,
                         timeout=_build_client_timeout(timeout, stream=False))
    return client, None, False


async def _resolve_model(client, ollama_client, is_ollama: bool,
                         host: str, model: str | None) -> str | None:
    """The model name, auto-detected when the caller did not name one."""
    if model:
        return model
    try:
        if is_ollama:
            from src.model.serving.chat import _ollama_resolve_model_id
            return await _ollama_resolve_model_id(ollama_client, host)
        from src.model.serving.chat import _resolve_model_id
        return await _resolve_model_id(client, host)
    except Exception as e:
        logger.debug("model auto-detect failed at %s: %s", host, e)
        return None


async def make_ask(config_data: dict | None = None, *, model: str | None = None,
                   timeout: float | None = 120.0):
    """Build the ``ask(messages)`` callable the loops take.

    Returns a coroutine function that takes a message list and returns
    the model's text.  It carries no tools, so a model asked to compare
    two runs cannot go and run anything while it thinks about them.
    """
    endpoint = resolve_endpoint(config_data)
    host = endpoint["host"]
    client, ollama_client, is_ollama = _build_client(host, endpoint["host_key"],
                                                     timeout)
    model_id = await _resolve_model(client, ollama_client, is_ollama, host,
                                    model or endpoint["model"])

    async def ask(messages: list) -> str:
        if is_ollama:
            from src.model.serving.chat import _adapt_messages_for_ollama
            resp = await ollama_client.chat(
                model=model_id,
                messages=_adapt_messages_for_ollama(messages),
                stream=False, think=False,
                options={"temperature": 0.0, "num_predict": JUDGE_MAX_TOKENS
                         if len(messages) < 3 else 400, "stop": []},
            )
            return getattr(resp.message, "content", "") or ""
        from src.model.serving.chat import _is_openai_host
        _kw = {"model": model_id, "messages": messages, "temperature": 0.0,
               "stream": False}
        if _is_openai_host(host):
            _kw["max_completion_tokens"] = 400
        else:
            _kw["max_tokens"] = 400
        resp = await client.chat.completions.create(**_kw)
        try:
            return resp.choices[0].message.content or ""
        except (AttributeError, IndexError):
            return ""

    return ask
