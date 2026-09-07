# Regression test: Ollama chat() must not receive OpenAI-style part-list
# content. Reproduces the ValidationError from the 2026-09-07 crash log
# (tool message with [{'type': 'text', ...}, {'type': 'image_url',
# ... data:...;base64,...'}]).
import base64
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.serving.chat import _adapt_messages_for_ollama

# 1x1 red PNG, base64-encoded (same tail as the crashing payload).
_PNG_B64 = base64.b64encode(bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108020000009077"
    "53de0000000c4944415408d763f8cfc000000003000001952a8a4400000000"
    "0049454e44ae426082")).decode()


def _crashing_tool_message():
    """The exact shape from the traceback: text part + data-URL image part."""
    return {
        "role": "tool",
        "content": [
            {"type": "text", "text": "Chart rendered from data."},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{_PNG_B64}"}},
        ],
        "name": "plot",
        "tool_call_id": "call_123",
    }


def test_part_list_content_is_converted_to_string():
    out = _adapt_messages_for_ollama([_crashing_tool_message()])
    (msg,) = out
    assert isinstance(msg["content"], str)
    assert msg["content"] == "Chart rendered from data."


def _image_values(msg):
    """Serialized base64 of each image, whatever shape the client wants."""
    return [i if isinstance(i, str) else i.value for i in msg["images"]]


def test_image_moves_to_images_field_as_bare_base64():
    out = _adapt_messages_for_ollama([_crashing_tool_message()])
    (msg,) = out
    assert _image_values(msg) == [_PNG_B64]  # data: prefix stripped
    assert "base64" not in _image_values(msg)[0]


def test_string_content_passes_through_untouched():
    msgs = [{"role": "user", "content": "hello"},
            {"role": "tool", "content": "plain result", "name": "bash"}]
    out = _adapt_messages_for_ollama(msgs)
    assert out == msgs  # same dicts, no copies, no extra keys


def test_original_history_is_not_mutated():
    msgs = [_crashing_tool_message()]
    _adapt_messages_for_ollama(msgs)
    assert isinstance(msgs[0]["content"], list)  # still the OpenAI shape
    assert "images" not in msgs[0]


def test_subagent_image_instruction_shape():
    """Second producer site: user message with inline image (line ~1662)."""
    msg = {"role": "user", "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{_PNG_B64}"}},
    ]}
    out = _adapt_messages_for_ollama([msg])
    (m,) = out
    assert m["content"] == "What is in this image?"
    assert _image_values(m) == [_PNG_B64]


def test_mixed_and_empty_parts():
    msg = {"role": "tool", "content": [
        {"type": "image_url", "image_url": {"url": ""}},   # empty url dropped
        {"type": "text", "text": "first"},
        {"type": "text", "text": "second"},
        "bare string part",
    ]}
    (m,) = _adapt_messages_for_ollama([msg])
    assert m["content"] == "first\nsecond\nbare string part"
    assert "images" not in m


def test_non_dict_messages_pass_through():
    msgs = [{"role": "user", "content": "x"}, "weird-but-keep"]
    assert _adapt_messages_for_ollama(msgs) == msgs


def test_adapted_messages_validate_against_ollama_message_model():
    """The real acceptance criterion: the installed ollama client's pydantic
    Message model must accept every adapted message."""
    ollama = pytest.importorskip("ollama")
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "draw a chart"},
        _crashing_tool_message(),
        {"role": "user", "content": [
            {"type": "text", "text": "now explain it"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{_PNG_B64}"}},
        ]},
    ]
    for m in _adapt_messages_for_ollama(msgs):
        ollama.Message.model_validate(m)