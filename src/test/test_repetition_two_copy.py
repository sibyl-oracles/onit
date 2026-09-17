"""Regression: the two-copy long-unit loop deepseek-v4.1-flash:cloud produced.

The user's reply was an eleven-sentence planning chant repeated twice, the
second copy cut off by the output budget.  ``_repetition_span`` cannot see it:
the scan caps the period at half the text, so a 118-character unit repeated
twice is never tried as a period, and even if it were, a long period needs
three copies and only two exist.  ``_sentence_loop_span`` exists for this case
and must catch it; the guards in chat() must fire on it; and the reply a user
would see must hold one copy of the chant, not two.
"""

import asyncio
import itertools
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.serving.chat import (  # noqa: E402
    chat, _repetition_span, _sentence_loop_span, _trim_sentence_loop,
)

# Exactly what the user pasted, whitespace as reported.
USER_LOOP = """Go.

(Producing.)

Let me write.

Now.

OK.

Let me write.

Go.

Alright.

Let me write the tool call.

Now.

OK.

Let me write.

Go.

(Producing.)

Let me write.

Now.

OK.

Let me write.

Go.

Alright.

Let me write the tool call.

Now.

OK."""

# The same chant with no truncation, to pin the two-full-copies case.
UNIT = ("Go. (Producing.) Let me write. Now. OK. Let me write. Go. Alright. "
        "Let me write the tool call. Now. OK. Let me write.")


def test_periodicity_scan_cannot_see_two_copies():
    """The blind spot this regression exists for, pinned as a fact.

    The character-periodicity scan returns nothing on both the truncated and
    the full two-copy text.  If this ever starts detecting, the sentence
    detector has a redundancy it did not have when written — fine — but the
    sentence detector must still agree.
    """
    assert _repetition_span(USER_LOOP) == (0, 0, 0)
    assert _repetition_span(UNIT + " " + UNIT) == (0, 0, 0)


def test_sentence_detector_sees_the_reported_loop():
    window, copies, first = _sentence_loop_span(USER_LOOP)
    assert window >= _SENT_LOOP_MIN_SENTENCES if False else window >= 3
    assert copies >= 2, (
        f"the reported loop was not seen as a repeat (window={window}, copies={copies})")


def test_sentence_detector_sees_two_full_copies():
    window, copies, first = _sentence_loop_span(UNIT + " " + UNIT)
    assert copies >= 2


def test_trim_keeps_one_copy_of_the_chant():
    trimmed = _trim_sentence_loop(USER_LOOP)
    assert trimmed.count("Let me write the tool call") == 1, (
        f"trim left {trimmed.count('Let me write the tool call')} copies of the chant")
    assert trimmed.count("Go.") == 2  # one per half-chant in the kept copy
    assert trimmed.strip().endswith("OK.")


# False positives the detector must not produce: emphasis, restated plans,
# and ordinary answers that mention something twice.
@pytest.mark.parametrize("text", [
    "I will not do that. I will not do that.",
    "very " * 6,
    "First we install. Then we test. First we install. Then we test.",
    "The task is done. I updated three files and ran the tests. All green.",
    ("Here is the plan.\n\n1. Read the config. 2. Update the timeout. "
     "3. Restart the service.\n\nReading the config. Updating the timeout. "
     "Restarting the service. That is the plan."),
])
def test_no_false_positives(text):
    window, copies, first = _sentence_loop_span(text)
    assert not (window and copies >= 2), (
        f"legitimate text flagged as a loop (window={window}, copies={copies}): {text[:60]!r}")


def _mock_completion(content, finish_reason="stop"):
    message = MagicMock()
    message.content = content
    message.tool_calls = None
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason
    completion = MagicMock()
    completion.choices = [choice]
    completion.usage.prompt_tokens = 0
    return completion


@pytest.mark.asyncio
async def test_chat_terminates_on_the_reported_loop():
    """The full chat() guard fires on the user's text and trims it."""
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=itertools.repeat(_mock_completion(USER_LOOP)))

    with patch("model.serving.chat.AsyncOpenAI", return_value=mock_client), \
         patch("model.serving.chat._resolve_model_id",
               new_callable=AsyncMock, return_value="test-model"):
        result = await asyncio.wait_for(chat(
            host="http://localhost:8000/v1",
            instruction="Write the file.",
            tool_registry=None,
            safety_queue=asyncio.Queue(),
            max_ack_continuations=0,
            verify_answers=False,
        ), timeout=30)

    assert result.count("Let me write the tool call") <= 1, (
        f"the loop was returned to the user "
        f"({result.count('Let me write the tool call')} copies of the chant)")
    assert mock_client.chat.completions.create.call_count <= 5