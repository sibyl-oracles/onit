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

``\\rate`` — the text UI's verdict on an answer.

The web UI has had a thumbs-up for months.  The text UI has had none, and
the corpus says what that costs: **zero** explicit ratings across 1,010
recorded tasks, because the only way to leave one was to open a browser
(``docs/SELF_IMPROVEMENT.md`` §3.0, gap 3).  A signal that needs a
different application to arrive is not a signal the terminal has.

**Why this is worth having when the implicit signals exist.**  The implicit
signals are a heuristic over what the user typed next.  This is the user
saying, in one keystroke, that the answer was wrong.  The RFC calls the
explicit rating "highest value per bit" and then correctly notes that the
bits never arrive — so the job here is to make them arrive, not to make
them better.

**The shape of it.**  ``\\rate`` with no argument rates the newest answer.
``\\rate up`` / ``\\rate down`` say which way.  ``\\rate 3 up`` rates the
third answer of the session.  ``\\rate clear`` takes a verdict back.

A rating with recording off is not an error: the user was asked, and the
deployment declined to keep the answer.  Saying so is the honest reply.
"""

# Spellings of each verdict, and of "take it back".  The same set the web
# endpoint accepts, so a thumb is a thumb whichever application sent it.
#
# "1" and "-1" are deliberately absent: they are a turn number and a
# sign, and a bare "1" that meant "up" would be indistinguishable from
# "rate the first answer".  The words are unambiguous, so the words are
# what the command takes.
_UP = ("up", "good", "yes", "y", "+", "thumbs up")
_DOWN = ("down", "bad", "no", "n", "-", "thumbs down")
_CLEAR = ("clear", "none", "null", "undo", "x")


def _session_turn_count(session_path: str) -> int:
    """How many answers the session file holds."""
    count = 0
    try:
        with open(session_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
    except OSError:
        pass
    return count


def _parse_arg(arg: str) -> tuple:
    """Parse ``\\rate``'s argument.

    Returns ``(turn_or_None, rating_or_None, error)``.  ``turn`` is None
    for "the newest answer".  ``rating`` is +1, -1 or None (a retraction).
    ``error`` is a message to show, or "".
    """
    parts = (arg or "").strip().lower().split()
    if not parts:
        return None, 1, ""
    turn = None
    rating = None
    seen_verdict = False
    for part in parts:
        if part in _UP:
            rating = 1
            seen_verdict = True
        elif part in _DOWN:
            rating = -1
            seen_verdict = True
        elif part in _CLEAR:
            rating = None
            seen_verdict = True
        elif part.isdigit():
            if turn is not None:
                return None, None, ("\\rate takes one turn number. Run "
                                    "\\help for the form.")
            turn = int(part)
        else:
            return None, None, ("\\rate understands 'up', 'down' and "
                                "'clear', optionally with a turn number "
                                "(\\rate 3 down). Got '%s'." % part)
    if not seen_verdict:
        return turn, None, ("\\rate needs a verdict: 'up', 'down' or "
                            "'clear'. Run \\help for the form.")
    return turn, rating, ""


def cmd_rate(agent, arg: str) -> str:
    """Record a verdict on an answer in this session."""
    from ..learn import append_rating, recording_enabled

    turn, rating, error = _parse_arg(arg)
    if error:
        return error

    session_path = getattr(agent, "session_path", None)
    session_id = getattr(agent, "session_id", None)
    if not session_path or not session_id:
        return ("This session has no store to rate against — the agent was "
                "started without a session file.")

    config_data = getattr(agent, "config_data", None) or {}
    if not recording_enabled(config_data):
        return ("Ratings are not being recorded: this deployment has the "
                "learn level at 'off'. Set 'learn.autonomy' to 'observe' or "
                "higher to keep them.")

    total = _session_turn_count(session_path)
    if total == 0:
        return "No answer to rate yet."
    if turn is None:
        turn = total
    if turn < 1 or turn > total:
        return ("There is no answer %d in this session — it has %d."
                % (turn, total))

    path = append_rating(session_id=session_id, turn=turn, rating=rating,
                         comment=None, source="text",
                         config_data=config_data)
    if path is None:
        return ("The rating could not be written — the learned path is not "
                "writable.")

    if rating is None:
        return "Cleared the verdict on answer %d." % turn
    word = "good" if rating > 0 else "wrong"
    return ("Recorded: answer %d was %s. It is in the trajectory store, so "
            "the loops can learn from it." % (turn, word))


def render_rate_help() -> str:
    """The ``\\rate`` entry for the help listing."""
    return ("  \\rate [up|down|clear] [n]  Rate an answer: the newest one, "
            "or number n.\n"
            "                                A verdict is the strongest outcome "
            "signal there is —\n"
            "                                it is the one thing the implicit "
            "signals cannot infer.")
