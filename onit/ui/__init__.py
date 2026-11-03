"""The user interfaces: terminal chat, web API, voice.

``ChatUI`` is re-exported for callers outside the harness that want the
terminal front end in one import.  The harness itself no longer uses this
re-export: ``onit.py`` imports ``ChatUI`` from ``onit.ui.text`` inside the
function that builds it, so ``import onit.onit`` pulls in no front end at
all.  See ``onit/contract/`` for the surface both sides share.

It used to sit beside two other names that had no caller left:
``text_prompt`` (the pre-rewrite prompt loop, already marked ``### Legacy``
in ``text.py``) and ``split_message`` (which split messages for the Viber
gateway and was only ever exercised by that gateway's tests).
"""

from .text import ChatUI

__all__ = ["ChatUI"]
