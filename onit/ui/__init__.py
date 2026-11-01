"""The user interfaces: terminal chat, web API, voice.

``ChatUI`` is re-exported because ``onit.py`` imports it from the package
(``from .ui import ChatUI``).  It used to sit beside two other names that
had no caller left: ``text_prompt`` (the pre-rewrite prompt loop, already
marked ``### Legacy`` in ``text.py``) and ``split_message`` (which split
messages for the Viber gateway and was only ever exercised by that
gateway's tests).
"""

from .text import ChatUI

__all__ = ["ChatUI"]
