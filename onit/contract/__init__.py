"""The harness/UI boundary: the only module both sides may import.

The harness (``onit/onit.py``, ``onit/model/serving/chat.py``) and every
front end (``onit/ui/text.py``, ``onit/ui/api.py``, and any future native
app) depend on this package and on nothing of each other's.  Keeping the
shared surface here is what lets the harness change without breaking a UI,
and a UI change without touching the harness.

What belongs here:

``ChatUIProtocol``
    What ``chat()`` may call on the UI it is handed.  Implemented by
    ``ChatUI`` (terminal) and ``StreamingAdapter`` (callback clients).

``UISettings``
    The read-only slice of the agent's configuration a UI is allowed to
    see, so a front end asks the harness for settings instead of reaching
    into its attributes with ``getattr``.

What does not belong here: anything that imports ``onit.ui``.  This package
must stay importable with no front end installed.
"""

from .protocol import ChatUIProtocol
from .settings import UISettings

__all__ = ["ChatUIProtocol", "UISettings"]
