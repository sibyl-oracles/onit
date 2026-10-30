"""Package marker; ``run`` is the stdio entry point ``servers/run.py`` calls.

``from .mcp_server import run`` is a real re-export, not a dead import:
the config names the *package* (``module: tasks.tools``), and ``run.py``
does ``getattr(server_module, "run")``, so the name has to be on the
package.  ``__all__`` says so to a reader and to a linter.
"""
from .mcp_server import run

__all__ = ["run"]
