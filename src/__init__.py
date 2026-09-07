"""OnIt package.

``import src`` used to eagerly import the full application stack via
``from .onit import OnIt``. That was harmless for the CLI — which imports
``src.onit`` anyway — but not for the MCP server children spawned by
``python -m src.mcp.servers.run``: the ``-m`` bootstrap imports every parent
package first, so each stdio child paid for the whole stack (a2a, protobuf,
UI, model serving) before ``run.py`` could start serving. On a Raspberry Pi 4
that is ~13 s of imports per child, and the discovery retries respawn the
child on every attempt — five attempts per pass, two passes in a failing
session — for a server that should boot in about a second.

The class is therefore exposed lazily: attribute access on the package
triggers the import, so ``from src import OnIt`` still works everywhere, and
``__version__`` stays a plain module attribute because it is defined before
any heavy import and is read by ``src/ui/text.py`` at runtime.
"""

__version__ = "0.1.5a"

__all__ = ["__version__", "OnIt"]


def __getattr__(name: str):
    # PEP 562 module __getattr__: called only when the attribute is not found
    # by normal means, i.e. only for OnIt — __version__ resolves directly.
    if name == "OnIt":
        from .onit import OnIt
        return OnIt
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")