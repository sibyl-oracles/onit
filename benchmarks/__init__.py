"""OnIt capability benchmark suite.

Drives the real OnIt agent (``OnIt.process_task``) through standard public
benchmarks using `Inspect AI <https://inspect.aisi.org.uk/>`_ as the backbone.

This package is intentionally kept out of ``src/`` and out of the default
``pytest`` run (it is slow, networked, and costs tokens). Run it explicitly via
``benchmarks/run.py`` or the ``bench-*`` Makefile targets.
"""
