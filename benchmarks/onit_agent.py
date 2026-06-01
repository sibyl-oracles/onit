"""Native-tools mode for tool-calling fidelity benchmarks (Phase 4 — scaffold).

For BFCL / tau-bench we want Inspect to drive the tool loop directly so it can
score tool *selection* and *argument* accuracy. Rather than redefining tools, we
re-expose OnIt's already-discovered tool schemas (from its MCP tool registry) as
Inspect tools, and bridge tool calls back to OnIt's registry for execution.

This is distinct from ``benchmarks.onit_provider`` (full-stack mode), which lets
OnIt run its own tool loop and only scores the final answer.

Phase 4 implementation outline:
    1. Build the OnIt agent (reuse ``onit_provider._build_agent_blocking``) to
       obtain ``agent.tool_registry`` (name -> schema + invoker).
    2. For each registry entry, construct an Inspect ``@tool`` whose JSON schema
       mirrors the OnIt schema and whose body awaits the registry's invoker.
    3. Return an Inspect ``react`` agent over those tools, using the standard
       ``onit/<label>`` model for the underlying LLM.
"""

from __future__ import annotations


def onit_native_agent():
    """Return an Inspect agent exposing OnIt's tools natively. (Phase 4.)"""
    raise NotImplementedError(
        "Native-tools mode is a Phase 4 deliverable. See module docstring for "
        "the implementation outline; reuse onit_provider._build_agent_blocking "
        "to obtain the tool registry."
    )
