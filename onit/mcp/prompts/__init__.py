"""Package marker.

Nothing is re-exported here: ``onit.mcp.prompts`` is the parent of the
Prompts server module, and importing that module costs a FastMCP build.
Callers name ``onit.mcp.prompts.prompts`` directly.
"""
