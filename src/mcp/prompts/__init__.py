"""Package marker.

Nothing is re-exported here: ``src.mcp.prompts`` is the parent of the
Prompts server module, and importing that module costs a FastMCP build.
Callers name ``src.mcp.prompts.prompts`` directly.
"""
