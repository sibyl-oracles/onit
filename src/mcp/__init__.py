"""MCP servers and prompts.

No eager re-export: ``import src.mcp`` is on the parent path of every
server module (``src.mcp.servers...``), so anything imported here is paid
for by every stdio child before it can start serving.
"""
