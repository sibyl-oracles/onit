"""MCP servers and prompts.

No eager re-export: ``import onit.mcp`` is on the parent path of every
server module (``onit.mcp.servers...``), so anything imported here is paid
for by every stdio child before it can start serving.
"""
