"""Verify the anchored rm-on-root patterns via the real policy module."""

import sys

sys.path.insert(0, __file__.rsplit("/test/", 1)[0] + "/..")
import onit.mcp.servers.tasks.os.bash.mcp_server as m

cases = [
    # (command, should_block)
    ("rm -rf /tmp/onit/data/abc123/scratch", False),
    ("rm -r /tmp/onit/data/abc123/scratch", False),
    ("rm -fr /Users/rowel/onit/data/s1/out", False),
    ("rm -f /tmp/onit/data/abc123/x.txt", False),
    ("rm /tmp/onit/data/abc123/x.txt", False),
    ("rm -rf data/abc123/scratch", False),
    ("rm -rf ./scratch", False),
    ("rm -rf /", True),
    ("rm -rf /*", True),
    ("rm -r /", True),
    ("rm -fr /", True),
    ("rm /", True),
    ("rm /*", True),
    ("rm -rf / ", True),
    ("rm  -rf  /", True),
    ("rm -rf / --no-preserve-root", True),
    ("rm -rf /tmp /", True),
    ("rm -rf /tmp /*", True),
    ("ls && rm -rf /", True),
    ("echo hi; rm -fr /", True),
]

fails = 0
for cmd, should_block in cases:
    blocked = None
    for pattern, description in m._ALWAYS_BLOCKED_PATTERNS:
        if pattern.search(cmd):
            blocked = description
            break
    if blocked is None:
        for pattern, description in m._BLOCKED_PATTERNS:
            if pattern.search(cmd):
                blocked = description
                break
    ok = (blocked is not None) == should_block
    if not ok:
        fails += 1
    print(f"{'OK ' if ok else 'FAIL'} blocked={blocked!s:16} want_block={should_block!s:5}  {cmd}")

print("\nFAILURES:", fails)
