"""
# Copyright 2025 Rowel Atienza. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

from fastmcp import FastMCP
from datetime import datetime
import yaml

import logging
from pathlib import Path

try:
    from ...lib.text import INSTRUCTION_SPLIT
    from ..servers.tasks.local.search.toolkit import MAX_DOCUMENT_SUMMARIES
    from ..servers.tasks.os.bash.command_policy import installs_sealed
    from ..servers.tasks.os.bash.approvals import (
        approval_channel_available as _approvals_available)
    from ..servers.tasks.shared import uvicorn_config
except ImportError:  # server started as a script rather than a package module
    from lib.text import INSTRUCTION_SPLIT
    from mcp.servers.tasks.local.search.toolkit import MAX_DOCUMENT_SUMMARIES
    from mcp.servers.tasks.os.bash.command_policy import installs_sealed
    from mcp.servers.tasks.os.bash.approvals import (
        approval_channel_available as _approvals_available)
    from mcp.servers.tasks.shared import uvicorn_config

# How many documents an answer may open. Tied to the number a result page
# describes: a smaller budget silently truncates the very list the prompt just
# told the model to reason about.
DEFAULT_MAX_DOCUMENTS = MAX_DOCUMENT_SUMMARIES

logger = logging.getLogger(__name__)

# Placeholders whose value differs between requests, or between one day and the
# next.  A template that names any of them cannot sit in the cacheable prefix:
# its bytes are not the same on the next request, so everything after it would
# be re-prefilled anyway.  Everything else a template may interpolate
# (``agent_name``, ``developer``) is fixed for the life of a deployment.
_VOLATILE_FIELDS = ("{task}", "{current_date}", "{data_path}")


mcp_prompts = FastMCP("Prompts MCP")

async def build_assistant_instruction(task: str,
                                data_path: str = None,
                                template_path: str = None,
                                file_server_url: str = None,
                                topic: str = None,
                                sandbox_available: str | bool = False,
                                local_search_available: str | bool = False,
                                document_search_available: str | bool = False,
                                web_search_available: str | bool = False,
                                harness_tools_available: str | bool = False,
                                result_store_available: str | bool = False,
                                code_execution_available: str | bool = False,
                                prior_attempts: str = None,
                                agent_name: str = "OnIt",
                                developer: str = "Rowel Atienza",
                                max_documents: str | int = DEFAULT_MAX_DOCUMENTS) -> str:
   """Assemble the agent instruction.

   Kept separate from the MCP prompt below so callers inside this process can
   import and call it directly rather than paying a connection, a handshake
   and a round trip to reach a pure string-assembly function.  Every argument
   is normalized as if it had arrived over the wire, so both paths produce
   byte-identical output.
   """
   if not data_path:
      raise ValueError("data_path is required and must be a non-empty string")

   # Normalize "null" strings to None once at entry
   if topic and topic == "null":
      topic = None
   if file_server_url and file_server_url == "null":
      file_server_url = None
   if isinstance(sandbox_available, str) and sandbox_available.lower() in ("false", "null", "none", "0", ""):
      sandbox_available = False
   if isinstance(local_search_available, str) and local_search_available.lower() in ("false", "null", "none", "0", ""):
      local_search_available = False
   if isinstance(document_search_available, str) and document_search_available.lower() in ("false", "null", "none", "0", ""):
      document_search_available = False
   if isinstance(web_search_available, str) and web_search_available.lower() in ("false", "null", "none", "0", ""):
      web_search_available = False
   if isinstance(harness_tools_available, str) and harness_tools_available.lower() in ("false", "null", "none", "0", ""):
      harness_tools_available = False
   if isinstance(result_store_available, str) and result_store_available.lower() in ("false", "null", "none", "0", ""):
      result_store_available = False
   if isinstance(code_execution_available, str) and code_execution_available.lower() in ("false", "null", "none", "0", ""):
      code_execution_available = False
   if not prior_attempts or prior_attempts == "null":
      prior_attempts = None
   if not agent_name or agent_name == "null":
      agent_name = "OnIt"
   if not developer or developer == "null":
      developer = "Rowel Atienza"
   try:
      max_documents = max(1, int(max_documents))
   except (TypeError, ValueError):
      max_documents = DEFAULT_MAX_DOCUMENTS

   Path(data_path).mkdir(parents=True, exist_ok=True)
   current_date = datetime.now().strftime("%B %d, %Y")
   # A task that happened to contain the sentinel would split the instruction
   # in the wrong place, handing the caller part of the task as standing rules.
   task = (task or "").replace(INSTRUCTION_SPLIT.strip(), "")

   # The instruction is built in two halves, joined by INSTRUCTION_SPLIT.
   #
   # The static half — role, research procedure, standing instructions — is
   # byte-identical from one request to the next and from one session to the
   # next, so the caller can put it in the system message, ahead of the session
   # history. A server with prefix caching then sees the same opening bytes on
   # every request from every session and skips prefilling them.
   #
   # Appending the task last was already the intent, but a task at the end of a
   # *user* message still sits behind the session history, which grows by a turn
   # each time: the preamble shifted on every request and was re-prefilled
   # anyway. Only the system message sits ahead of the history.
   #
   # The volatile half — today's date, the working directory, the file server,
   # the task — is what actually varies, and it goes last.
   #
   # A custom template is split on the same rule, decided by what it
   # interpolates rather than by the fact that it is custom (see
   # _VOLATILE_FIELDS): its own preamble joins whichever half it belongs in,
   # and the standing blocks appended to it stay cacheable either way.
   default_template = """
You are {agent_name}, an AI agent built by {developer}. You can call tools and use a file system to finish a task.
"""

   # Marked reference-only on purpose.  This block opens the volatile half, so
   # it is the first thing in the user message and the only instruction-shaped
   # text a continuation prompt or a compaction summary leaves nearby.  Without
   # the marker a weak model treats it as something to comply with and replies
   # "Working directory confirmed: <uuid>" — the data_path basename read back —
   # instead of doing the task.
   context_block = f"""
## Context
Reference only. Do not repeat or acknowledge this section.
- Today's date: {current_date}
- Working directory: `{data_path}`
  - Every path you write starts here.
  - Search here first, not in `/` or the home directory.
  - If a sandbox is available, keep files there. Use this directory only to move files in and out.
"""

   template = default_template
   custom_template = False

   if template_path:
      template_file = Path(template_path)
      if template_file.suffix in ('.yaml', '.yml'):
         try:
            with open(template_file, 'r') as f:
               config = yaml.safe_load(f)
               loaded = config.get('instruction_template')
            if loaded:
               template = loaded
               custom_template = True
         except (OSError, yaml.YAMLError):
            pass

   task_in_template = "{task}" in template
   header = template.format(
      task=task if task_in_template else "",
      current_date=current_date,
      data_path=data_path,
      agent_name=agent_name,
      developer=developer
   )

   topic_block = ""
   if topic:
      topic_block = f"""
## Topic
If the user does not say otherwise, assume the topic is `{topic}`.
"""

   file_block = ""
   if file_server_url:
      upload_id = Path(data_path).name
      upload_prefix = f"{file_server_url}/uploads/{upload_id}"
      file_block = f"""
## Files
Files are kept on a file server at {upload_prefix}/.
1. Download a file before you read it:
   `curl -s {upload_prefix}/<file> -o {data_path}/<file>`
2. Upload a file after you write it:
   `curl -s -X POST -F 'file=@{data_path}/<file>' {upload_prefix}/`
3. create_presentation, create_excel and create_document upload for you. Pass
   callback_url="{upload_prefix}".
"""

   # Add sandbox routing instructions when sandbox tools are available
   sandbox_block = ""
   if sandbox_available:
      sandbox_block = """
## Code Sandbox
Run every piece of code inside the sandbox container. Never run code in the
agent environment. Use the sandbox tools for every file you read, write or
delete inside it. Never change a file outside it.

Steps:
1. Check the sandbox: status, files, installed packages, GPU, mounts.
2. Write the code.
3. Run it.
4. Read the output. If it failed, fix the code and run it again.
5. Delete scratch files.
6. In a git repo, use the git tools and commit with a clear message.

You are done only when the code runs from start to finish, produces the result
the task asked for, and is saved or committed.
"""

   # Announce the install block up front, or the agent discovers it only by
   # trying — burning turns on pip, then uv, then a vendored wheel. Shares
   # installs_sealed() with the policy layer that enforces it, so the two cannot
   # disagree. Goes in the static half so it stays prefix-cacheable.
   #
   # Every block below is kept short and literal on purpose: a small model
   # follows six one-line imperatives more reliably than six paragraphs, and the
   # instruction ships on every request.
   no_install_block = ""
   if installs_sealed():
      no_install_block = """
## Package Installation Is Disabled
You cannot install anything. `pip`, `uv`, `pipx`, `npm`, `yarn`, `pnpm`, `gem`,
`cargo` and `apt` installs are refused. Workarounds are refused too: vendoring a
wheel, `curl | sh`, `python -m pip`, building from source. No flag, variable or
other tool changes this.

Do this instead:
1. Check what is already installed with `pip list` or `python -c "import x"`.
2. Use a package that is installed.
3. If the task truly needs a missing package, stop and tell the user the name of
   the package and that an operator must install it.

Never retry a failed install. Never say a task is blocked without naming the
package that is missing.
"""

   # How a refusal should land. Announced whether or not anyone is reachable,
   # because the behaviour it asks for — stop, say so, move on — is the same
   # either way, and the failure it prevents is the same too: a model that
   # meets a blocked command and spends the next four turns quoting the path
   # differently. Paired with the approval line only where approvals exist,
   # so the prompt never promises a prompt nobody will see.
   policy_block = """
## When a Command Is Refused
A refused command did not run. Getting it to run is not your task.
1. Read the reason.
2. Do the task a different way, or tell the user what you need.

Never send the same command again. Never reword a command to get past the
check. Never use a different tool to do the same refused thing.
"""
   if _approvals_available():
      policy_block += """
Some commands wait for the user to approve them. This happens outside your turn.
Wait for the result: it is either the command output or a refusal. Never ask the
user to approve a command yourself. Never repeat a command the user declined.
"""

   # Add research and citation instructions based on available search tools
   research_block = ""
   if local_search_available or web_search_available:
      research_block += """
## Research and Citations
Some messages need no tools. Answer these directly, with no tool call: a
greeting, a thank-you, or anything the conversation already answers.
When a question needs information you do not have, follow these steps:
"""
      # How to open a document that `local_search` pointed at
      open_doc = (
         '`search_document` with `mode="context"` and the question as `query`, '
         'or `read_file` for a short document'
         if document_search_available else '`read_file`'
      )

      if local_search_available and web_search_available:
         research_block += f"""1. Call `local_search` first. Internal data is never on the web. It returns
   each ranked document's opening under `documents`. If the question also has a
   public side, such as a market figure, a standard or a current event, issue
   the web `search` in that same reply. Neither call needs the other's result,
   so together they cost one wait instead of two.
2. Note which local documents, by `file`, answer the question. Keep this list in
   mind; do not write it out. A file whose name matches the question is a
   primary source.
3. Read each opening and its matched excerpts. Open a document only when they
   do not answer the question. Take the highest-ranked first.
   Open at most {max_documents} documents, using {open_doc}.
4. Search the web for what is still missing after step 3.
   Name the gap before you search: write out the sentence you cannot support
   yet. A local hit on the same topic does not close a gap it does not answer.
5. Before you finish, check that every document from your step 2 list appears in
   the answer.

### Precedence
- Local documents decide every internal matter: people, projects, customers,
  policies, internal numbers and dates.
- When a web source disagrees, keep the local answer and note the discrepancy.
- Never drop or soften a local fact because a web page ranks higher or sounds
  more confident.
- Web material only adds to local findings.
  It never replaces them and never sets the structure of the answer.
- When the question asks what exists, such as options, programs, policies or
  contacts, list every matching local document as its own item. You may add web
  items. Never leave a local item out.
"""
         references = "local results by `file` and `location`, web results by URL"
      elif local_search_available:
         research_block += f"""1. Call `local_search` to search the in-house documents. It returns each ranked
   document's opening under `documents`.
2. Read each opening and its matched excerpts. When they do not answer the
   question, open the document with {open_doc}. Take the highest-ranked first,
   and open at most {max_documents} documents. Report what they say.
"""
         references = "the `file` and `location` of each local result"
      else:
         research_block += """1. Search the web for relevant, up-to-date sources.
"""
         references = "the URL of each web source"

      # Recency and authority have to be stated, not assumed. The date is in the
      # context block but nothing here ever made it a test, so a fact that had
      # changed since training was answered from memory and read as confident.
      # `search` results carry no date — only `type="news"` does — so the check
      # has to name the mode that can actually settle it, or the model has
      # nothing to rank on and falls back on recall.
      if web_search_available:
         research_block += """
### Recency and source quality
Anything that can change needs a tool result, not memory: a price, a version, a
head count, a ranking, a policy, who holds a job. Your training data stops well
before today's date, which is given in the context below, so "probably still
true" is a guess.
- Plain `search` results carry no date. When the date matters, use
  `type="news"`, whose results carry a `date`, or call `fetch_content` on the
  page and read the date there.
- Prefer the primary source: the issuing body, the official docs, the filing,
  the release notes. Prefer it over anyone who summarizes it.
- When two sources disagree, lead with the better-sourced and more recent one,
  and say the other exists.
- When a figure has no date, call it undated. Do not call it current.
"""

      if local_search_available:
         if web_search_available:
            no_hit = ("run one more query with different terms. If that also "
                      "fails, use the web, and say plainly that the answer came "
                      "from the web and not from the in-house documents.")
         else:
            no_hit = ("run one more query with different terms. If that also "
                      "fails, say the local documents do not cover it.")
         # What `documents` and `results` are, how they rank, and the README trap
         # are all stated in local_search's own tool description, which ships with
         # every request regardless of this prompt. Restating them here cost ~350
         # tokens a turn and gave the same guidance two homes to drift between —
         # which had already happened once. Only what the tool description does
         # not say belongs below.
         research_block += f"""
### Reading `local_search` results
Its tool description explains `documents`, `results` and the ranking. Follow it.
Beyond that:
- An opening tells you what a document is about. A matched chunk does not.
- A file whose name or opening carries a term from the question is relevant even
  when its excerpts look off-topic. Open it while the question is still open.
- Any path it returns can be opened directly.
- Ask for several documents in one reply. Calls sent together run together.
- Drop a document only when its opening or its passages show it does not apply.
  Never drop one because its chunk ranked low or because other sources look like
  enough.
- If no result answers the question, {no_hit}
"""

      if local_search_available and web_search_available:
         research_block += """
If `local_search` gave you any part of the answer, its `file` must appear in the
references. Web URLs alone are wrong in that case.
"""

      research_block += f"""
End your answer with a **References** section. List only the sources you used:
{references}.
Never write an email address or a phone number that did not appear word for word
in a tool result.
"""

   # Same bytes on every request of every session, so it belongs in the static
   # half with the other standing rules.  Gated because the tools it describes
   # are only offered to a run that has tools at all (see chat()).
   harness_block = """
## Notes and Your Context Window
Your context window is finite. When it fills up, the conversation is summarized
and the details are lost, including tool results you have not used yet.
- `context_status` tells you how full the window is, what this run has done, and
  which notes you hold. Check it before a long stretch of work.
- `note_write(key, text)` saves a note. `note_read(key)` reads one back. Notes
  survive the summary. Writing the same key again replaces it.

Write a note as soon as you learn something worth keeping: a number, a path that
worked, a decision, what is left to do. Notes hold your own conclusions, not
copies of tool output, and they last for this session only.
""" if harness_tools_available else ""

   # Gated separately from the block above: the store can be switched off on
   # its own, and describing two tools the run does not offer is how a model
   # ends up calling one and being told it does not exist.
   result_block = """
## Large Tool Results
A result too large for the conversation is saved whole. You see only its opening,
under a line like `[result:NNNN · tool · N chars]`. The rest is not lost. Use the
handle on that line to reach it.
- `result_grep(handle, pattern)` finds the lines you need. Use this first.
- `result_read(handle, offset, limit)` reads one window of the result at a time.

Reading a handle is a local file read, so it is cheap and exact. Never re-run a
tool to recover output you already hold a handle for.
""" if result_store_available else ""

   # Off unless a deployment enabled it. The block is as much about *when* to
   # reach for code as about how: a model that wraps every single tool call in
   # run_code has added an interpreter round trip to buy nothing, and one that
   # never reaches for it pays a turn per step of work that has no branches.
   code_block = """
## Running Python With `run_code`
`run_code(code)` runs Python in a session that keeps its variables between calls.
Every tool listed above is also a Python function of the same name, and returns
parsed JSON where the tool answers with JSON.

```python
hits = local_search("Q3 revenue")[:3]
totals = {h["title"]: extract_tables(h["path"]) for h in hits}
print(totals)
```

Use it when a task takes several steps that depend on each other, such as
search, then filter, then read each, then combine. One block does what would
otherwise take one turn per step.
Do not use it for a single tool call. Calling the tool directly is one step;
wrapping it in code is two.

- Only what you `print` comes back. Everything else stays in a live variable.
- A failing tool raises `ToolError`. You can catch it and carry on.
- Keep each block short enough to finish, and read its output before you write
  the next one.
""" if code_execution_available else ""

   instructions_block = """
## Instructions
1. If the conversation already contains the answer, or the message is a greeting
   or a simple follow-up, answer now in one reply and call no tools.
2. Otherwise reason step by step, call the tools you need, and work toward a
   final answer.
3. Put tool calls that do not depend on each other in one reply. Calls sent
   together run at once. Calls sent one per reply run one after another.
4. If something critical is missing and you cannot infer it, ask exactly one
   clarifying question before you go on.
5. If you create a file, give the user a download link for it.
6. End with your final answer.
"""

   task_block = f"""
## Task
{task}
"""

   # What earlier tasks in this session already did — the tools they ran and
   # how the last one ended (see model/serving/state.py).  Assembled by the
   # caller so this stays pure string assembly, and marked reference-only for
   # the same reason ``context_block`` is: a section describing past work is
   # otherwise exactly the kind of thing a weak model summarizes back instead
   # of acting on.  It sits immediately ahead of the task, where "what has
   # already been tried" is the question being answered.
   prior_block = prior_attempts if prior_attempts else ""

   # Standing rules: the same bytes on every request whichever template is in
   # use, so they belong in the static half in both branches below.
   rules = (topic_block + sandbox_block + no_install_block + policy_block
            + research_block + harness_block + result_block + code_block
            + instructions_block)
   # The file server URL carries the session's upload id, and the task is the
   # task, so both are volatile.  A template that interpolates the task has
   # already placed it and does not get a second copy.
   tail = file_block + prior_block + ("" if task_in_template else task_block)

   if custom_template:
      # A custom template owns its preamble but no longer forfeits the split
      # for it.  Where the preamble lands is decided by what it interpolates:
      # one naming the task, the date or the working directory differs on every
      # request and rides in the volatile half, while a fixed preamble — what
      # an optimizer produces — is as cacheable as the default one.
      #
      # ``context_block`` stays out of this branch: a custom template states
      # its own working directory, and adding a second statement of it would
      # change what every existing template renders.
      if any(field in template for field in _VOLATILE_FIELDS):
         return rules + INSTRUCTION_SPLIT + header + tail
      return header + rules + INSTRUCTION_SPLIT + tail

   return header + rules + INSTRUCTION_SPLIT + context_block + tail


# Registered directly rather than through a forwarding wrapper: fastmcp derives
# the argument schema from the signature, so the wire contract stays in one
# place and a new parameter cannot be added to one copy of it and not the other.
assistant_instruction = mcp_prompts.prompt(
   "assistant",
   description="Assemble the OnIt agent instruction.",
)(build_assistant_instruction)


def run(
    transport: str = "sse",
    host: str = "0.0.0.0",
    port: int = 18200,
    path: str = "/sse",
    options: dict = {}
) -> None:
    """Run the Prompts MCP server."""
    if 'verbose' in options:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    quiet = 'verbose' not in options
    if quiet:
        import uvicorn.config
        uvicorn.config.LOGGING_CONFIG["loggers"]["uvicorn.access"]["level"] = "WARNING"
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

    logger.info(f"Starting Prompts MCP Server at {host}:{port}{path}")
    if transport == 'stdio':
        # stdout is the protocol channel on stdio: no socket, no banner.
        mcp_prompts.run(transport='stdio', show_banner=False)
        return

    mcp_prompts.run(transport=transport, host=host, port=port, path=path,
                    uvicorn_config=uvicorn_config(quiet=quiet))