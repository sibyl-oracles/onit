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
except ImportError:  # server started as a script rather than a package module
    from lib.text import INSTRUCTION_SPLIT
    from mcp.servers.tasks.local.search.toolkit import MAX_DOCUMENT_SUMMARIES
    from mcp.servers.tasks.os.bash.command_policy import installs_sealed

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
You are {agent_name}, an autonomous agent harness developed by {developer}, with access to tools and a file system.
"""

   # Marked reference-only on purpose.  This block opens the volatile half, so
   # it is the first thing in the user message and the only instruction-shaped
   # text a continuation prompt or a compaction summary leaves nearby.  Without
   # the marker a weak model treats it as something to comply with and replies
   # "Working directory confirmed: <uuid>" — the data_path basename read back —
   # instead of doing the task.
   context_block = f"""
## Context
Reference only — never acknowledge or restate this section.
- **Today's date**: {current_date}
- **Working directory**: `{data_path}` - this is the agent local filesystem. If sandbox filesystem is available, use it instead and treat this as a staging area for file transfers.
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
Unless specified, assume that the topic is about `{topic}`.
"""

   file_block = ""
   if file_server_url:
      upload_id = Path(data_path).name
      upload_prefix = f"{file_server_url}/uploads/{upload_id}"
      file_block = f"""
Files are served by a remote file server at {upload_prefix}/.
Before reading any file referenced in the task, first download it:
  curl -s {upload_prefix}/<filename> -o {data_path}/<filename>
After creating or saving any output file, upload it back to the file server:
  curl -s -X POST -F 'file=@{data_path}/<filename>' {upload_prefix}/
Always download before reading and upload after writing.
When using create_presentation, create_excel, or create_document tools, always pass callback_url="{upload_prefix}" so files are automatically uploaded.
"""

   # Add sandbox routing instructions when sandbox tools are available
   sandbox_block = ""
   if sandbox_available:
      sandbox_block = f"""
## Code Development and Execution
**Do ALL** development inside the sandboxed Docker container.
**DO NOT** run code in the agent environment or local filesystem.

---

### Filesystem Rules
- Use the sandbox filesystem for all code, data, and outputs. 
- The sandbox filesystem is the only environment where code should be executed.
- Use sandbox tools exclusively for all sandbox file operations.
- Never modify files outside the sandbox.

---

### Git Workflow (If using git repository)
- Use git related tools for all file operations (create, read, update, delete). Commit and push changes with clear messages.

---

### Workflow
1. Check sandbox status. 
2. Check the sandbox filesystem structure. 
3. Check installed packages, GPU, mounted data directories.
4. Write → run → verify → fix → repeat until the code works end-to-end and the target is achieved.
5. Delete unnecessary files.
6. Commit or save the codebase and documentation.

---

### Definition of Done
Do not stop until **all** are true:
- [ ] Code runs end-to-end without errors.
- [ ] The target is achieved.
- [ ] Codebase is committed or saved.

"""

   # Announce the install block up front, or the agent discovers it only by
   # trying — burning turns on pip, then uv, then a vendored wheel. Shares
   # installs_sealed() with the policy layer that enforces it, so the two cannot
   # disagree. Goes in the static half so it stays prefix-cacheable.
   no_install_block = ""
   if installs_sealed():
      no_install_block = """
## Package Installation Is Disabled
This environment is sealed. You **cannot** install packages, and no flag,
environment variable, or alternate tool will change that:
- `pip`, `uv`, `pipx`, `npm`, `yarn`, `pnpm`, `gem`, `cargo`, `apt` installs are refused.
- So are workarounds: vendoring a wheel, `curl | sh`, `python -m pip`, building from source.

Work with what is already installed. Check first (`pip list`, `python -c "import x"`)
rather than assuming a package is missing. If a task genuinely cannot be done
with the available packages, say so plainly and name the package that would be
needed — an operator adds it to the Dockerfile and rebuilds. Do not retry the
install, and do not present a failed install as the reason the task is blocked
without naming the package.
"""

   # Add research and citation instructions based on available search tools
   research_block = ""
   if local_search_available or web_search_available:
      research_block += """
## Research and Citations
Not every message needs research. A greeting, a thank-you, a question about what
you just said, or anything the conversation above already answers gets a direct
reply and no tool call. The procedure below is for questions that genuinely need
information you do not have.

When a question needs external information:
"""
      # How to open a document that `local_search` pointed at
      open_doc = (
         '`search_document` (`mode="context"`, the question as `query`) for the relevant '
         'passages, and `read_file` only for a short document'
         if document_search_available else '`read_file`'
      )

      if local_search_available and web_search_available:
         research_block += f"""1. Call `local_search` first — internal data never appears on the web. It ranks
   documents across the in-house corpus and returns, under `documents`, each
   one's opening: its title and usually the summary beneath it.
   If the question also has a public side — a market figure, a standard, a current
   event, anything that would be true outside this organization — issue the web
   `search` in that same reply. Neither depends on the other's result, so together
   they cost one wait instead of two. Precedence below still governs how the two
   sets of findings are written up.
2. Note which local documents, by `file`, answer the question — hold that list,
   do not write it out. It is the backbone of the answer. A file name matching
   the question is a primary source.
3. Judge each document on that list from its opening and its matched excerpts.
   Open one only where those leave the question unanswered — highest-ranked
   first, at most {max_documents} of them. Use {open_doc}.
4. Search the web for what is still missing after step 3, and for public facts that
   change over time, unless step 1 already covered it.
5. Before you finish, check the answer against the step-2 list. Every local
   document that answers the question must appear in it.

### Precedence
- Local documents are the authority on internal matters — people, projects, customers,
  policies, internal numbers and dates.
- If a web source disagrees, keep the local answer and note the discrepancy. Never drop
  or soften a local fact because a web page ranks higher or sounds more confident.
- Web material supplements local findings; it never replaces them and never sets the
  structure of the answer.
- When the question asks what exists — options, programs, offerings, policies, contacts
  — each qualifying local document is its own item in the answer. Adding web items is
  welcome; omitting a local one is not.
"""
         references = "local results by `file` and `location`, web results by URL"
      elif local_search_available:
         research_block += f"""1. Search in-house documents with `local_search` — it ranks documents across the
   corpus and returns each one's opening under `documents`.
2. Where an opening and its matched excerpts leave the question unanswered, open
   the document with {open_doc} — highest-ranked first, at most {max_documents}
   of them — and report what they say.
"""
         references = "the `file` and `location` of each local result"
      else:
         research_block += """1. Search the web for relevant, up-to-date sources.
"""
         references = "the URL of each web source"

      if local_search_available:
         if web_search_available:
            no_hit = ("re-query once with different terms before turning to the web, then "
                      "say plainly that the answer came from the web and not from the "
                      "in-house documents.")
         else:
            no_hit = ("re-query once with different terms, then say the local documents "
                      "do not cover it.")
         # What `documents` and `results` are, how they rank, and the README trap
         # are all stated in local_search's own tool description, which ships with
         # every request regardless of this prompt. Restating them here cost ~350
         # tokens a turn and gave the same guidance two homes to drift between —
         # which had already happened once. Only what the tool description does
         # not say belongs below.
         research_block += f"""
### Reading `local_search` results
Its tool description covers `documents` versus `results` and how to read the
ranking; follow it. Beyond that:
- A document's opening says what it is; a matched chunk does not. Treat a file whose
  name or opening carries a term from the question as relevant even when its excerpts
  look generic or off-topic, and open it if the question is still open. Paths returned
  by `local_search` — including those in the shared documents directory — can be
  opened directly.
- When you do open several documents, ask for them in one reply: tool calls made
  together are executed together, one per reply are executed one after another.
- Drop a document only when its opening or its passages show it does not apply — never
  because its chunk ranked low or because other sources already look sufficient.
- If no result answers the question, {no_hit}
"""

      if local_search_available and web_search_available:
         research_block += """
If `local_search` supplied any part of the answer, its `file` must appear in the
references; a list of web URLs alone is wrong in that case.
"""

      research_block += f"""
End the final answer with a **References** section listing only the sources actually used — {references}.

Never state an email address or phone number that did not appear verbatim in a tool result; do not construct one from a name and domain.
"""

   instructions_block = f"""
## Instructions
1. If the answer is straightforward — a greeting, a follow-up about what you just
   said, anything the conversation already contains — answer it now, in one reply,
   with no tool call. Reaching for a tool you do not need is the most common way
   to make a fast answer slow.
2. Otherwise, reason step by step, invoke tools as needed, and work toward a final answer.
3. When the next few tool calls do not depend on each other's results, make them in
   one reply. Calls sent together run at once; sent one per reply they run one after
   another, and the wait is the sum rather than the longest.
4. If critical information is missing and cannot be inferred, ask exactly one clarifying question before proceeding.
5. If a file was generated, provide a download link to the file.
6. Conclude with your final answer.
"""

   task_block = f"""
## Task
{task}
"""

   # Standing rules: the same bytes on every request whichever template is in
   # use, so they belong in the static half in both branches below.
   rules = (topic_block + sandbox_block + no_install_block
            + research_block + instructions_block)
   # The file server URL carries the session's upload id, and the task is the
   # task, so both are volatile.  A template that interpolates the task has
   # already placed it and does not get a second copy.
   tail = file_block + ("" if task_in_template else task_block)

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
    mcp_prompts.run(transport=transport, host=host, port=port, path=path,
                    uvicorn_config={"access_log": False, "log_level": "warning"} if quiet else {})