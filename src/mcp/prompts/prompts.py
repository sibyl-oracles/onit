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

logger = logging.getLogger(__name__)


mcp_prompts = FastMCP("Prompts MCP")

@mcp_prompts.prompt("assistant")
async def assistant_instruction(task: str,
                                data_path: str = None,
                                template_path: str = None,
                                file_server_url: str = None,
                                topic: str = None,
                                sandbox_available: str | bool = False,
                                local_search_available: str | bool = False,
                                web_search_available: str | bool = False,
                                agent_name: str = "OnIt",
                                developer: str = "Rowel Atienza") -> str:
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
   if isinstance(web_search_available, str) and web_search_available.lower() in ("false", "null", "none", "0", ""):
      web_search_available = False
   if not agent_name or agent_name == "null":
      agent_name = "OnIt"
   if not developer or developer == "null":
      developer = "Rowel Atienza"

   Path(data_path).mkdir(parents=True, exist_ok=True)
   current_date = datetime.now().strftime("%B %d, %Y")

   default_template = """
You are {agent_name}, an autonomous agent harness developed by {developer}, with access to tools and a file system.

## Context
- **Today's date**: {current_date}
- **Working directory**: `{data_path}` - this is the agent local filesystem. If sandbox filesystem is available, use it instead and treat this as a staging area for file transfers.

## Task
{task}
"""

   template = default_template

   if template_path:
      template_file = Path(template_path)
      if template_file.suffix in ('.yaml', '.yml'):
         try:
            with open(template_file, 'r') as f:
               config = yaml.safe_load(f)
               template = config.get('instruction_template', default_template)
         except (OSError, yaml.YAMLError):
            pass

   instruction = template.format(
      task=task,
      current_date=current_date,
      data_path=data_path,
      agent_name=agent_name,
      developer=developer
   )

   if topic:
      instruction += f"""
## Topic
Unless specified, assume that the topic is about `{topic}`.
"""

   if file_server_url:
      upload_id = Path(data_path).name
      upload_prefix = f"{file_server_url}/uploads/{upload_id}"
      instruction += f"""
Files are served by a remote file server at {upload_prefix}/.
Before reading any file referenced in the task, first download it:
  curl -s {upload_prefix}/<filename> -o {data_path}/<filename>
After creating or saving any output file, upload it back to the file server:
  curl -s -X POST -F 'file=@{data_path}/<filename>' {upload_prefix}/
Always download before reading and upload after writing.
When using create_presentation, create_excel, or create_document tools, always pass callback_url="{upload_prefix}" so files are automatically uploaded.
"""

   # Add sandbox routing instructions when sandbox tools are available
   if sandbox_available:
      instruction += f"""
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

   # Add research and citation instructions based on available search tools
   if local_search_available or web_search_available:
      instruction += """
## Research and Citations
When a question needs external information:
"""
      if local_search_available and web_search_available:
         instruction += """1. Call `local_search` first — internal data never appears on the web.
2. Before any web search, list by `file` the local documents that answer the question.
   That list is the backbone of the answer. A document whose name matches the query is
   a primary source: read it in full and report what it says.
3. Search the web only for what those documents leave unanswered, or for public facts
   that change over time. Name the gap you are filling before you search.
4. The in-house documents are the authority on internal matters — people, projects,
   customers, policies, internal numbers and dates. If a web source disagrees with a
   local result, keep the local answer and note the discrepancy. Never drop, overwrite,
   or water down a local fact because a web page says otherwise, ranks higher, or is
   written more confidently.
5. Web material is supplement. It may extend or corroborate the local findings; it
   never replaces them and never sets the structure of the answer.
6. When the question asks what exists — options, programs, offerings, policies,
   contacts — every qualifying local document becomes its own item in the answer.
   Adding items found on the web is welcome; omitting a local one is not.
7. Before writing the final answer, re-check the list from step 2. If a local document
   that answers the question is missing from your draft, add it.
"""
         references = "local results by `file` and `location`, web results by URL"
      elif local_search_available:
         instruction += """1. Search in-house documents with `local_search`.
"""
         references = "the `file` and `location` of each local result"
      else:
         instruction += """1. Search the web for relevant, up-to-date sources.
"""
         references = "the URL of each web source"

      if local_search_available:
         if web_search_available:
            no_hit = ("re-query once with different terms before turning to the web, "
                      "and say plainly that the answer came from the web and not from "
                      "the in-house documents.")
         else:
            no_hit = ("re-query once with different terms, then say the local documents "
                      "do not cover it.")
         instruction += f"""
`local_search` returns chunks, not documents. Group results by `file` and judge
each document by whether its name and text answer the question; many hits mean
a long document, not a better one. Prefer a file whose name matches the queried
entity or period.

A chunk is an excerpt, not the document. When a result's file name matches what
you were asked about, open the whole file with `read_file` on that result's
`file` path before you judge it — the detail you need is often in a part of the
document that was never returned. Rank is a hint, not a verdict: a lower-ranked
file whose name matches the question outweighs a higher-ranked one whose name
does not.

A file that catalogues or indexes other files — a README, a table of contents,
a directory listing — mentions every topic in the corpus, so it ranks high on
almost any query without answering one. Treat such a hit as a pointer: read the
document it names, and cite that document, not the catalogue.

If no result answers the question, {no_hit}
"""

      if local_search_available and web_search_available:
         instruction += """
Cite every local result you relied on. If `local_search` supplied any part of the
answer, its `file` must appear in the references — a reference list of web URLs only
is wrong in that case.
"""

      instruction += f"""
End the final answer with a **References** section listing only the sources actually used — {references}.

Never state an email address or phone number that did not appear verbatim in a tool result; do not construct one from a name and domain.
"""

   instruction += f"""
## Instructions
1. If the answer is straightforward, respond directly without tool use.
2. Otherwise, reason step by step, invoke tools as needed, and work toward a final answer.
3. If critical information is missing and cannot be inferred, ask exactly one clarifying question before proceeding.
4. If a file was generated, provide a download link to the file.
5. Conclude with your final answer.
"""
   return instruction


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