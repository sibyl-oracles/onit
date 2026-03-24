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
                                documents_path: str = None,
                                topic: str = None,
                                sandbox_available: str | bool = False) -> str:
   if not data_path:
      raise ValueError("data_path is required and must be a non-empty string")

   # Normalize "null" strings to None once at entry
   if topic and topic == "null":
      topic = None
   if file_server_url and file_server_url == "null":
      file_server_url = None
   if documents_path and documents_path == "null":
      documents_path = None
   if isinstance(sandbox_available, str) and sandbox_available.lower() in ("false", "null", "none", "0", ""):
      sandbox_available = False

   Path(data_path).mkdir(parents=True, exist_ok=True)
   current_date = datetime.now().strftime("%B %d, %Y")

   default_template = """
You are an autonomous agent with access to tools and a file system.

## Context
- **Today's date**: {current_date}
- **Working directory**: `{data_path}` - this is the agent local filesystem.

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
      data_path=data_path
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

   if documents_path:
      instruction += f"""
## Relevant Information
Search and read related documents (PDF, TXT, DOCX, XLSX, PPTX, and Markdown (MD)) in `{documents_path}`.
Search the web for additional information **if and only if** above documents are insufficient to complete the task.
"""

   # Add sandbox routing instructions when sandbox tools are available
   if sandbox_available:
      instruction += f"""
## Code Development and Execution

**Do ALL** development inside the sandboxed Docker container.
**DO NOT** run code in the agent environment or local filesystem.

---

### Prime Directive
**Do not stop until the goal is fully achieved.** 
**Code must be written, executed, verified, and committed.**

---

### Filesystem Rules
- Use **sandbox tools exclusively** for all file operations.
- Never touch files outside the sandbox.

---

### Git Workflow (Required)
All changes must go through Git. Never modify files without tracking them.

1. **Orient** — run `git status`, `git branch`, `git log --oneline -10` before starting.
2. **Branch** — create a feature branch: `git checkout -b <type>/<short-description>`.
3. **Commit** — atomic commits with Conventional Commit messages. Never commit broken code.
4. **Clean up** — confirm `git status` is clean and `git log` looks correct before finishing.

---

### Workflow
1. Check sandbox status. 
2. Understand the sandbox filesystem structure. 
3. Check installed packages, GPU, mounted data directories.
4. Explore the repo structure and conventions before writing anything.
5. Install missing packages; update the manifest (`requirements.txt`, `package.json`, etc.) and commit it with the code.
6. Write → run → verify → fix → repeat until the code works end-to-end.

---

### Definition of Done
Do not stop until **all** are true:
- [ ] Code runs end-to-end without errors.
- [ ] If data preprocessing is involved, the preprocessing completes successfully and outputs expected results. Do not use timeouts. Let it run until completion.
- [ ] If model training is involved, the model trains successfully and achieves expected performance. Do not use timeouts. Let it run until completion.
- [ ] If evaluation is involved, the evaluation completes successfully and achieves expected results. Do not use timeouts. Let it run until completion.
- [ ] Output matches the goal.
- [ ] All changes committed with clean history.
- [ ] `git status` is clean.
- [ ] New dependencies recorded in the manifest.
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