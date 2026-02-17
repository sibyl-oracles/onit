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
import yaml
import uuid
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


mcp_prompts = FastMCP("Prompts MCP")

@mcp_prompts.prompt("assistant")
async def assistant_instruction(task: str,
                                session_id: str = None,
                                template_path: str = None,
                                file_server_url: str = None) -> str:
   import tempfile

   if session_id is None:
      session_id = str(uuid.uuid4())

   data_path = str(Path(tempfile.gettempdir()) / "onit" / "data" / session_id)
   Path(data_path).mkdir(parents=True, exist_ok=True)

   default_template = """
Think step by step on how to complete the following task enclosed in <task> and </task> tags.

<task>
{task}
</task>

Execute the step by step action plan to complete the task.
If you need additional information or the task is unclear given the context and previous interactions, ask for clarification.
If you know the answer, provide it right away. Else, use the tools to complete the action plan.
Avoid repeated tool call sequences that do not lead to progress. 

## File Operations Policy
- **Working directory**: `{data_path}` — all file operations must use this directory.
- **Session ID**: `{session_id}` — files created in this session are owned by this session only.
- **NEVER** create files in the user home directory or any location outside `{data_path}`.
- All temporary and output files must be saved within `{data_path}`.
- Files are created with restricted permissions — only the session owner can access them.
- Other sessions cannot read or write files belonging to this session.
"""

   template = default_template

   if template_path:

      template_file = Path(template_path)
      if template_file.exists() and template_file.suffix in ('.yaml', '.yml'):
         with open(template_file, 'r') as f:
               config = yaml.safe_load(f)
               template = config.get('instruction_template', default_template)

   instruction = template.format(
      task=task,
      data_path=data_path,
      session_id=session_id
   )

   if file_server_url:
      instruction += f"""
Files are served by a remote file server at {file_server_url}/uploads/.
Before reading any file referenced in the task, first download it:
  curl -s {file_server_url}/uploads/<filename> -o {data_path}/<filename>
After creating or saving any output file, upload it back to the file server:
  curl -s -X POST -F 'file=@{data_path}/<filename>' {file_server_url}/uploads/
Always download before reading and upload after writing.
When using create_presentation, create_excel, or create_document tools, always pass callback_url="{file_server_url}" so files are automatically uploaded.
"""

   return instruction


def run(
    transport: str = "streamable-http",
    host: str = "0.0.0.0",
    port: int = 9002,
    path: str = "/prompts",
    options: dict = {}
) -> None:
    """Run the Prompts MCP server."""
    if 'verbose' in options:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f"Starting Prompts MCP Server at {host}:{port}{path}")
    mcp_prompts.run(transport=transport, host=host, port=port, path=path)