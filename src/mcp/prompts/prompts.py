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
                                topic: str = None) -> str:
   if not data_path:
      raise ValueError("data_path is required and must be a non-empty string")

   Path(data_path).mkdir(parents=True, exist_ok=True)
   current_date = datetime.now().strftime("%B %d, %Y")

   default_template = """
You are an autonomous agent with access to tools and a file system.

## Context
- **Today's date**: {current_date}
- **Working directory**: {data_path} — sandbox folder for reading and writing files. 

## Constraints
- NEVER create or modify files outside of `{data_path}`.

## Task
{task}
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
      current_date=current_date,
      data_path=data_path
   )

   if topic and topic != "null":
      instruction += f"""
## Topic
Unless specified, assume that the topic is about `{topic}`.
"""

   if file_server_url and file_server_url != "null":
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

   if documents_path and documents_path != "null":
      instruction += f"""
## Relevant Information
Search and read related documents (PDF, TXT, DOCX, XLSX, PPTX, and Markdown (MD)) in `{documents_path}`.
Search the web for additional information **if and only if** above documents are insufficient to complete the task.
"""

   instruction += f"""
## Instructions
1. If the answer is straightforward, respond directly without tool use.
2. Otherwise, reason step by step, invoke tools as needed, and work toward a final answer.
3. If critical information is missing and cannot be inferred, ask exactly one clarifying question before proceeding.
4. If a file was generated, provide a download link to the file.
5. Conclude with your final answer in this format:

<your answer here>
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