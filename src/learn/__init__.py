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

The experience substrate: what OnIt keeps about its own runs.

This package is the ground floor of the plan in ``docs/SELF_IMPROVEMENT.md``.
It records and nothing else — no retrieval, no playbook, no self-modification.
That ordering is deliberate: an agent cannot improve on experience it never
wrote down, and the recording is worth having on its own for answering which
tools fail and on what.

Default autonomy is ``observe``: trajectories are written, nothing is injected,
and every request is byte-for-byte what it would have been without this package.
"""

from .config import (OFF, OBSERVE, ADAPT, EXTEND, EVOLVE,
                     autonomy, level_name, recording_enabled,
                     redact_tool_args, trajectory_dir)
from .report import format_status, summarize
from .trajectory import (SCHEMA_VERSION, append_rating, args_digest,
                         build_record, describe_tool_call, derive_signals,
                         iter_records, normalize_rating, owner_hash,
                         read_session, record_task, session_file)

__all__ = [
    "OFF", "OBSERVE", "ADAPT", "EXTEND", "EVOLVE",
    "SCHEMA_VERSION",
    "append_rating", "args_digest", "autonomy", "build_record",
    "describe_tool_call", "derive_signals", "format_status", "iter_records",
    "level_name", "normalize_rating", "owner_hash", "read_session",
    "record_task", "recording_enabled", "redact_tool_args", "session_file",
    "summarize", "trajectory_dir",
]
