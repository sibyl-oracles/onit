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

from .actuator import (apply as apply_lifecycle,
                       evaluate as evaluate_lifecycle,
                       filter_tools, format_report as format_lifecycle,
                       lifecycle_enabled, read_archive, restore,
                       withheld_tools, write_archive)
from .config import (OFF, OBSERVE, ADAPT, EXTEND, EVOLVE,
                     autonomy, capped, learn_root, level_name,
                     recording_enabled, redact_tool_args, requested_level,
                     trajectory_dir)
from .corpus import (format_summary as format_signal_summary,
                     label_corpus, labels_path, read_labels,
                     write_labels)
from .eval import (assert_outside_agent_surface, baseline_path,
                   build_split, dev_path, eval_dir, holdout_path,
                   is_outside_agent_surface, read_baseline, read_dev,
                   read_holdout, read_split, summarize as summarize_eval,
                   write_baseline, write_split)
from .events import (TOOL_EVENTS, iter_events, record_event,
                     record_tool_event, summarize_events, tool_timeline)
from .playbook import (ACTIVE, DEPRECATED, PROBATION, SCOPE_GLOBAL,
                       SCOPE_OWNER, apply_deltas, attribute, build_context,
                       bullet_id, cluster, contrast, curate, decay,
                       global_path, load, make_bullet, owner_path,
                       playbook_dir, read_playbook, reflect, render,
                       select, tool_signature, write_playbook)
from .report import format_status, summarize
from .skills import (manifest_errors, pinned_path, pinned_skills,
                     read_manifest, read_pin, skill_dir, skills_dir,
                     summarize as summarize_skills, write_manifest,
                     write_pin)
from .skills_curator import (gate_skill, promote, rollback as rollback_skills,
                              synthesize, write_skill)
from .skills_gate import (ALL_CHECKS, CHECK_DEPS, CHECK_MANIFEST,
                          CHECK_REGRESSION, CHECK_SOURCE, CHECK_TESTS,
                          GateResult, run_gate)
from .skills_mining import (MIN_SESSIONS, format_candidates, mine,
                            mine_store, suggest_name)
from .signals import (is_acceptance, is_rephrase, label_pair, label_records,
                      label_session, similarity,
                      summarize as summarize_signals)
from .trajectory import (SCHEMA_VERSION, append_rating, args_digest,
                         build_record, describe_tool_call, derive_signals,
                         iter_records, normalize_rating, owner_hash,
                         read_session, record_task, session_file)

__all__ = [
    "ACTIVE", "DEPRECATED", "OFF", "OBSERVE", "ADAPT", "EXTEND", "EVOLVE",
    "PROBATION", "SCOPE_GLOBAL", "SCOPE_OWNER",
    "SCHEMA_VERSION", "TOOL_EVENTS",
    "apply_deltas", "apply_lifecycle", "attribute", "build_context",
    "bullet_id", "capped", "cluster",
    "append_rating", "args_digest", "assert_outside_agent_surface",
    "autonomy", "baseline_path", "build_record", "build_split",
    "describe_tool_call", "derive_signals", "dev_path", "eval_dir",
    "evaluate_lifecycle",
    "filter_tools", "format_candidates", "format_lifecycle",
    "format_signal_summary",
    "format_status", "gate_skill", "holdout_path",
    "is_acceptance", "is_outside_agent_surface", "is_rephrase",
    "lifecycle_enabled",
    "iter_events", "iter_records", "label_corpus", "label_pair",
    "label_records", "label_session", "labels_path", "learn_root",
    "level_name",
    "manifest_errors", "mine", "mine_store", "normalize_rating",
    "owner_hash", "promote", "read_baseline", "read_dev",
    "read_archive", "read_holdout", "read_labels", "read_manifest",
    "read_pin", "read_session",
    "read_split", "requested_level", "restore", "rollback_skills",
    "run_gate",
    "record_event", "record_task", "record_tool_event", "recording_enabled",
    "redact_tool_args", "session_file", "similarity", "skill_dir",
    "skills_dir", "suggest_name", "summarize",
    "summarize_eval", "summarize_events", "summarize_signals",
    "summarize_skills", "synthesize", "tool_timeline", "trajectory_dir",
    "withheld_tools", "write_archive", "write_baseline", "write_labels",
    "write_manifest", "write_pin", "write_playbook", "write_skill",
    "write_split",
]
