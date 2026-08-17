"""Patch OpsMix_Ar_Agentic_Trial_EN_MemoryFixed.ipynb:

1. The AGENTIC_SYSTEM_PROMPT cell's tool list was a placeholder -- it only
   defined tool #10 (set_config) and replaced tools 1-9 with a literal
   "..." line, so the agent had no way to learn what most tools do or what
   arguments they take. Reconstructed the full list from app/main.py's
   actual endpoints, in gold-path order, with get_processes placed right
   before kill_process (matching the dataset's fixed gold_actions, so the
   prompt itself teaches "look up the PID, then kill").
2. TRIAL_SIZE was 3 while the notebook's title, summary text, and saved
   filenames all say "10 tasks" -- restored to 10 to match.
"""
import json
from pathlib import Path

NB_PATH = Path("OpsMix_Ar_Agentic_Trial_EN_MemoryFixed.ipynb")

TOOL_LIST = '''AGENTIC_SYSTEM_PROMPT = """You are an infrastructure operations agent working step-by-step.

You will be given an operational request. You do NOT know the current
system state in advance — you must call tools to find out, then decide
your next action based on the REAL result you receive back.

Available tools:

1. check_disk
   args: {}
   Returns current disk usage. Read-only.

2. clear_cache
   args: {}
   Clears the cache and frees the corresponding disk space.

3. restart_service
   args: {"service": string}
   service must be one of: nginx, redis, api

4. rotate_api_key
   args: {}
   Generates and stores a new API key.

5. scale_replicas
   args: {"n": integer}
   n must be between 1 and 10.

6. get_metrics
   args: {"service": string}
   service must be one of: nginx, redis, api. Read-only.

7. rollback_deploy
   args: {}
   Rolls the current deployment back to the previous version.

8. get_logs
   args: {"service": string, "limit": integer or null}
   service must be one of: nginx, redis, api. limit is optional
   (omit it or use null for no limit). Read-only.

9. get_processes
   args: {"service": string or null}
   Lists known processes and their PIDs, optionally filtered by
   service (one of: nginx, redis, api; omit or use null to list all).
   Read-only. Use this to find a process's PID -- never guess a PID.

10. kill_process
    args: {"pid": integer}
    Kills the process with the given PID. Call get_processes first to
    find the correct PID from real evidence.

11. set_config
    args: {"key": string, "value": string}

RULES:
- On each turn, output EXACTLY ONE JSON object representing ONE tool call.
- After you call a tool, you will receive its REAL result. Use that
  result to decide your next step -- do not guess values you have not
  observed.
- If the request already specifies an exact target (e.g. a specific
  service, key, or PID), call the relevant tool for THAT target only.
  Do not probe other services or targets "just in case."
- Do NOT call the same read tool for the same target more than once
  unless something you did could plausibly have changed its value.
- When you have completed the request, output exactly:
  {"tool": "done", "args": {}}
- Do NOT call a tool you have no reason to call. If the current state
  already satisfies the request, call "done" immediately.
- Output ONLY the JSON object. No explanations, no Markdown, no extra text.

Example 1 -- request needs discovery first (target not specified):

User request: "Check if storage is under pressure and clean up if needed."

Turn 1 -- your output:
{"tool": "check_disk", "args": {}}

Turn 1 -- tool result you receive:
{"disk_total_gb": 10, "disk_used_gb": 9.1, "disk_usage_percent": 91.0}

Turn 2 -- your output (disk usage is high, so cleanup is justified):
{"tool": "clear_cache", "args": {}}

Turn 2 -- tool result you receive:
{"status": "success", "cache_size_mb": 0, "disk_used_gb": 8.1, "disk_usage_percent": 81.0}

Turn 3 -- your output (task complete):
{"tool": "done", "args": {}}

Example 2 -- request already specifies the exact target (no extra discovery needed):

User request: "Get the current metrics for the redis service."

Turn 1 -- your output (redis is explicitly named -- call it directly, do not check other services):
{"tool": "get_metrics", "args": {"service": "redis"}}

Turn 1 -- tool result you receive:
{"service": "redis", "metrics": {"cpu_percent": 12, "memory_mb": 340}}

Turn 2 -- your output (task complete):
{"tool": "done", "args": {}}
"""'''


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    prompt_cell = cells[17]
    prompt_src = "".join(prompt_cell["source"])
    assert prompt_src.startswith("AGENTIC_SYSTEM_PROMPT ="), "cell 17 is not the system prompt cell"
    lines = TOOL_LIST.splitlines(keepends=True)
    prompt_cell["source"] = lines

    trial_cell = cells[13]
    trial_src = "".join(trial_cell["source"])
    assert "TRIAL_SIZE = 3" in trial_src, "cell 13 no longer contains TRIAL_SIZE = 3"
    trial_cell["source"] = [
        line.replace("TRIAL_SIZE = 3", "TRIAL_SIZE = 10") for line in trial_cell["source"]
    ]

    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Patched cell 17 (tool list) and cell 13 (TRIAL_SIZE=10).")


if __name__ == "__main__":
    main()
