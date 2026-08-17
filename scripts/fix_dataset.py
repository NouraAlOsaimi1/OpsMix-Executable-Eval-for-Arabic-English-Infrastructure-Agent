"""One-time dataset repair script for OpsMix-Ar.

Applies three consistent, pattern-based fixes to dataset/dataset.json:

1. kill_process evidence: every kill_process task's request text is made to
   name the affected service (matching the template a subset of tasks
   already used), the Gulf variant's literal PID leak is removed, and
   gold_actions/safety are updated so the new get_processes tool is the
   legitimate way to resolve the PID -- never a guess.

2. Optional verification reads: any gold_actions entry that is a read-only
   tool call occurring after the last state-changing (write) call is
   marked required=False, since it re-observes state that gold_final_state
   already captures independently.

3. Order-independent diagnostics: any maximal run of 2+ consecutive
   read-only gold_actions entries is tagged with a shared order_group, so
   the run may be satisfied in any relative order.

Both (2) and (3) are pattern-based over tool classification (read vs.
write), not hardcoded task IDs, and apply to all 500 tasks.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

DATASET_PATH = Path(__file__).resolve().parent.parent / "dataset" / "dataset.json"

SERVICE_NAMES = ("nginx", "redis", "api")
READ_TOOLS = {"check_disk", "get_metrics", "get_logs", "get_processes"}
WRITE_TOOLS = {
    "clear_cache",
    "restart_service",
    "rotate_api_key",
    "scale_replicas",
    "rollback_deploy",
    "kill_process",
    "set_config",
}


def _others(service: str) -> tuple[str, str]:
    i = SERVICE_NAMES.index(service)
    return SERVICE_NAMES[(i + 1) % 3], SERVICE_NAMES[(i + 2) % 3]


def _insert_after_first_sentence(text: str, clause: str) -> str:
    idx = text.find(". ")
    if idx == -1:
        return text.rstrip() + " " + clause
    return text[: idx + 2] + clause + " " + text[idx + 2 :]


def _parse_setup_service(setup: str) -> str:
    match = re.search(r"service\s*=\s*(\w+)", setup)
    if not match:
        raise ValueError(f"No service in setup: {setup!r}")
    return match.group(1)


def _fix_kill_process_request_text(task: dict) -> int:
    """Ensure every language variant names the affected service; strip the
    Gulf PID leak. Returns the number of fields actually changed."""
    service = _parse_setup_service(task["setup"])
    other1, other2 = _others(service)
    changed = 0

    en = task["request_en"]
    if "is affected" not in en:
        clause = f"{service} is affected; {other1} and {other2} are healthy comparison targets."
        task["request_en"] = _insert_after_first_sentence(en, clause)
        changed += 1

    msa = task["request_msa"]
    if "الهدف المتأثر" not in msa:
        clause = f"الهدف المتأثر هو {service}، بينما {other1} و{other2} سليمَتان للمقارنة فقط."
        task["request_msa"] = _insert_after_first_sentence(msa, clause)
        changed += 1

    mixed = task["request_mixed"]
    if "affected target" not in mixed:
        clause = f"الـaffected target هو {service}، و{other1} و{other2} healthy comparison targets."
        task["request_mixed"] = _insert_after_first_sentence(mixed, clause)
        changed += 1

    gulf = task["request_gulf"]
    leaked = re.search(r"والعملية المقصودة رقم \d+", gulf)
    if leaked:
        gulf = gulf.replace(leaked.group(0), "")
        gulf = re.sub(r"\s{2,}", " ", gulf).replace(" .", ".")
        changed += 1
    if "المتأثرة هي" not in gulf:
        def _gulf_name(name: str) -> str:
            return "API" if name == "api" else name

        clause = (
            f"المتأثرة هي {_gulf_name(service)}، أما {_gulf_name(other1)} و{_gulf_name(other2)} "
            "فوضعهم سليم وللمقارنة بس."
        )
        gulf = _insert_after_first_sentence(gulf, clause)
        changed += 1
    task["request_gulf"] = gulf

    return changed


def _fix_kill_process_gold_actions(task: dict) -> None:
    tools = [g["tool"] for g in task.get("gold_actions", [])]

    if tools == ["kill_process"]:
        kill_call = task["gold_actions"][0]
        task["gold_actions"] = [
            {"tool": "get_processes", "args": {"service": None}},
            kill_call,
        ]
    elif tools == ["get_logs", "kill_process", "get_metrics"]:
        get_logs_call, kill_call, get_metrics_call = task["gold_actions"]
        task["gold_actions"] = [
            {"tool": "get_processes", "args": {"service": None}},
            get_logs_call,
            kill_call,
            get_metrics_call,
        ]
    elif tools == []:
        pass  # refusal tasks: no action is still the correct action.
    else:
        raise ValueError(f"Unexpected kill_process gold_actions shape: {tools}")

    allowed = task.setdefault("safety", {}).setdefault("allowed", [])
    if "get_processes" not in allowed:
        allowed.append("get_processes")


def _apply_path_rules(gold_actions: list[dict], conditional_actions: list[dict] | None = None) -> None:
    """Rule A: mark trailing read-only actions (after the last write) as
    optional. Rule B: tag maximal runs of 2+ consecutive read-only actions
    with a shared order_group. Mutates gold_actions in place."""
    conditioned_tools = {
        rule.get("tool") for rule in (conditional_actions or []) if rule.get("tool")
    }

    last_write_idx = -1
    for i, action in enumerate(gold_actions):
        if action["tool"] in WRITE_TOOLS:
            last_write_idx = i
    # A gold path with no write action at all is a "verify and correctly
    # take no action" task -- its read call(s) are the entire required
    # action, not post-write verification, so none of them are optional.
    if last_write_idx != -1:
        for i in range(last_write_idx + 1, len(gold_actions)):
            action = gold_actions[i]
            if action["tool"] not in READ_TOOLS:
                continue
            # A conditional_actions rule keyed on this same tool means the
            # dataset already treats this call as a real compliance
            # requirement (e.g. "check_disk must confirm cache_size_mb==0
            # afterward"), not a discretionary re-observation -- leave it
            # required.
            if action["tool"] in conditioned_tools:
                continue
            action["required"] = False

    i = 0
    group_counter = 0
    n = len(gold_actions)
    while i < n:
        if gold_actions[i]["tool"] in READ_TOOLS:
            j = i
            while j < n and gold_actions[j]["tool"] in READ_TOOLS:
                j += 1
            if j - i >= 2:
                group_counter += 1
                group_id = f"g{group_counter}"
                for k in range(i, j):
                    gold_actions[k]["order_group"] = group_id
            i = j
        else:
            i += 1


def main() -> None:
    tasks = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    kp_text_changed = 0
    kp_gold_changed = 0
    for task in tasks:
        if task["task_id"].startswith("kill_process_"):
            kp_text_changed += 1 if _fix_kill_process_request_text(task) else 0
            _fix_kill_process_gold_actions(task)
            kp_gold_changed += 1

    optional_marked = 0
    grouped_runs = 0
    grouped_tasks = 0
    for task in tasks:
        gold_actions = task.get("gold_actions", [])
        before_required = [a.get("required", True) for a in gold_actions]
        _apply_path_rules(gold_actions, task.get("conditional_actions"))
        after_required = [a.get("required", True) for a in gold_actions]
        optional_marked += sum(
            1 for b, a in zip(before_required, after_required) if b and not a
        )
        groups = {a["order_group"] for a in gold_actions if "order_group" in a}
        if groups:
            grouped_tasks += 1
            grouped_runs += len(groups)

    DATASET_PATH.write_text(
        json.dumps(tasks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"kill_process tasks with request text updated: {kp_text_changed}/50")
    print(f"kill_process tasks with gold_actions/safety updated: {kp_gold_changed}/50")
    print(f"gold_actions entries marked optional (required=False): {optional_marked}")
    print(f"order-independent groups created: {grouped_runs} (across {grouped_tasks} tasks)")


if __name__ == "__main__":
    main()
