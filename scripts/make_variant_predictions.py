"""Build a second predictions file that perturbs every task's gold_actions
in a way that should still be graded as a pass:

* drop any action marked required=False (optional verification)
* reverse the relative order of any order_group's members

Tasks with neither annotation are replayed exactly (no perturbation
possible/needed). Used to stress-test the checker.py fixes across the full
dataset, not just hand-picked examples.
"""
import json
from pathlib import Path

from app.tasks import get_all_tasks

out = {}
affected = []

for task in get_all_tasks():
    gold = task.get("gold_actions", [])
    calls = [{"tool": a["tool"], "args": a.get("args", {}) or {}} for a in gold]
    flags = [a.get("required", True) for a in gold]
    groups = [a.get("order_group") for a in gold]

    perturbed = False

    # Reverse each order_group's members in place.
    i = 0
    while i < len(gold):
        if groups[i] is not None:
            j = i
            while j < len(gold) and groups[j] == groups[i]:
                j += 1
            if j - i >= 2:
                calls[i:j] = list(reversed(calls[i:j]))
                perturbed = True
            i = j
        else:
            i += 1

    # Drop optional actions.
    kept = [c for c, required in zip(calls, flags) if required]
    if len(kept) != len(calls):
        perturbed = True
    calls = kept

    out[task["task_id"]] = calls
    if perturbed:
        affected.append(task["task_id"])

Path("scripts/variant_predictions_full.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
)
Path("scripts/affected_task_ids.json").write_text(
    json.dumps(affected, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(f"wrote {len(out)} tasks, {len(affected)} perturbed")
