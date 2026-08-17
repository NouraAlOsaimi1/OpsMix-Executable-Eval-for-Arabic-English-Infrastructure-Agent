"""Build a predictions file from each task's own gold_actions, for
end-to-end validation of checker.py/evaluate.py against the real sandbox."""
import json
from pathlib import Path

from app.tasks import get_all_tasks

out = {}
for task in get_all_tasks():
    out[task["task_id"]] = [
        {"tool": a["tool"], "args": a.get("args", {}) or {}}
        for a in task.get("gold_actions", [])
    ]

Path("scripts/gold_predictions.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(f"wrote {len(out)} tasks")
