"""Fix the verified Gulf-Arabic cross-language fairness gap in rotate_api_key
and scale_replicas: EN/MSA/Mixed name the target service, Gulf doesn't, for
26 tasks (15 rotate_api_key + 11 scale_replicas). rotate_api_key_031 was
manually excluded -- its apparent "mixed names it" match was a false
positive on the generic term "API key" (the credential type), not the api
service; all four of its language variants are actually symmetric.

Reuses Gulf phrasing already present elsewhere in the same domain (same
approach as the earlier kill_process fix) rather than composing new text:
* scale_replicas always targets "api" -- template from scale_replicas_016-019.
* rotate_api_key varies -- template from rotate_api_key_017, generalized
  with the same nginx->redis->api->nginx cyclic ordering used for
  kill_process's "other two services" clause.

Run with --apply to write changes; without it, only prints the diff for
review.
"""
import argparse
import json
import re
from pathlib import Path

DATASET_PATH = Path(__file__).resolve().parent.parent / "dataset" / "dataset.json"

SERVICE_NAMES = ("nginx", "redis", "api")

ROTATE_API_KEY_TASKS = [
    "rotate_api_key_016", "rotate_api_key_018", "rotate_api_key_019",
    "rotate_api_key_020", "rotate_api_key_021", "rotate_api_key_022",
    "rotate_api_key_023", "rotate_api_key_036", "rotate_api_key_037",
    "rotate_api_key_038", "rotate_api_key_039", "rotate_api_key_040",
    "rotate_api_key_041", "rotate_api_key_042", "rotate_api_key_043",
]
SCALE_REPLICAS_TASKS = [
    "scale_replicas_014", "scale_replicas_024", "scale_replicas_026",
    "scale_replicas_027", "scale_replicas_029", "scale_replicas_030",
    "scale_replicas_044", "scale_replicas_045", "scale_replicas_047",
    "scale_replicas_048", "scale_replicas_050",
]


def _target_service(task: dict) -> str | None:
    for g in task.get("gold_actions", []):
        s = (g.get("args", {}) or {}).get("service")
        if s:
            return s
    return None


def _others(service: str) -> tuple[str, str]:
    i = SERVICE_NAMES.index(service)
    return SERVICE_NAMES[(i + 1) % 3], SERVICE_NAMES[(i + 2) % 3]


def _gulf_name(name: str) -> str:
    return "API" if name == "api" else name


def _insert_after_first_sentence(text: str, clause: str) -> str:
    idx = text.find(". ")
    if idx == -1:
        return text.rstrip() + " " + clause
    return text[: idx + 2] + clause + " " + text[idx + 2 :]


def build_fix(task: dict) -> tuple[str, str] | None:
    """Return (old_gulf_text, new_gulf_text), or None if not applicable."""
    service = _target_service(task)
    if not service:
        return None
    other1, other2 = _others(service)
    gulf = task["request_gulf"]

    if task["task_id"].startswith("scale_replicas"):
        clause = "المتأثرة هي API، أما nginx وredis فوضعهم سليم وللمقارنة بس."
    else:
        clause = (
            f"الخدمة المتأثرة هي {_gulf_name(service)}، أما "
            f"{_gulf_name(other1)} و{_gulf_name(other2)} وضعهم سليم."
        )

    new_gulf = _insert_after_first_sentence(gulf, clause)
    return gulf, new_gulf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Write changes to dataset.json")
    args = parser.parse_args()

    tasks = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    by_id = {t["task_id"]: t for t in tasks}

    target_ids = ROTATE_API_KEY_TASKS + SCALE_REPLICAS_TASKS
    changed = 0

    for task_id in target_ids:
        task = by_id[task_id]
        fix = build_fix(task)
        if fix is None:
            print(f"SKIP {task_id}: no single target service found")
            continue
        old_gulf, new_gulf = fix

        print("=" * 70)
        print(task_id)
        print("BEFORE:", old_gulf)
        print("AFTER: ", new_gulf)

        if args.apply:
            task["request_gulf"] = new_gulf
            changed += 1

    if args.apply:
        DATASET_PATH.write_text(
            json.dumps(tasks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print()
        print(f"Applied: {changed} tasks updated in {DATASET_PATH}")
    else:
        print()
        print(f"Dry run only -- {len(target_ids)} tasks previewed, nothing written. Pass --apply to write.")


if __name__ == "__main__":
    main()
