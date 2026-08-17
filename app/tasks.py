"""Task loader and schema adapter for OpsMix-Ar.

Supports the raw dataset schema and exposes every task through the
canonical schema expected by reset.py and checker.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from copy import deepcopy

from app.state import _initial_state as _default_sandbox_state


DATASET_PATH = Path(__file__).resolve().parent.parent / "dataset" / "dataset.json"


REQUIRED_REQUEST_LANGS = {"en", "msa", "gulf", "mixed"}
REQUIRED_SAFETY_TIERS = {"allowed", "risky", "forbidden"}


def _parse_setup(setup: str | dict | None) -> dict:
    """Convert the dataset's setup field into state overrides."""

    if setup is None:
        return {}

    if isinstance(setup, dict):
        return deepcopy(setup)

    if not isinstance(setup, str):
        raise ValueError(f"Unsupported setup type: {type(setup)}")

    result = {}

    for item in setup.split(";"):
        item = item.strip()

        if not item or "=" not in item:
            continue

        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()

        # Convert common primitive values.
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        elif value.lower() in ("null", "none"):
          value = None
        else:
            try:
                if "." in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass

        result[key] = value

    return result


def _build_initial_state(task: dict) -> dict:
    """Convert raw setup fields into the sandbox state structure."""

    setup = _parse_setup(task.get("setup"))

    # If the task already contains canonical initial_state,
    # preserve it.
    if isinstance(task.get("initial_state"), dict):
        return deepcopy(task["initial_state"])

    state = {}

    # ---------------------------------------------------------
    # Storage / disk tasks
    # ---------------------------------------------------------

    if "disk_used_gb" in setup:
        state["disk_used_gb"] = setup["disk_used_gb"]

    if "disk_total_gb" in setup:
        state["disk_total_gb"] = setup["disk_total_gb"]

    if "cache_size_mb" in setup:
        state["cache_size_mb"] = setup["cache_size_mb"]

    # ---------------------------------------------------------
    # Service / availability tasks
    # ---------------------------------------------------------

    service = setup.get("service")

    if service:
        service = str(service).strip().lower()

        service_state = {}

        if "status" in setup:
            service_state["status"] = setup["status"]

        if "restart_count" in setup:
            service_state["restart_count"] = setup["restart_count"]

        if service_state:
            state.setdefault("services", {})
            state["services"][service] = service_state

        if "cpu_percent" in setup:
            state.setdefault("metrics", {})
            state["metrics"].setdefault(service, {})
            state["metrics"][service]["cpu_percent"] = setup["cpu_percent"]

        if "memory_mb" in setup:
            state.setdefault("metrics", {})
            state["metrics"].setdefault(service, {})
            state["metrics"][service]["memory_mb"] = setup["memory_mb"]

    # ---------------------------------------------------------
    # Process state
    # ---------------------------------------------------------

    if "pid" in setup and service:
        pid = int(setup["pid"])

        # If this task assigns `service` to a pid other than its default
        # one, the default pid that used to hold that service must be
        # freed explicitly: reset.py deep-merges this override onto the
        # full default state, so simply omitting that pid from the
        # override would leave its stale duplicate entry in place --
        # get_processes(service=...) would then return two ambiguous
        # matches for the same service.
        default_processes = _default_sandbox_state()["processes"]
        freed_pid = next(
            (
                existing_pid
                for existing_pid, proc in default_processes.items()
                if proc["service"] == service and existing_pid != pid
            ),
            None,
        )

        state.setdefault("processes", {})

        if freed_pid is not None:
            state["processes"][freed_pid] = {
                "pid": freed_pid,
                "service": None,
                "status": "not_running",
            }

        state["processes"][pid] = {
            "pid": pid,
            "service": service,
            "status": "running",
        }

    # ---------------------------------------------------------
    # Deployment state
    # ---------------------------------------------------------

    if "current_version" in setup or "previous_version" in setup:
        state.setdefault("deployment", {})

        if "current_version" in setup:
            state["deployment"]["current_version"] = setup["current_version"]

        if "previous_version" in setup:
            state["deployment"]["previous_version"] = setup["previous_version"]

    # ---------------------------------------------------------
    # Logs
    # Supports setup fields such as:
    # logs.redis=...
    # logs.nginx=...
    # logs.api=...
    # ---------------------------------------------------------

    for service_name in ("nginx", "redis", "api"):
        setup_key = f"logs.{service_name}"

        if setup_key in setup:
            state.setdefault("logs", {})
            state["logs"][service_name] = setup[setup_key]

    # ---------------------------------------------------------
    # Config
    # Supports setup fields such as:
    # config.log_level=debug
    # config.traffic_profile=high
    # config.maintenance_mode=true
    # ---------------------------------------------------------

    for setup_key, value in setup.items():
        if setup_key.startswith("config."):
            config_key = setup_key.split(".", 1)[1]

            state.setdefault("config", {})
            state["config"][config_key] = value

    # ---------------------------------------------------------
    # Target replicas
    # ---------------------------------------------------------

    if "target_replicas" in setup:
        state["target_replicas"] = setup["target_replicas"]

    # ---------------------------------------------------------
    # Generic known state fields
    # ---------------------------------------------------------

    known_top_level = {
        "api_key",
        "api_key_last_rotated",
        "replicas",
    }

    for key in known_top_level:
        if key in setup:
            state[key] = setup[key]

    return state


def _build_request(task: dict) -> dict:
    """Build the four-language request dictionary."""

    # Already canonical.
    if isinstance(task.get("request"), dict):
        return {
            "en": task["request"].get("en", ""),
            "msa": task["request"].get("msa", ""),
            "gulf": task["request"].get("gulf", ""),
            "mixed": task["request"].get("mixed", ""),
        }

    return {
        "en": task.get("request_en", ""),
        "msa": task.get("request_msa", ""),
        "gulf": task.get("request_gulf", ""),
        "mixed": task.get("request_mixed", ""),
    }


def _build_safety(task: dict) -> dict:
    """Normalize safety information."""

    safety = task.get("safety")

    if isinstance(safety, dict):
        return {
            "allowed": safety.get("allowed", []),
            "risky": safety.get("risky", []),
            "forbidden": safety.get("forbidden", []),
        }

    # Fallback for datasets that use a single safety field.
    safety_level = str(task.get("safety", "allowed")).lower()

    if safety_level == "forbidden":
        return {
            "allowed": [],
            "risky": [],
            "forbidden": [],
        }

    if safety_level == "risky":
        return {
            "allowed": [],
            "risky": [],
            "forbidden": [],
        }

    return {
        "allowed": [],
        "risky": [],
        "forbidden": [],
    }


def _normalize_task(task: dict, index: int) -> dict:
    """Convert one raw dataset row to the canonical task schema."""

    task_id = task.get(
        "task_id",
        f"<index {index}, no task_id>"
    )

    normalized = dict(task)

    normalized["task_id"] = task_id

    normalized["domain"] = task.get("domain", "")
    normalized["incident"] = task.get("incident", "")

    normalized["request"] = _build_request(task)

    normalized["initial_state"] = _build_initial_state(task)

    normalized["gold_actions"] = task.get(
        "gold_actions",
        []
    )

    normalized["gold_final_state"] = task.get(
        "gold_final_state",
        {}
    )

    normalized["conditional_actions"] = task.get(
        "conditional_actions",
        []
    )

    normalized["safety"] = _build_safety(task)

    normalized["difficulty"] = task.get(
        "difficulty",
        "medium"
    )

    return normalized


def _validate_task(task: dict, index: int) -> None:
    """Validate the normalized canonical task."""

    label = task.get(
        "task_id",
        f"<index {index}, no task_id>"
    )

    required = {
        "task_id",
        "domain",
        "incident",
        "request",
        "initial_state",
        "gold_actions",
        "gold_final_state",
        "safety",
        "difficulty",
    }

    missing = required - task.keys()

    if missing:
        raise ValueError(
            f"Task '{label}' is missing fields: {missing}"
        )

    if not isinstance(task["request"], dict):
        raise ValueError(
            f"Task '{label}' request must be a dictionary."
        )

    missing_langs = (
        REQUIRED_REQUEST_LANGS
        - task["request"].keys()
    )

    if missing_langs:
        raise ValueError(
            f"Task '{label}' request is missing languages: "
            f"{missing_langs}"
        )

    if not isinstance(task["safety"], dict):
        raise ValueError(
            f"Task '{label}' safety must be a dictionary."
        )

    missing_tiers = (
        REQUIRED_SAFETY_TIERS
        - task["safety"].keys()
    )

    if missing_tiers:
        raise ValueError(
            f"Task '{label}' safety is missing tiers: "
            f"{missing_tiers}"
        )

    if "conditional_actions" in task:
        for entry in task["conditional_actions"]:

            if (
                "tool" not in entry
                or "condition" not in entry
            ):
                raise ValueError(
                    f"Task '{label}' has malformed "
                    f"conditional_actions entry: {entry}"
                )

            condition = entry["condition"]

            required_condition = {
                "field",
                "operator",
                "value",
            }

            if not required_condition <= condition.keys():
                raise ValueError(
                    f"Task '{label}' conditional_actions "
                    f"condition is missing fields: "
                    f"{required_condition - condition.keys()}"
                )


def _load_all_tasks() -> dict:
    """Load and normalize the complete dataset."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"dataset.json not found at {DATASET_PATH}"
        )

    with open(
        DATASET_PATH,
        encoding="utf-8"
    ) as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(
            "dataset.json must contain a JSON array."
        )

    tasks = {}

    for index, raw_task in enumerate(raw):

        if not isinstance(raw_task, dict):
            raise ValueError(
                f"Task at index {index} must be a JSON object."
            )

        task = _normalize_task(
            raw_task,
            index
        )

        _validate_task(
            task,
            index
        )

        task_id = task["task_id"]

        if task_id in tasks:
            raise ValueError(
                f"Duplicate task_id in dataset.json: "
                f"'{task_id}'"
            )

        tasks[task_id] = task

    return tasks


TASKS_BY_ID = _load_all_tasks()


def get_task(task_id: str) -> dict:
    """Return one normalized task by ID."""

    if task_id not in TASKS_BY_ID:
        raise KeyError(
            f"Unknown task_id '{task_id}'. "
            f"Known tasks: {list(TASKS_BY_ID)}"
        )

    return TASKS_BY_ID[task_id]


def get_task_count() -> int:
    """Return the number of loaded tasks."""

    return len(TASKS_BY_ID)


def get_all_tasks() -> list[dict]:
    """Return all normalized tasks."""

    return list(TASKS_BY_ID.values())