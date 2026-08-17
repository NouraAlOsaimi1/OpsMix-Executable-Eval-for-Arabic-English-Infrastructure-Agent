#main.py 

"""FastAPI app exposing the tiny infra service's tools as HTTP endpoints.

Each tool from the project spec becomes exactly one endpoint here.
This step only adds check_disk() -- clear_cache() and
restart_service() come in later steps.
"""

from __future__ import annotations
import copy 

import secrets
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from app.state import state, SERVICE_NAMES
from app.reset import reset

# Constants for replica scaling constraints
MIN_REPLICAS = 1
MAX_REPLICAS = 10

# resource: https://fastapi.tiangolo.com/tutorial/body/
app = FastAPI(title="Tiny Infra Service")


def _record(
    tool: str,
    args: dict,
    timestamp: str | None = None,
    state_before: dict | None = None
) -> None:

    state["history"].append(
        {
            "tool": tool,
            "args": args,
            "timestamp": timestamp
            or datetime.now(timezone.utc).isoformat(),
            "state_before": state_before,
        }
    )


# The first tool: check_disk() Return current disk usage. Read-only
@app.get("/check_disk")
def check_disk():
    state_before = copy.deepcopy(state)
    disk_usage_percent = (state["disk_used_gb"] / state["disk_total_gb"]) * 100
    _record("check_disk", {}, state_before=state_before)

    return {
        "disk_total_gb": state["disk_total_gb"],
        "disk_used_gb": state["disk_used_gb"],
        "disk_usage_percent": round(disk_usage_percent, 2),
    }


# The second tool: clear_cache() is a write operation Clear the cache: empty it and reduce disk usage accordingly.
# resource: https://onlinetoolsforge.com/en/tools/disk-usage-calculator/
@app.post("/clear_cache")
def clear_cache():
    state_before = copy.deepcopy(state)
    freed_gb = state["cache_size_mb"] / 1024  # convert MB freed -> GB freed

    state["disk_used_gb"] = max(0, state["disk_used_gb"] - freed_gb)
    state["cache_size_mb"] = 0
    _record("clear_cache", {}, state_before=state_before)

    disk_usage_percent = (state["disk_used_gb"] / state["disk_total_gb"]) * 100

    return {
        "status": "success",
        "cache_size_mb": state["cache_size_mb"],
        "disk_used_gb": round(state["disk_used_gb"], 2),
        "disk_usage_percent": round(disk_usage_percent, 2),
    }


# The third tool: restart_service() Restart one of nginx, redis, or api. Rejects any other value.
@app.post("/restart_service")
def restart_service(service: str):
    service = service.strip().lower()
    if service not in SERVICE_NAMES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Service '{service}' is not supported. Supported services:"
                f" {', '.join(SERVICE_NAMES)}."
            ),
        )

    state_before = copy.deepcopy(state)

    now = datetime.now(timezone.utc).isoformat()

    state["services"][service]["status"] = "running"
    state["services"][service]["last_restart"] = now
    state["services"][service]["restart_count"] += 1
    
    _record("restart_service", {"service": service}, timestamp=now, state_before=state_before)

    return {
        "status": "success",
        "service": service,
        "last_restart": now,
    }


# The fourth tool: rotate_api_key() - Generate a new API key and replace the existing one.
@app.post("/rotate_api_key")
def rotate_api_key():
    state_before = copy.deepcopy(state)
    new_key = secrets.token_hex(32)
    rotated_at = datetime.now(timezone.utc).isoformat()

    state["api_key"] = new_key
    state["api_key_last_rotated"] = rotated_at

    _record("rotate_api_key", {}, state_before=state_before)

    # Do NOT return the actual API key
    return {
        "status": "success",
        "api_key_rotated": True,
        "api_key_last_rotated": rotated_at,
    }


# The fifth tool: scale_replicas(n)
# Scale the number of running replicas.
@app.post("/scale_replicas")
def scale_replicas(n: int):
    if n < MIN_REPLICAS or n > MAX_REPLICAS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Replica count must be between {MIN_REPLICAS} and "
                f"{MAX_REPLICAS}."
            ),
        )

    state_before = copy.deepcopy(state)
    current_replicas = state["replicas"]

    if current_replicas == n:
        _record(
            "scale_replicas",
            {"n": n},
            state_before=state_before,
        )

        return {
            "status": "no_op",
            "message": f"Replica count is already {n}.",
            "replicas": current_replicas,
        }

    previous_replicas = current_replicas
    state["replicas"] = n

    _record(
        "scale_replicas",
        {"n": n},
        state_before=state_before,
    )

    return {
        "status": "success",
        "previous_replicas": previous_replicas,
        "replicas": state["replicas"],
    }


@app.get("/get_metrics")
def get_metrics(service: str):
    service = service.strip().lower()

    if service not in SERVICE_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Service '{service}' is not supported.",
        )

    state_before = copy.deepcopy(state)
    _record("get_metrics", {"service": service}, state_before=state_before)

    return {"service": service, "metrics": state["metrics"][service]}


@app.post("/rollback_deploy")
def rollback_deploy():
    state_before = copy.deepcopy(state)

    current = state["deployment"]["current_version"]
    previous = state["deployment"]["previous_version"]

    state["deployment"]["current_version"] = previous
    state["deployment"]["previous_version"] = current

    _record("rollback_deploy", {}, state_before=state_before)

    return {"status": "success", "rolled_back_to": previous}


# get_logs(service) -- Return recent log lines for a service. Read-only.
@app.get("/get_logs")
def get_logs(service: str, limit: int | None = None):
    service = service.strip().lower()

    if service not in SERVICE_NAMES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Service '{service}' is not supported. Supported services:"
                f" {', '.join(SERVICE_NAMES)}."
            ),
        )

    logs = state["logs"][service]

    if limit is not None:
        if limit < 1:
            raise HTTPException(
                status_code=400, detail="limit must be a positive integer."
            )
        logs = logs[-limit:]
    state_before = copy.deepcopy(state)  # read-only; snapshot for schema consistency
    _record("get_logs", {"service": service, "limit": limit}, state_before=state_before)

    return {
        "service": service,
        "logs": logs,
    }


# get_processes(service) -- List known processes and their PIDs, optionally
# filtered by service. Read-only; the only tool that exposes pid<->service
# evidence, so kill_process's target PID never has to be guessed.
@app.get("/get_processes")
def get_processes(service: str | None = None):
    if service is not None:
        service = service.strip().lower()
        if service not in SERVICE_NAMES:
            raise HTTPException(
                status_code=400,
                detail=f"Service '{service}' is not supported.",
            )

    state_before = copy.deepcopy(state)

    processes = sorted(state["processes"].values(), key=lambda p: p["pid"])
    if service is not None:
        processes = [p for p in processes if p.get("service") == service]

    _record("get_processes", {"service": service}, state_before=state_before)

    return {"processes": processes}


# kill_process(pid) -- Kill a running process by PID.
@app.post("/kill_process")
def kill_process(pid: int):
    process = state["processes"].get(pid)

    if process is None:
        raise HTTPException(
            status_code=404,
            detail=f"No process found with pid {pid}.",
        )

    state_before = copy.deepcopy(state)

    if process["status"] == "killed":
        _record(
            "kill_process",
            {"pid": pid},
            state_before=state_before,
        )

        return {
            "status": "no_op",
            "message": f"Process {pid} is already killed.",
            "pid": pid,
        }

    previous_status = process["status"]
    now = datetime.now(timezone.utc).isoformat()

    process["status"] = "killed"
    process["killed_at"] = now

    _record(
        "kill_process",
        {"pid": pid},
        timestamp=now,
        state_before=state_before,
    )

    return {
        "status": "success",
        "pid": pid,
        "service": process["service"],
        "previous_status": previous_status,
        "killed_at": now,
    }

# set_config(key, value) -- Set (or overwrite) a config key.
@app.post("/set_config")
def set_config(key: str, value: str):
    key = key.strip()

    if not key:
        raise HTTPException(status_code=400, detail="key must not be empty.")

    state_before = copy.deepcopy(state)
    previous_value = state["config"].get(key)

    state["config"][key] = value

    _record("set_config", {"key": key, "value": value}, state_before=state_before)

    return {
        "status": "success",
        "key": key,
        "value": value,
        "previous_value": previous_value,
    }

@app.post("/reset/{task_id}")
def reset_endpoint(task_id: str):
    try:
        reset(task_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown task_id '{task_id}'"
        )
    return {"status": "reset", "task_id": task_id}


@app.get("/state")
def get_state():
    return state

@app.get("/history")
def get_history():
    return state["history"]

@app.get("/check/{task_id}")
def check_task(task_id: str):
    from app.checker import check

    try:
        return check(task_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown task_id '{task_id}'"
        )