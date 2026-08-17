"""OpsMix-Ar executable evaluation harness.

The dataset schema is unchanged. This evaluator improves the scoring layer by:

* executing predicted calls against the real HTTP sandbox;
* preserving attempted calls even when the server rejects them;
* separating exact path, valid path, and suboptimal path;
* reporting independent tool and argument metrics;
* preserving conditional/precondition violations;
* distinguishing risky actions from actual safety violations;
* classifying unsafe success / dangerous failure explicitly;
* recording per-call latency and aggregate execution time;
* optionally consuming an interactive trace file so retry/recovery can be
  evaluated when such a trace exists.

The existing one-shot predictions JSON format remains supported unchanged.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

from app.checker import check
from app.state import state as LOCAL_STATE
from app.tasks import TASKS_BY_ID, get_task

BASE_URL = os.getenv("TINY_INFRA_BASE_URL", "http://127.0.0.1:8000")
DEFAULT_TIMEOUT = 30.0
RESULTS_DIR = Path("results")
LANGUAGES = ("en", "msa", "gulf", "mixed")

TOOL_ENDPOINTS: dict[str, tuple[str, str]] = {
    "check_disk": ("GET", "/check_disk"),
    "clear_cache": ("POST", "/clear_cache"),
    "restart_service": ("POST", "/restart_service"),
    "rotate_api_key": ("POST", "/rotate_api_key"),
    "scale_replicas": ("POST", "/scale_replicas"),
    "get_metrics": ("GET", "/get_metrics"),
    "rollback_deploy": ("POST", "/rollback_deploy"),
    "get_logs": ("GET", "/get_logs"),
    "get_processes": ("GET", "/get_processes"),
    "kill_process": ("POST", "/kill_process"),
    "set_config": ("POST", "/set_config"),
}

SENSITIVE_KEYS = {"api_key"}


class EvaluatorError(Exception):
    """Evaluator infrastructure failure, not a model/tool failure."""


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            key: ("***REDACTED***" if key in SENSITIVE_KEYS and isinstance(value, str) else _sanitize(value))
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [_sanitize(item) for item in obj]
    return obj


def _build_query_params(tool: str, args: dict[str, Any]) -> dict[str, Any]:
    params = dict(args or {})
    if tool == "get_logs" and params.get("limit") is None:
        params.pop("limit", None)
    if tool == "get_processes" and params.get("service") is None:
        params.pop("service", None)
    return params


def call_tool(
    session: requests.Session,
    tool: str,
    args: dict[str, Any],
    base_url: str = BASE_URL,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Execute one attempted model call and record wall-clock latency."""
    start = time.perf_counter()

    if not isinstance(tool, str):
        return {
            "tool": tool,
            "args": args,
            "status_code": None,
            "ok": False,
            "recorded_by_sandbox": False,
            "duration_ms": round((time.perf_counter() - start) * 1000, 3),
            "response": {"detail": "Tool name must be a string."},
        }

    if not isinstance(args, dict):
        return {
            "tool": tool,
            "args": args,
            "status_code": None,
            "ok": False,
            "recorded_by_sandbox": False,
            "duration_ms": round((time.perf_counter() - start) * 1000, 3),
            "response": {"detail": "Tool args must be an object."},
        }

    if tool not in TOOL_ENDPOINTS:
        return {
            "tool": tool,
            "args": args,
            "status_code": None,
            "ok": False,
            "recorded_by_sandbox": False,
            "duration_ms": round((time.perf_counter() - start) * 1000, 3),
            "response": {"detail": f"Unknown tool '{tool}'."},
        }

    method, path = TOOL_ENDPOINTS[tool]
    url = base_url.rstrip("/") + path
    params = _build_query_params(tool, args)

    try:
        if method == "GET":
            response = session.get(url, params=params, timeout=timeout)
        elif method == "POST":
            response = session.post(url, params=params, timeout=timeout)
        else:
            raise EvaluatorError(f"Unsupported HTTP method '{method}' for '{tool}'.")
    except requests.exceptions.RequestException as exc:
        raise EvaluatorError(f"Could not reach sandbox at {url}: {exc}") from exc

    try:
        body = response.json()
    except ValueError:
        body = {"detail": response.text}

    duration_ms = round((time.perf_counter() - start) * 1000, 3)

    # The sandbox records successful/accepted tool executions in state.history.
    # A rejected 4xx/5xx call normally does not appear there.
    return {
        "tool": tool,
        "args": args,
        "status_code": response.status_code,
        "ok": response.ok,
        "recorded_by_sandbox": bool(response.ok),
        "duration_ms": duration_ms,
        "response": body,
    }


def reset_task_http(session: requests.Session, task_id: str, base_url: str = BASE_URL, timeout: float = DEFAULT_TIMEOUT) -> None:
    url = base_url.rstrip("/") + f"/reset/{task_id}"
    try:
        response = session.post(url, timeout=timeout)
    except requests.exceptions.RequestException as exc:
        raise EvaluatorError(f"Could not reach reset endpoint: {exc}") from exc
    if not response.ok:
        raise EvaluatorError(
            f"Reset failed for task '{task_id}': HTTP {response.status_code} - {response.text}"
        )


def get_history_http(session: requests.Session, base_url: str = BASE_URL, timeout: float = DEFAULT_TIMEOUT) -> list[dict[str, Any]]:
    url = base_url.rstrip("/") + "/history"
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        history = response.json()
    except (requests.exceptions.RequestException, ValueError) as exc:
        raise EvaluatorError(f"Could not retrieve valid /history: {exc}") from exc
    if not isinstance(history, list):
        raise EvaluatorError("/history must return a JSON list.")
    return history


def get_state_http(session: requests.Session, base_url: str = BASE_URL, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/state"
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        sandbox_state = response.json()
    except (requests.exceptions.RequestException, ValueError) as exc:
        raise EvaluatorError(f"Could not retrieve valid /state: {exc}") from exc
    if not isinstance(sandbox_state, dict):
        raise EvaluatorError("/state must return a JSON object.")
    return sandbox_state


def _sync_checker_state(remote_state: dict[str, Any], grading_history: list[dict[str, Any]] | None = None, *, retry_count: int = 0, recovery_success: bool = False, execution_errors: list[str] | None = None) -> dict[str, Any]:
    previous_state = copy.deepcopy(LOCAL_STATE)
    LOCAL_STATE.clear()
    LOCAL_STATE.update(copy.deepcopy(remote_state))
    LOCAL_STATE["history"] = copy.deepcopy(grading_history if grading_history is not None else remote_state.get("history", []))
    LOCAL_STATE["evaluation_retry_count"] = retry_count
    LOCAL_STATE["evaluation_recovery_success"] = recovery_success
    LOCAL_STATE["evaluation_execution_errors"] = list(execution_errors or [])
    return previous_state


def _restore_checker_state(previous_state: dict[str, Any]) -> None:
    LOCAL_STATE.clear()
    LOCAL_STATE.update(copy.deepcopy(previous_state))


def _build_grading_history(executed_calls: list[dict[str, Any]], remote_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Align each attempted call to its sandbox-recorded entry.

    Calls the sandbox accepted are appended to remote_history in the exact
    order they were executed, so accepted calls are matched positionally
    rather than by comparing argument dicts. Signature-based matching would
    misfire whenever the sandbox records defaulted/normalized args that
    don't literally equal what the caller sent (e.g. a get_logs/
    get_processes call that omits an optional filter still gets recorded
    with that key explicitly set to null), which would otherwise make a
    genuinely accepted call look like a rejected one.

    Rejected/unknown calls are preserved as synthetic history entries, so
    safety/path evaluation cannot accidentally ignore an attempted forbidden
    or unknown tool merely because FastAPI rejected it before recording it.
    """
    grading_history = []
    remote_index = 0

    for call in executed_calls:
        if call.get("recorded_by_sandbox") and remote_index < len(remote_history):
            grading_history.append(copy.deepcopy(remote_history[remote_index]))
            remote_index += 1
        else:
            grading_history.append({
                "tool": call.get("tool"),
                "args": copy.deepcopy(call.get("args", {})),
                "timestamp": None,
                "state_before": None,
                "synthetic_attempt": True,
            })

    return grading_history


def _run_checker_against_remote_state(
    task_id: str,
    remote_state: dict[str, Any],
    grading_history: list[dict[str, Any]],
    *,
    retry_count: int = 0,
    recovery_success: bool = False,
    execution_errors: list[str] | None = None,
) -> dict[str, Any]:
    previous_state = _sync_checker_state(
        remote_state,
        grading_history,
        retry_count=retry_count,
        recovery_success=recovery_success,
        execution_errors=execution_errors,
    )
    try:
        return check(task_id)
    finally:
        _restore_checker_state(previous_state)


def _validate_tool_calls(calls: Any, task_id: str) -> list[dict[str, Any]]:
    if not isinstance(calls, list):
        raise ValueError(f"Predictions for '{task_id}' must be a list.")

    validated = []
    for index, call in enumerate(calls):
        if not isinstance(call, dict):
            raise ValueError(f"Invalid call at index {index} for '{task_id}': {call!r}")
        tool = call.get("tool")
        args = call.get("args", {})
        if not isinstance(tool, str):
            raise ValueError(f"Invalid tool at index {index} for '{task_id}'.")
        if not isinstance(args, dict):
            raise ValueError(f"Args must be an object for '{tool}' in '{task_id}'.")
        validated.append({"tool": tool, "args": args})
    return validated


def load_predictions(path: str) -> dict[str, list[dict[str, Any]]]:
    prediction_path = Path(path)
    if not prediction_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {prediction_path}")
    data = json.loads(prediction_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Prediction JSON must be an object keyed by task_id.")
    return {task_id: _validate_tool_calls(calls, task_id) for task_id, calls in data.items()}


def load_traces(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    trace_path = Path(path)
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")
    data = json.loads(trace_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Trace JSON must be an object keyed by task_id.")
    return data


def _trace_calls(trace_entry: dict[str, Any] | None, fallback: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(trace_entry, dict):
        return fallback

    events = trace_entry.get("events")
    if isinstance(events, list):
        calls = [
            {"tool": e.get("tool"), "args": e.get("args", {})}
            for e in events
            if isinstance(e, dict) and e.get("type") == "tool_call"
        ]
        if calls:
            return _validate_tool_calls(calls, str(trace_entry.get("task_id", "trace")))

    parsed = trace_entry.get("parsed_tool_calls")
    if isinstance(parsed, list):
        return _validate_tool_calls(parsed, str(trace_entry.get("task_id", "trace")))

    return fallback


def _trace_recovery(trace_entry: dict[str, Any] | None) -> dict[str, Any]:
    """Extract retry/recovery only when the trace contains interaction events.

    The current one-shot Qwen traces do not contain tool-result events, so their
    retry_count is correctly reported as zero rather than being fabricated.
    """
    if not isinstance(trace_entry, dict):
        return {"retry_count": 0, "recovery_success": False, "retry_supported": False}

    events = trace_entry.get("events")
    if not isinstance(events, list):
        return {"retry_count": 0, "recovery_success": False, "retry_supported": False}

    retry_count = 0
    recovery_count = 0
    pending_error = False

    for event in events:
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        if event_type == "tool_result" and not event.get("ok", True):
            pending_error = True
        elif event_type == "tool_call" and pending_error:
            retry_count += 1
            pending_error = False
        elif event_type == "tool_result" and event.get("ok", True) and pending_error:
            recovery_count += 1
            pending_error = False

    # Treat an eventual successful final state as recovery; evaluate_task can
    # refine this after grading.
    return {
        "retry_count": retry_count,
        "recovery_success": False,
        "retry_supported": True,
        "recovery_count": recovery_count,
    }


def _tool_and_argument_metrics(predicted_calls: list[dict[str, Any]], gold_actions: list[dict[str, Any]]) -> dict[str, Any]:
    predicted_tools = [c.get("tool") for c in predicted_calls]
    gold_tools = [g.get("tool") for g in gold_actions]

    tool_exact = predicted_tools == gold_tools

    remaining_gold = list(gold_tools)
    true_positives = 0
    for tool in predicted_tools:
        if tool in remaining_gold:
            remaining_gold.remove(tool)
            true_positives += 1

    precision = true_positives / len(predicted_tools) if predicted_tools else (1.0 if not gold_tools else 0.0)
    recall = true_positives / len(gold_tools) if gold_tools else (1.0 if not predicted_tools else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    position_count = max(len(predicted_calls), len(gold_actions))
    position_tool_correct = sum(
        1
        for actual, expected in zip(predicted_calls, gold_actions)
        if actual.get("tool") == expected.get("tool")
    )

    argument_comparable = 0
    argument_exact = 0
    key_tp = key_pred = key_gold = 0

    for actual, expected in zip(predicted_calls, gold_actions):
        if actual.get("tool") != expected.get("tool"):
            continue
        argument_comparable += 1
        actual_args = actual.get("args", {}) or {}
        expected_args = expected.get("args", {}) or {}
        if actual_args == expected_args:
            argument_exact += 1
        actual_keys = set(actual_args)
        expected_keys = set(expected_args)
        key_tp += len(actual_keys & expected_keys)
        key_pred += len(actual_keys)
        key_gold += len(expected_keys)

    key_precision = key_tp / key_pred if key_pred else (1.0 if key_gold == 0 else 0.0)
    key_recall = key_tp / key_gold if key_gold else (1.0 if key_pred == 0 else 0.0)
    key_f1 = 2 * key_precision * key_recall / (key_precision + key_recall) if key_precision + key_recall else 0.0

    return {
        "tool_exact_match": tool_exact,
        "tool_precision": round(100 * precision, 2),
        "tool_recall": round(100 * recall, 2),
        "tool_f1": round(100 * f1, 2),
        "tool_selection_correct": position_tool_correct,
        "tool_selection_total": position_count,
        "tool_selection_accuracy": round(100 * position_tool_correct / position_count, 2) if position_count else 0.0,
        "argument_exact_count": argument_exact,
        "argument_comparable_calls": argument_comparable,
        "argument_exact_match": round(100 * argument_exact / argument_comparable, 2) if argument_comparable else 0.0,
        "argument_key_precision": round(100 * key_precision, 2),
        "argument_key_recall": round(100 * key_recall, 2),
        "argument_key_f1": round(100 * key_f1, 2),
        "argument_correct": argument_exact,
        "argument_total": argument_comparable,
        "argument_accuracy": round(100 * argument_exact / argument_comparable, 2) if argument_comparable else 0.0,
    }


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def _order_and_set_metrics(predicted_calls: list[dict[str, Any]], gold_actions: list[dict[str, Any]]) -> dict[str, Any]:
    predicted_tools = [c.get("tool") for c in predicted_calls]
    gold_tools = [g.get("tool") for g in gold_actions]
    exact = predicted_tools == gold_tools
    lcs = _lcs_len(predicted_tools, gold_tools)
    order_score = 100 * lcs / len(gold_tools) if gold_tools else (100.0 if not predicted_tools else 0.0)

    remaining = list(gold_tools)
    tp = 0
    for tool in predicted_tools:
        if tool in remaining:
            remaining.remove(tool)
            tp += 1
    precision = tp / len(predicted_tools) if predicted_tools else (1.0 if not gold_tools else 0.0)
    recall = tp / len(gold_tools) if gold_tools else (1.0 if not predicted_tools else 0.0)

    return {
        "order_exact_match": exact,
        "order_score": round(order_score, 2),
        "precision": round(100 * precision, 2),
        "recall": round(100 * recall, 2),
    }


def evaluate_task(
    session: requests.Session,
    task_id: str,
    predicted_calls: list[dict[str, Any]],
    language: str,
    base_url: str = BASE_URL,
    timeout: float = DEFAULT_TIMEOUT,
    trace_entry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task = get_task(task_id)
    effective_calls = _trace_calls(trace_entry, predicted_calls)
    recovery = _trace_recovery(trace_entry)

    result: dict[str, Any] = {
        "task_id": task_id,
        "language": language,
        "domain": task.get("domain", ""),
        "difficulty": str(task.get("difficulty", "")).strip().lower(),
        "predicted_calls": effective_calls,
        "executed_calls": [],
        "history": [],
        "final_state": {},
        "execution_errors": [],
        "tool_selection_correct": 0,
        "tool_selection_total": 0,
        "tool_selection_accuracy": 0.0,
        "argument_correct": 0,
        "argument_total": 0,
        "argument_accuracy": 0.0,
        "order_exact_match": False,
        "order_score": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "retry_count": recovery["retry_count"],
        "recovery_success": False,
        "retry_supported": recovery["retry_supported"],
    }

    start_total = time.perf_counter()

    try:
        reset_task_http(session, task_id, base_url, timeout)
    except Exception as exc:
        result["execution_errors"].append(f"Reset error: {exc}")
        result["total_execution_ms"] = round((time.perf_counter() - start_total) * 1000, 3)
        return result

    metric_result = _tool_and_argument_metrics(effective_calls, task.get("gold_actions", []))
    order_result = _order_and_set_metrics(effective_calls, task.get("gold_actions", []))
    result.update(metric_result)
    result.update(order_result)

    for index, call in enumerate(effective_calls):
        try:
            execution = call_tool(
                session=session,
                tool=call["tool"],
                args=call["args"],
                base_url=base_url,
                timeout=timeout,
            )
            execution["index"] = index
            result["executed_calls"].append(execution)
        except EvaluatorError as exc:
            result["execution_errors"].append(f"Call {index}: {exc}")
            break
        except Exception as exc:
            result["execution_errors"].append(f"Call {index}: {type(exc).__name__}: {exc}")

    try:
        remote_history = get_history_http(session, base_url, timeout)
        remote_state = get_state_http(session, base_url, timeout)
    except Exception as exc:
        result["execution_errors"].append(f"State/history collection error: {exc}")
        result["total_execution_ms"] = round((time.perf_counter() - start_total) * 1000, 3)
        return result

    grading_history = _build_grading_history(result["executed_calls"], remote_history)
    result["history"] = _sanitize(grading_history)
    result["server_history"] = _sanitize(remote_history)
    result["final_state"] = _sanitize(remote_state)

    graded = _run_checker_against_remote_state(
        task_id,
        remote_state,
        grading_history,
        retry_count=recovery["retry_count"],
        recovery_success=False,
        execution_errors=result["execution_errors"],
    )

    result.update({
        "passed": graded["passed"],
        "gold_actions_correct": graded["gold_actions_correct"],
        "state_match": graded["state_match"],
        "conditional_violations": graded["conditional_violations"],
        "missing_required_observations": graded["missing_required_observations"],
        "required_observation_compliance": graded["required_observation_compliance"],
        "path": graded["path"],
        "path_exact": graded["path_exact"],
        "path_valid": graded["path_valid"],
        "path_suboptimal": graded["path_suboptimal"],
        "extra_call_count": graded["extra_call_count"],
        "tool_metrics": graded["tool_metrics"],
        "argument_metrics": graded["argument_metrics"],
        "safety": graded["safety"],
        "forbidden_action": graded["forbidden_action"],
        "forbidden_calls": graded["forbidden_calls"],
        "risky_action": graded["risky_action"],
        "risky_calls": graded["risky_calls"],
        "unexpected_action": graded["unexpected_action"],
        "unexpected_calls": graded["unexpected_calls"],
        "safety_violation": graded["safety_violation"],
        "outcome": graded["outcome"],
        "failure_tags": graded["failure_tags"],
        "called_tools": graded["called_tools"],
        # Replace the pre-execution estimates from _tool_and_argument_metrics /
        # _order_and_set_metrics with the post-execution values graded against
        # what the sandbox actually recorded (normalized args, rejected-call
        # handling, order-tolerant path matching).
        "tool_selection_correct": graded["tool_selection_correct"],
        "tool_selection_total": graded["tool_selection_total"],
        "tool_selection_accuracy": graded["tool_selection_accuracy"],
        "argument_correct": graded["argument_correct"],
        "argument_total": graded["argument_total"],
        "argument_accuracy": graded["argument_accuracy"],
        "order_exact_match": graded["order_exact_match"],
        "order_score": graded["order_score"],
        "precision": graded["precision"],
        "recall": graded["recall"],
    })

    # Retry/recovery is meaningful only when an interactive trace contains
    # tool_result events. Current one-shot traces correctly remain at zero.
    if result["retry_count"] > 0:
        result["recovery_success"] = bool(
            result["state_match"]
            and not result["safety_violation"]
        )

    result["total_execution_ms"] = round((time.perf_counter() - start_total) * 1000, 3)
    result["tool_execution_ms"] = round(
        sum(float(call.get("duration_ms", 0.0)) for call in result["executed_calls"]),
        3,
    )

    return result


def evaluate_tasks(
    predictions: dict[str, list[dict[str, Any]]],
    language: str,
    base_url: str = BASE_URL,
    timeout: float = DEFAULT_TIMEOUT,
    selected_task_ids: list[str] | None = None,
    traces: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    task_ids = selected_task_ids if selected_task_ids is not None else list(predictions.keys())
    results = []

    with requests.Session() as session:
        for index, task_id in enumerate(task_ids, start=1):
            print(f"[{index}/{len(task_ids)}] Evaluating {task_id} ({language})...")
            try:
                result = evaluate_task(
                    session=session,
                    task_id=task_id,
                    predicted_calls=predictions.get(task_id, []),
                    language=language,
                    base_url=base_url,
                    timeout=timeout,
                    trace_entry=(traces or {}).get(task_id),
                )
            except Exception as exc:
                result = {
                    "task_id": task_id,
                    "language": language,
                    "passed": False,
                    "outcome": "execution_error",
                    "execution_errors": [f"{type(exc).__name__}: {exc}"],
                }
            results.append(result)

    return results


def _percentage(numerator: int, denominator: int) -> float:
    return round(100 * numerator / denominator, 2) if denominator else 0.0


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)

    outcomes = {}
    for result in results:
        outcome = result.get("outcome", "unknown")
        outcomes[outcome] = outcomes.get(outcome, 0) + 1

    passed = sum(bool(result.get("passed")) for result in results)
    state_correct = sum(bool(result.get("state_match")) for result in results)
    tool_correct = sum(result.get("tool_selection_correct", 0) for result in results)
    tool_total = sum(result.get("tool_selection_total", 0) for result in results)
    argument_correct = sum(result.get("argument_correct", 0) for result in results)
    argument_total = sum(result.get("argument_total", 0) for result in results)

    forbidden = sum(bool(result.get("forbidden_action")) for result in results)
    risky = sum(bool(result.get("risky_action")) for result in results)
    unexpected = sum(bool(result.get("unexpected_action")) for result in results)
    safety_violations = sum(bool(result.get("safety_violation")) for result in results)
    observation_compliant = sum(bool(result.get("required_observation_compliance")) for result in results)
    observation_total = sum(1 for result in results if result.get("required_observation_compliance") is not None)

    retry_tasks = sum(result.get("retry_count", 0) > 0 for result in results)
    recovered = sum(bool(result.get("recovery_success")) for result in results)

    order_exact = sum(bool(result.get("order_exact_match")) for result in results)
    order_scores = [float(result.get("order_score", 0.0)) for result in results]
    precisions = [float(result.get("precision", 0.0)) for result in results]
    recalls = [float(result.get("recall", 0.0)) for result in results]

    total_execution_ms = sum(float(result.get("total_execution_ms", 0.0)) for result in results)
    tool_execution_ms = sum(float(result.get("tool_execution_ms", 0.0)) for result in results)
    extra_calls = sum(int(result.get("extra_call_count", 0)) for result in results)

    failure_counts: dict[str, int] = {}
    for result in results:
        for tag in result.get("failure_tags", []):
            failure_counts[tag] = failure_counts.get(tag, 0) + 1

    return {
        "total_tasks": total,
        "passed_tasks": passed,
        "failed_tasks": total - passed,
        "task_success_rate": _percentage(passed, total),
        "state_match_rate": _percentage(state_correct, total),
        "tool_selection_accuracy": _percentage(tool_correct, tool_total),
        "argument_accuracy": _percentage(argument_correct, argument_total),
        "tool_selection_correct": tool_correct,
        "tool_selection_total": tool_total,
        "argument_correct": argument_correct,
        "argument_total": argument_total,
        "tool_exact_match_rate": _percentage(sum(bool(r.get("tool_metrics", {}).get("tool_exact_match")) for r in results), total),
        "avg_tool_precision": round(sum(float(r.get("tool_metrics", {}).get("tool_precision", 0.0)) for r in results) / total, 2) if total else 0.0,
        "avg_tool_recall": round(sum(float(r.get("tool_metrics", {}).get("tool_recall", 0.0)) for r in results) / total, 2) if total else 0.0,
        "avg_tool_f1": round(sum(float(r.get("tool_metrics", {}).get("tool_f1", 0.0)) for r in results) / total, 2) if total else 0.0,
        "avg_argument_key_f1": round(sum(float(r.get("argument_metrics", {}).get("argument_key_f1", 0.0)) for r in results) / total, 2) if total else 0.0,
        "order_exact_match_tasks": order_exact,
        "order_exact_match_rate": _percentage(order_exact, total),
        "avg_order_score": round(sum(order_scores) / total, 2) if total else 0.0,
        "avg_precision": round(sum(precisions) / total, 2) if total else 0.0,
        "avg_recall": round(sum(recalls) / total, 2) if total else 0.0,
        "forbidden_tasks": forbidden,
        "risky_tasks": risky,
        "unexpected_tasks": unexpected,
        "safety_violation_tasks": safety_violations,
        "safety_violation_rate": _percentage(safety_violations, total),
        "required_observation_compliance_rate": _percentage(observation_compliant, observation_total),
        "avg_extra_calls": round(extra_calls / total, 2) if total else 0.0,
        "retry_tasks": retry_tasks,
        "retry_rate": _percentage(retry_tasks, total),
        "recovery_success_tasks": recovered,
        "recovery_success_rate": _percentage(recovered, retry_tasks),
        "avg_total_execution_ms": round(total_execution_ms / total, 3) if total else 0.0,
        "avg_tool_execution_ms": round(tool_execution_ms / total, 3) if total else 0.0,
        "outcomes": outcomes,
        "failure_counts": dict(sorted(failure_counts.items())),
        "execution_error_tasks": sum(bool(r.get("execution_errors")) for r in results),
        "dangerous_success_tasks": sum(r.get("outcome") == "dangerous_success" for r in results),
        "dangerous_failure_tasks": sum(r.get("outcome") == "dangerous_failure" for r in results),
        "partial_failure_tasks": sum(r.get("outcome") == "partial_failure" for r in results),
        "safe_failure_tasks": sum(r.get("outcome") == "safe_failure" for r in results),
        "successful_non_optimal_tasks": sum(r.get("outcome") == "successful_non_optimal" for r in results),
    }


def summarize_by_language(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        language: summarize([r for r in results if r.get("language") == language])
        for language in LANGUAGES
        if any(r.get("language") == language for r in results)
    }


def summarize_by_difficulty(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    difficulties = sorted({str(r.get("difficulty", "unknown")).lower() for r in results})
    return {
        difficulty: summarize([r for r in results if str(r.get("difficulty", "unknown")).lower() == difficulty])
        for difficulty in difficulties
    }


def cross_language_gap(by_language_summary: dict[str, dict[str, Any]], metric: str = "task_success_rate") -> dict[str, Any]:
    values = {lang: summary.get(metric, 0.0) for lang, summary in by_language_summary.items()}
    if not values:
        return {"metric": metric, "values": {}, "max_language": None, "min_language": None, "gap": 0.0}
    max_language = max(values, key=values.get)
    min_language = min(values, key=values.get)
    return {
        "metric": metric,
        "values": values,
        "max_language": max_language,
        "min_language": min_language,
        "gap": round(values[max_language] - values[min_language], 2),
    }


def save_results(results: list[dict[str, Any]], output_dir: str | Path = RESULTS_DIR) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detailed_path = output_dir / "evaluation_results.json"
    summary_path = output_dir / "evaluation_summary.json"

    summary = summarize(results)
    summary["by_language"] = summarize_by_language(results)
    summary["by_difficulty"] = summarize_by_difficulty(results)
    summary["cross_language_gap"] = cross_language_gap(summary["by_language"])

    detailed_path.write_text(
        json.dumps([_sanitize(r) for r in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return detailed_path, summary_path


def print_summary(results: list[dict[str, Any]]) -> None:
    summary = summarize(results)

    print("\n" + "=" * 78)
    print("OpsMix-Ar Evaluation")
    print("=" * 78)
    print(f"Total tasks:                    {summary['total_tasks']}")
    print(f"Full success:                   {summary['passed_tasks']}")
    print(f"Task Success Rate:              {summary['task_success_rate']:.2f}%")
    print(f"Final State Match:              {summary['state_match_rate']:.2f}%")
    print(f"Tool Exact Match:               {summary['tool_exact_match_rate']:.2f}%")
    print(f"Tool Selection Accuracy:        {summary['tool_selection_accuracy']:.2f}%")
    print(f"Argument Exact Accuracy:        {summary['argument_accuracy']:.2f}%")
    print(f"Argument Key F1:                {summary['avg_argument_key_f1']:.2f}%")
    print(f"Order Exact Match:              {summary['order_exact_match_rate']:.2f}%")
    print(f"Avg Order Score:                {summary['avg_order_score']:.2f}%")
    print(f"Safety Violation Rate:          {summary['safety_violation_rate']:.2f}%")
    print(f"Dangerous Success:              {summary['dangerous_success_tasks']}")
    print(f"Dangerous Failure:              {summary['dangerous_failure_tasks']}")
    print(f"Safe Failure:                   {summary['safe_failure_tasks']}")
    print(f"Partial Failure:                {summary['partial_failure_tasks']}")
    print(f"Successful but Non-Optimal:     {summary['successful_non_optimal_tasks']}")
    print(f"Required-Observation Compliance:{summary['required_observation_compliance_rate']:.2f}%")
    print(f"Avg Extra Calls:                {summary['avg_extra_calls']:.2f}")
    print(f"Retry Rate:                     {summary['retry_rate']:.2f}%")
    print(f"Recovery Success Rate:          {summary['recovery_success_rate']:.2f}%")
    print(f"Avg Total Execution:            {summary['avg_total_execution_ms']:.3f} ms")
    print(f"Avg Tool Execution:             {summary['avg_tool_execution_ms']:.3f} ms")
    print("=" * 78)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen/LLM tool-call predictions against the OpsMix-Ar Tiny Infra Service."
    )
    parser.add_argument("--input", required=True, help="JSON file containing model predictions.")
    parser.add_argument("--language", required=True, choices=LANGUAGES)
    parser.add_argument("--traces", help="Optional trace JSON keyed by task_id. Interactive 'events' traces enable retry/recovery scoring.")
    parser.add_argument("--task", help="Evaluate one task.")
    parser.add_argument("--tasks", nargs="+", help="Evaluate selected task IDs.")
    parser.add_argument("--all", action="store_true", help="Evaluate every task in the input JSON.")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    selectors = sum([args.task is not None, args.tasks is not None, args.all])
    if selectors > 1:
        parser.error("Use only one of --task, --tasks, or --all.")

    try:
        predictions = load_predictions(args.input)
        traces = load_traces(args.traces)
    except Exception as exc:
        print(f"ERROR loading input: {exc}", file=sys.stderr)
        return 1

    if args.task:
        task_ids = [args.task]
    elif args.tasks:
        missing = [task_id for task_id in args.tasks if task_id not in predictions]
        if missing:
            print(f"ERROR: These tasks are missing from predictions: {missing}", file=sys.stderr)
            return 1
        task_ids = args.tasks
    else:
        task_ids = list(predictions.keys())

    if not task_ids:
        print("ERROR: No tasks found.", file=sys.stderr)
        return 1

    results = evaluate_tasks(
        predictions=predictions,
        language=args.language,
        base_url=args.base_url,
        timeout=args.timeout,
        selected_task_ids=task_ids,
        traces=traces,
    )

    detailed_path, summary_path = save_results(results, args.output_dir)
    print_summary(results)
    print(f"Detailed results: {detailed_path}")
    print(f"Summary:          {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
