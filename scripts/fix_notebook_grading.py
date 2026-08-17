"""Patch evaluate_agentic_task (cell 27) in
OpsMix_Ar_Agentic_Trial_EN_MemoryFixed.ipynb.

Two bugs, both independent copies of issues already fixed in
app/evaluate.py -- this notebook has its own evaluate_agentic_task rather
than calling app.evaluate.evaluate_task directly (it needs turn-by-turn
control that the batch function doesn't support), so the fixes never
propagated here:

1. _build_grading_history was called with `effective_calls` -- a stripped
   {"tool", "args"} view of each call with the "ok"/"recorded_by_sandbox"
   flag removed. Since app/evaluate.py's _build_grading_history now aligns
   calls to sandbox history positionally using that flag, every call here
   read as falsy and was treated as an unrecorded/rejected synthetic
   attempt, nulling state_before on the ENTIRE grading history regardless
   of whether the call actually succeeded. That breaks every check that
   depends on state_before (conditional_actions preconditions, risky-call
   justification), producing false-positive safety violations. Fixed by
   passing agent_result["executed_calls"] (which retains "ok") instead.

2. The result.update({...}) block after grading never overwrote the
   flat tool_selection_*/argument_*/order_*/precision/recall fields set
   earlier from the strict, pre-execution _tool_and_argument_metrics/
   _order_and_set_metrics call, so the printed "Tool Selection Accuracy"/
   "Argument Accuracy" reflect that stricter positional metric instead of
   the actual graded one. Fixed by adding those keys from `graded`.
"""
import json
from pathlib import Path

NB_PATH = Path("OpsMix_Ar_Agentic_Trial_EN_MemoryFixed.ipynb")

OLD_CALL = 'grading_history = _build_grading_history(effective_calls, remote_history)'
NEW_CALL = 'grading_history = _build_grading_history(agent_result["executed_calls"], remote_history)'

OLD_UPDATE_TAIL = '''            "outcome": graded["outcome"],
            "failure_tags": graded["failure_tags"],
            "called_tools": graded["called_tools"],
        })'''

NEW_UPDATE_TAIL = '''            "outcome": graded["outcome"],
            "failure_tags": graded["failure_tags"],
            "called_tools": graded["called_tools"],
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
        })'''


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cell = nb["cells"][27]
    src = "".join(cell["source"])
    assert "def evaluate_agentic_task(" in src, "cell 27 is not evaluate_agentic_task"

    assert src.count(OLD_CALL) == 1, f"expected exactly one match for grading_history call, found {src.count(OLD_CALL)}"
    src = src.replace(OLD_CALL, NEW_CALL)

    assert src.count(OLD_UPDATE_TAIL) == 1, "expected exactly one match for result.update tail"
    src = src.replace(OLD_UPDATE_TAIL, NEW_UPDATE_TAIL)

    cell["source"] = src.splitlines(keepends=True)
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Patched cell 27: grading_history call + flat metric overwrite.")


if __name__ == "__main__":
    main()
