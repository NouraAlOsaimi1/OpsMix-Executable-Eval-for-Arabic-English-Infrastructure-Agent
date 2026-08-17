"""Raise run_agentic_task's max_new_tokens_per_step (cell 25).

All 13 parse-failed tasks in the post-prompt-tightening 40-task run showed
the identical failure shape: generation cut off mid <think> block (no
closing tag, no JSON), always around ~2048 tokens -- the exact
max_new_tokens_per_step budget. Confirmed via raw_turns tails, e.g.
scale_replicas_011 and restart_service_010, which cut off mid-sentence
while reasoning through the tightened prompt's rules. Doubling the budget
gives Qwen3-4B-Thinking room to finish deliberating before the hard cutoff.
"""
import json
from pathlib import Path

NB_PATH = Path("OpsMix_Ar_Agentic_Trial_EN_MemoryFixed.ipynb")

OLD = "    max_new_tokens_per_step: int = 2048,"
NEW = "    max_new_tokens_per_step: int = 4096,"


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cell = nb["cells"][25]
    src = "".join(cell["source"])
    assert "def run_agentic_task(" in src, "cell 25 is not run_agentic_task"
    assert src.count(OLD) == 1, f"expected exactly one match, found {src.count(OLD)}"
    src = src.replace(OLD, NEW)
    cell["source"] = src.splitlines(keepends=True)
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Patched cell 25: max_new_tokens_per_step 2048 -> 4096.")


if __name__ == "__main__":
    main()
