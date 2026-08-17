"""Translate all markdown cells in
OpsMix_Ar_Agentic_Trial_4Models_2000Tasks.ipynb to English.

Code cells (and their comments, which are already English/bilingual-code
style) are untouched -- only the 14 markdown section headers/notes.
"""
import json
from pathlib import Path

NB_PATH = Path("OpsMix_Ar_Agentic_Trial_4Models_2000Tasks.ipynb")

# index -> new markdown source (verified against the current cell list before writing)
TRANSLATIONS = {
    0: (
        "# OpsMix-Ar — 4-Model Comparison Trial — 2000 Tasks per Model\n"
        "\n"
        "Runs 4 models sequentially (Llama-3.1-8B-Instruct, QCRI/Fanar-1-9B-Instruct, "
        "ALLaM-7B-Instruct-preview, Qwen3-4B-Thinking-2507) across all 500 tasks × 4 "
        "languages (2000 runs per model, 8000 total). Each model has its own resumable "
        "checkpoint, and each model's results are saved to a separate JSON file. "
        "**This is a very long run (likely several days of aggregated GPU time) — the "
        "checkpointing system is what makes it practical across more than one session.**"
    ),
    1: "## Step 1: Install dependencies — run, then restart the session if prompted",
    6: "## Step 2: Clone the repo and load the dataset",
    12: (
        "## Step 3: Log in to Hugging Face (required for Llama-3.1)\n"
        "\n"
        "`meta-llama/Llama-3.1-8B-Instruct` is a **gated** model — you must first accept "
        "its license on its page (https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), "
        "then either set a token in the `HF_TOKEN` environment variable before running this "
        "cell, or log in interactively. Without this, loading Llama will fail while Fanar / "
        "ALLaM / Qwen3 (unaffected) continue normally."
    ),
    14: (
        "## Step 4: The four-model registry\n"
        "\n"
        "`use_thinking` enables the `enable_thinking` chat-template kwarg (Qwen3-specific "
        "only). `attn_implementation` for Fanar is set to `\"eager\"` as a precaution — "
        "Gemma-2 models (which Fanar is built on) have a documented output-quality issue "
        "with some SDPA/flash-attention implementations."
    ),
    17: "## Step 5: System prompt (identical to the original trial, unmodified)",
    19: "## Step 6: Single tool-call parser (same logic as the original)",
    21: "## Step 7: Start the real Tiny Infra Service (sandbox)",
    23: "## Step 8: Import the ready-made evaluate.py functions (no modifications)",
    25: (
        "## Step 9: `run_agentic_task` — same generate ↔ execute loop as the original, "
        "with multi-model support\n"
        "\n"
        "The only addition over the original version: a one-time-per-task check of "
        "whether the tokenizer accepts a separate \"system\" role (some Gemma-based model "
        "families reject it), with a fallback that folds the system prompt into the first "
        "user message if it's rejected."
    ),
    27: (
        "## Step 10: `evaluate_agentic_task` — same grading logic as the original, with "
        "`use_thinking` passed through"
    ),
    29: (
        "## Step 11: The full run — 4 models × 2000 tasks, with a separate checkpoint "
        "per model\n"
        "\n"
        "For each model: if it's already fully complete in the checkpoint, loading it is "
        "skipped entirely. If it's partial, it resumes from the last saved run. After it "
        "finishes (or is skipped), memory is freed before the next model loads."
    ),
    31: "## Step 12: Final summary — comparing the four models",
    33: "## Step 13 (optional): Stop the server when done",
}


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    for idx, new_text in TRANSLATIONS.items():
        cell = cells[idx]
        assert cell["cell_type"] == "markdown", f"cell {idx} is not markdown"
        cell["source"] = new_text.splitlines(keepends=True)

    # Sanity: every markdown cell in the notebook was covered.
    md_indices = {i for i, c in enumerate(cells) if c["cell_type"] == "markdown"}
    assert md_indices == set(TRANSLATIONS.keys()), (
        f"mismatch -- markdown cells in file: {sorted(md_indices)}, "
        f"translated: {sorted(TRANSLATIONS.keys())}"
    )

    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Translated {len(TRANSLATIONS)} markdown cells to English.")


if __name__ == "__main__":
    main()
