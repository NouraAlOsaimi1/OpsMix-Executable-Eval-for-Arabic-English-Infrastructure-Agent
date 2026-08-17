"""Append practical run notes to the existing "**Note:**" cell (index 1) in
OpsMix_Ar_Agentic_Trial_4Models_2000Tasks.ipynb, without touching what's
already there (MAX_STEPS/MAX_NEW_TOKENS defaults, per-person model
assignment, the HF-token reminder).

Covers what's missing for someone running this without the context built
up over the rest of the project: hardware requirement, realistic runtime,
how to run just one assigned model instead of all four, what happens on
interruption, where results land, and the RunPod ephemeral-storage trap
(checkpointing survives a disconnect/crash but not a pod actually being
stopped/terminated without a persistent volume).
"""
import json
from pathlib import Path

NB_PATH = Path("OpsMix_Ar_Agentic_Trial_4Models_2000Tasks.ipynb")

ADDITIONAL_NOTES = (
    "\n"
    "* **Hardware:** requires a real GPU with at least ~16-24 GB VRAM (e.g. an "
    "RTX 4090). Models are loaded one at a time, but each is a 7-9B parameter model.\n"
    "\n"
    "* **Runtime:** this is a long run, not a \"run it and wait a few minutes\" "
    "notebook. All 4 models x 2000 tasks each is on the order of a day or more of "
    "GPU time combined; even one model's 2000 tasks alone can take several hours.\n"
    "\n"
    "* **Running just your assigned model:** to run only the model listed next to "
    "your name above instead of all four, edit `MODEL_ORDER` in the model-registry "
    "cell (Step 4) to a list with just that one key -- e.g. `MODEL_ORDER = "
    "[\"llama3.1-8b\"]` for Layan. Leaving all four keys in `MODEL_ORDER` runs them "
    "sequentially in one session.\n"
    "\n"
    "* **If interrupted:** just re-run the notebook from the top (or at least from "
    "the sandbox-start cell onward). Each model has its own checkpoint file and "
    "resumes automatically from its last saved run -- completed (task, language) "
    "pairs are never re-run, and a fully-completed model is skipped without even "
    "being loaded.\n"
    "\n"
    "* **Where results go:** saved automatically -- Google Drive on Colab, or "
    "`/workspace/OpsMix-Ar_4models/checkpoints_4models/` on RunPod -- one JSON file "
    "per model, plus `all_models_summary.json` at the end. No manual path setup "
    "needed.\n"
    "\n"
    "* **RunPod only:** if `/workspace` is not on a persistent Network Volume, "
    "*stopping or terminating the pod* (not just a disconnect or kernel crash) wipes "
    "the checkpoint files with it. Attach a Network Volume if the run needs to "
    "survive between sessions.\n"
    "\n"
    "* **HF token:** only `meta-llama/Llama-3.1-8B-Instruct` needs one (its license "
    "must also be accepted on its model page first) -- Fanar, ALLaM, and Qwen3 don't "
    "require it."
)


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cell = nb["cells"][1]
    assert cell["cell_type"] == "markdown"
    current = "".join(cell["source"])
    assert current.startswith("**Note:**"), "cell 1 is not the expected Note cell"

    updated = current.rstrip("\n") + "\n" + ADDITIONAL_NOTES
    cell["source"] = updated.splitlines(keepends=True)

    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Appended additional run notes to cell 1.")


if __name__ == "__main__":
    main()
