"""Switch the Llama entry in OpsMix_Ar_Agentic_Trial_4Models_2000Tasks.ipynb
back to the official gated meta-llama/Llama-3.1-8B-Instruct (from the
ungated NousResearch mirror), and restore the HF-login step's wording to
reflect that it's required again for this model specifically.
"""
import json
from pathlib import Path

NB_PATH = Path("OpsMix_Ar_Agentic_Trial_4Models_2000Tasks.ipynb")

OLD_ID = "NousResearch/Meta-Llama-3.1-8B-Instruct"
NEW_ID = "meta-llama/Llama-3.1-8B-Instruct"

NEW_MD_12 = (
    "## Step 3: Log in to Hugging Face (required for Llama-3.1)\n"
    "\n"
    f"`{NEW_ID}` is a **gated** model — you must first accept its license on its page "
    f"(https://huggingface.co/{NEW_ID}), then either set a token in the `HF_TOKEN` "
    "environment variable before running this cell, or log in interactively. Without "
    "this, loading Llama will fail while Fanar / ALLaM / Qwen3 (unaffected) continue "
    "normally."
)

NEW_CODE_13 = (
    'import os\n'
    'from huggingface_hub import login as hf_login\n'
    '\n'
    'HF_TOKEN = os.environ.get("HF_TOKEN")\n'
    'if HF_TOKEN:\n'
    '    hf_login(token=HF_TOKEN)\n'
    '    print("Logged in to Hugging Face Hub via HF_TOKEN.")\n'
    'else:\n'
    '    print(\n'
    '        "WARNING: HF_TOKEN is not set. meta-llama/Llama-3.1-8B-Instruct is gated --\\n"\n'
    '        "accept its license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct,\\n"\n'
    '        "then either set the HF_TOKEN environment variable before running this cell,\\n"\n'
    '        "or run `from huggingface_hub import login; login()` in a new cell now."\n'
    '    )'
)


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    src15 = "".join(cells[15]["source"])
    assert src15.count(OLD_ID) == 1, f"expected exactly one match, found {src15.count(OLD_ID)}"
    cells[15]["source"] = src15.replace(OLD_ID, NEW_ID).splitlines(keepends=True)

    assert cells[12]["cell_type"] == "markdown"
    cells[12]["source"] = NEW_MD_12.splitlines(keepends=True)

    assert cells[13]["cell_type"] == "code"
    cells[13]["source"] = NEW_CODE_13.splitlines(keepends=True)

    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Swapped Llama model id: {OLD_ID} -> {NEW_ID}")
    print("Restored HF-login markdown (cell 12) and code (cell 13) to reflect gated status.")


if __name__ == "__main__":
    main()
