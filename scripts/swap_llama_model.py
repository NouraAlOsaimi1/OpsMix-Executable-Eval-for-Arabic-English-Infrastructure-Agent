"""Switch the Llama entry in OpsMix_Ar_Agentic_Trial_4Models_2000Tasks.ipynb
from the gated meta-llama/Llama-3.1-8B-Instruct to the ungated
NousResearch/Meta-Llama-3.1-8B-Instruct mirror (same weights, no license
gate), and update the HF-login step's wording to match -- it's no longer
required for Llama specifically, just optional (helps with rate limits).
"""
import json
from pathlib import Path

NB_PATH = Path("OpsMix_Ar_Agentic_Trial_4Models_2000Tasks.ipynb")

OLD_ID = "meta-llama/Llama-3.1-8B-Instruct"
NEW_ID = "NousResearch/Meta-Llama-3.1-8B-Instruct"

NEW_MD_12 = (
    "## Step 3: Hugging Face login (optional)\n"
    "\n"
    f"`{NEW_ID}` is an ungated mirror of the same weights (no license "
    "acceptance needed), so this step is optional for this notebook as "
    "written. Still fine to run if you have a token -- logging in can help "
    "with Hugging Face Hub rate limits when downloading several multi-GB "
    "models back to back."
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
    '        "No HF_TOKEN set -- proceeding without login. All four configured models "\n'
    '        "are ungated, so this is fine; a token just helps avoid Hub rate limits."\n'
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
    print("Updated HF-login markdown (cell 12) and code (cell 13) to reflect ungated status.")


if __name__ == "__main__":
    main()
