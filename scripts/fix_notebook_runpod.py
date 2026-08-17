"""Make OpsMix_Ar_Agentic_Trial_EN_MemoryFixed.ipynb environment-aware so it
runs on RunPod (or any non-Colab Jupyter env) as well as Colab, from the
same file.

Only two cells were Colab-specific:
* cell 7 unconditionally imported google.colab and mounted Drive -- crashes
  outside Colab.
* cell 35 saved results under /content/drive/MyDrive/... -- that path only
  exists in Colab.

Both now branch on an IN_COLAB flag (set once in cell 7 by trying to import
google.colab) and fall back to a /workspace/... path otherwise, matching
RunPod's persistent-volume convention. Cell 8's git clone now targets an
explicit REPO_DIR built the same way, instead of relying on the notebook's
starting working directory being /content (true in Colab, not guaranteed
elsewhere).
"""
import json
from pathlib import Path

NB_PATH = Path("OpsMix_Ar_Agentic_Trial_EN_MemoryFixed.ipynb")

CELL7_OLD = '''from google.colab import drive
drive.mount("/content/drive")'''

CELL7_NEW = '''# Detect Colab vs. any other Jupyter environment (e.g. a RunPod pod) so
# this notebook can save/load from the right place either way.
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    print("Running in Colab -- Google Drive mounted.")
else:
    print(
        "Not running in Colab (e.g. RunPod) -- skipping Drive mount. "
        "Results will be saved under /workspace instead (see the save-results cell)."
    )'''

CELL8_OLD = '''import os

if not os.path.exists("/content/OpsMix-Ar-Executable-Evaluation-of-Arabic-English-Infrastructure-Agents"):
    !git clone https://github.com/MadaweeAlabdulkreem/OpsMix-Ar-Executable-Evaluation-of-Arabic-English-Infrastructure-Agents.git

os.chdir("/content/OpsMix-Ar-Executable-Evaluation-of-Arabic-English-Infrastructure-Agents")
print("cwd:", os.getcwd())
!ls'''

CELL8_NEW = '''import os

REPO_NAME = "OpsMix-Ar-Executable-Evaluation-of-Arabic-English-Infrastructure-Agents"
# /content is Colab's convention; /workspace is RunPod's (usually the
# persistent-volume mount point, if one is attached to the pod).
REPO_PARENT = "/content" if IN_COLAB else "/workspace"
REPO_DIR = f"{REPO_PARENT}/{REPO_NAME}"

os.makedirs(REPO_PARENT, exist_ok=True)

if not os.path.exists(REPO_DIR):
    !git clone https://github.com/MadaweeAlabdulkreem/OpsMix-Ar-Executable-Evaluation-of-Arabic-English-Infrastructure-Agents.git {REPO_DIR}

os.chdir(REPO_DIR)
print("cwd:", os.getcwd())
!ls'''

CELL35_OLD = '''from pathlib import Path
import json

TRIAL_DIR = Path(
    "/content/drive/MyDrive/OpsMix-Ar_Qwen3_500/"
    "OpsMix-Ar_Qwen3_500/OpsMix-Ar_Qwen3_500/checkpoints_agentic_trial"
)
TRIAL_DIR.mkdir(parents=True, exist_ok=True)'''

CELL35_NEW = '''from pathlib import Path
import json

if IN_COLAB:
    TRIAL_DIR = Path(
        "/content/drive/MyDrive/OpsMix-Ar_Qwen3_500/"
        "OpsMix-Ar_Qwen3_500/OpsMix-Ar_Qwen3_500/checkpoints_agentic_trial"
    )
else:
    # RunPod convention: /workspace is normally the persistent-volume mount,
    # so results survive a pod stop/restart if a volume is attached. If you
    # didn't attach one, this still writes locally -- copy it out before
    # terminating the pod.
    TRIAL_DIR = Path("/workspace/OpsMix-Ar_Qwen3_500/checkpoints_agentic_trial")
TRIAL_DIR.mkdir(parents=True, exist_ok=True)'''


def _patch(nb: dict, index: int, old: str, new: str, label: str) -> None:
    cell = nb["cells"][index]
    src = "".join(cell["source"])
    assert old in src, f"{label}: expected snippet not found in cell {index}"
    assert src.count(old) == 1, f"{label}: snippet not unique in cell {index}"
    src = src.replace(old, new)
    cell["source"] = src.splitlines(keepends=True)


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    _patch(nb, 7, CELL7_OLD, CELL7_NEW, "cell 7 (drive mount)")
    _patch(nb, 8, CELL8_OLD, CELL8_NEW, "cell 8 (git clone)")
    _patch(nb, 35, CELL35_OLD, CELL35_NEW, "cell 35 (save results)")
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Patched cells 7, 8, 35 for Colab/RunPod portability.")


if __name__ == "__main__":
    main()
