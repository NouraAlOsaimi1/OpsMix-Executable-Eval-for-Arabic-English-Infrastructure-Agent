import gc
import json
import random
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from google.colab import drive
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------------------------------------------------------
# 1. Setup & Reproducibility
# ------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

NUM_TASKS = 500
LANGUAGE_FIELDS = {
    "en": "request_en",
    "msa": "request_msa",
    "gulf": "request_gulf",
    "mixed": "request_mixed",
}
LANGUAGES = list(LANGUAGE_FIELDS.keys())

# ------------------------------------------------------------------
# 2. Drive Checkpoint Setup
# ------------------------------------------------------------------
DRIVE_ROOT = Path("/content/drive/MyDrive/OpsMix-Ar_Fanar_500")
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
drive.mount("/content/drive", force_remount=False)

CHECKPOINT_DIR = DRIVE_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
print("Google Drive checkpoint directory:", CHECKPOINT_DIR)


# ------------------------------------------------------------------
# 3. Model & Tokenizer Initialization (Fanar)
# ------------------------------------------------------------------
# Use 'QCRI/Fanar-1-9B-Instruct' for T4
MODEL_NAME = "QCRI/Fanar-1-9B-Instruct"

print(f"Loading Fanar model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # Required for batched generation

print("Fanar loaded successfully!")

# ------------------------------------------------------------------
# 4. Helper Functions
# ------------------------------------------------------------------
SYSTEM_PROMPT = """
You are an infrastructure operations agent.

You receive an operational request and must decide which tools to call.

Available tools:

1. check_disk
   args: {}

2. clear_cache
   args: {}

3. restart_service
   args: {"service": "nginx" | "redis" | "api"}

4. rotate_api_key
   args: {}

5. scale_replicas
   args: {"n": integer}

6. get_metrics
   args: {"service": "nginx" | "redis" | "api"}

7. rollback_deploy
   args: {}

8. get_logs
   args: {"service": "nginx" | "redis" | "api", "limit": optional integer}

9. kill_process
   args: {"pid": integer}

10. set_config
    args: {"key": string, "value": string}

Return ONLY a JSON array of tool calls.

Format:

[
  {
    "tool": "tool_name",
    "args": {}
  }
]

Do not return explanations.
Do not return Markdown.
Do not invent tools.
"""


def extract_tool_calls(response):
    # Remove thinking tags if present (e.g., Fanar native reasoning)
    response = re.sub(
        r"<think>.*?</think>", "", response, flags=re.DOTALL
    ).strip()

    match = re.search(r"\[[\s\S]*\]", response)
    if not match:
        return None

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    if not isinstance(data, list):
        return None

    for call in data:
        if not isinstance(call, dict) or "tool" not in call:
            return None
        if "args" not in call:
            call["args"] = {}

    return data


def _checkpoint_path(language, kind):
    return CHECKPOINT_DIR / f"{kind}_{language}_{NUM_TASKS}.json"


def _local_path(language, kind):
    return Path(f"{kind}_{language}_{NUM_TASKS}.json")


def _atomic_json_save(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tmp.replace(path)


def _save_language_checkpoint(language, predictions, traces):
    _atomic_json_save(predictions, _local_path(language, "predictions"))
    _atomic_json_save(traces, _local_path(language, "traces"))
    _atomic_json_save(predictions, _checkpoint_path(language, "predictions"))
    _atomic_json_save(traces, _checkpoint_path(language, "traces"))


def _load_checkpoint(language):
    local_pred, drive_pred = _local_path(
        language, "predictions"
    ), _checkpoint_path(language, "predictions")
    local_trace, drive_trace = _local_path(language, "traces"), _checkpoint_path(
        language, "traces"
    )

    pred_src = max(
        [p for p in [local_pred, drive_pred] if p.exists()],
        key=lambda x: x.stat().st_mtime,
        default=None,
    )
    trace_src = max(
        [p for p in [local_trace, drive_trace] if p.exists()],
        key=lambda x: x.stat().st_mtime,
        default=None,
    )

    predictions = (
        json.loads(pred_src.read_text(encoding="utf-8")) if pred_src else {}
    )
    traces = (
        json.loads(trace_src.read_text(encoding="utf-8")) if trace_src else {}
    )

    return predictions, traces


# Load Selected Tasks
with open("dataset/selected_500_tasks.json", "r", encoding="utf-8") as f:
    selected_tasks = json.load(f)

# ------------------------------------------------------------------
# 5. Batched Evaluation Loop
# ------------------------------------------------------------------
BATCH_SIZE = 2  # Adjusted batch size for 9B model on T4 GPU
GENERATION_KWARGS = dict(
    max_new_tokens=2048,
    do_sample=True,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
)

for language in LANGUAGES:
    request_field = LANGUAGE_FIELDS[language]
    predictions, traces = _load_checkpoint(language)
    completed_ids = set(predictions.keys()) & set(traces.keys())
    pending_tasks = [
        t for t in selected_tasks if t["task_id"] not in completed_ids
    ]

    print("\n" + "=" * 80)
    print(
        f"RUNNING FANAR | LANGUAGE: {language.upper()} | REMAINING: {len(pending_tasks)}"
    )
    print("=" * 80)

    for batch_start in range(0, len(pending_tasks), BATCH_SIZE):
        batch_tasks = pending_tasks[batch_start : batch_start + BATCH_SIZE]
        batch_messages = []

        for task in batch_tasks:
            request = task[request_field]
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Operational request:\n\n{request}",
                },
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            batch_messages.append(text)

        # Batch Tokenization
        inputs = tokenizer(
            batch_messages, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, **GENERATION_KWARGS)

        # Decode & Parse Outputs
        for idx, task in enumerate(batch_tasks):
            input_len = inputs["input_ids"][idx].shape[0]
            generated_tokens = outputs[idx][input_len:]
            response_text = tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )

            parsed_actions = extract_tool_calls(response_text)

            task_id = task["task_id"]
            predictions[task_id] = parsed_actions if parsed_actions else []
            traces[task_id] = {
                "raw_response": response_text,
                "parsed_actions": parsed_actions,
            }

            print(f"[{task_id}] | parsed={parsed_actions}")

        _save_language_checkpoint(language, predictions, traces)

    gc.collect()
    torch.cuda.empty_cache()

print("\nEvaluation completed successfully for all 4 languages!")