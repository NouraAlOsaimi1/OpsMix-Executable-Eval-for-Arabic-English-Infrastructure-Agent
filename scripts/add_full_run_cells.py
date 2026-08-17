"""Append cells implementing the full 500-task x 4-language run to
OpsMix_Ar_Agentic_Trial_EN_MemoryFixed.ipynb, with resumable checkpointing.

Fixes two blocking gaps found in the pre-flight audit:
1. The notebook only ever ran the trial_tasks pilot sample, EN-only
   (cell 29 hardcodes LANGUAGES = ["en"]) -- there was no code path that
   actually covers all 500 tasks x 4 languages.
2. No checkpointing existed. At the pilot's observed ~46-48s/task, a
   2000-run full evaluation is on the order of a day of unattended GPU
   time; without incremental saves, one crash/disconnect/preemption
   anywhere in that window loses everything. Checkpointing saves after
   every CHECKPOINT_EVERY completed runs and resumes by skipping
   (task_id, language) pairs already present in the checkpoint file, so a
   restarted run picks up where it left off instead of starting over.

The original pilot cells (0-38) are left untouched as a reusable smoke
test; this appends new cells after them rather than replacing anything.
"""
import json
from pathlib import Path

NB_PATH = Path("OpsMix_Ar_Agentic_Trial_EN_MemoryFixed.ipynb")

MD_INTRO = """---
## السادس عشر: التشغيل الكامل — 500 مهمة × 4 لغات (Checkpointing قابل للاستئناف)

يشغّل هذا القسم **كل** مهام dataset.json عبر اللغات الأربع (en / msa / gulf / mixed)، ويحفظ التقدم كل عدد محدد من المهام. لو انقطعت الجلسة (تعطل، انقطاع اتصال، إعادة تشغيل)، إعادة تشغيل هذا القسم فقط يكمل من حيث وقف بدل ما يبدأ من الصفر.

**متطلب:** لازم الخلايا 2 إلى 27 تكون اشتغلت قبل هذا القسم (تحميل النموذج، تشغيل الـ sandbox، تعريف `run_agentic_task` / `evaluate_agentic_task`). خلايا التجربة الأولية (12-38) اختيارية -- مو لازم تشتغل قبل هذا القسم."""

CELL_SETUP = '''# التوسيع الكامل: كل الـ 500 مهمة عبر الأربع لغات، بدل عيّنة trial_tasks المحدودة
FULL_LANGUAGES = ["en", "msa", "gulf", "mixed"]
MAX_STEPS = 6

for _name in ("model", "tokenizer", "server", "evaluate_agentic_task", "AGENTIC_SYSTEM_PROMPT"):
    assert _name in dir(), (
        f"'{_name}' is not defined -- run the setup cells above (2 through 27) first."
    )

full_tasks = get_all_tasks()
total_full_runs = len(full_tasks) * len(FULL_LANGUAGES)
print(f"Full run scope: {len(full_tasks)} tasks x {len(FULL_LANGUAGES)} languages = {total_full_runs} total runs")'''

CELL_LOOP = '''import json
import time as _time
import gc
from pathlib import Path

CHECKPOINT_EVERY = 10  # احفظ التقدم كل 10 مهام منجزة

if IN_COLAB:
    FULL_RUN_DIR = Path(
        "/content/drive/MyDrive/OpsMix-Ar_Qwen3_500/"
        "OpsMix-Ar_Qwen3_500/OpsMix-Ar_Qwen3_500/checkpoints_full_run"
    )
else:
    # RunPod: /workspace عادة نقطة تركيب الـ persistent volume إذا مربوط بالـ pod
    FULL_RUN_DIR = Path("/workspace/OpsMix-Ar_Qwen3_500/checkpoints_full_run")
FULL_RUN_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = FULL_RUN_DIR / "full_run_results.json"


def _load_checkpoint() -> list[dict]:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_checkpoint(results: list[dict]) -> None:
    # كتابة atomic: نكتب لملف مؤقت ثم نستبدل -- يمنع تلف الملف لو انقطعت الجلسة أثناء الكتابة
    tmp_path = CHECKPOINT_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    tmp_path.replace(CHECKPOINT_PATH)


full_results = _load_checkpoint()
completed_pairs = {
    (r["task_id"], r["language"])
    for r in full_results
    if isinstance(r, dict) and "task_id" in r and "language" in r
}
print(f"Resuming from checkpoint: {len(completed_pairs)} runs already completed ({CHECKPOINT_PATH})")

all_pairs = [(task, lang) for lang in FULL_LANGUAGES for task in full_tasks]
remaining_pairs = [(t, l) for (t, l) in all_pairs if (t["task_id"], l) not in completed_pairs]
print(f"Remaining: {len(remaining_pairs)} / {len(all_pairs)} runs")

since_last_save = 0

with requests.Session() as session:
    for idx, (task, language) in enumerate(remaining_pairs, start=1):
        t0 = _time.time()
        print(f"[{idx}/{len(remaining_pairs)}] Running {task['task_id']} ({language})...")

        try:
            result = evaluate_agentic_task(
                model=model,
                tokenizer=tokenizer,
                task=task,
                language=language,
                session=session,
                base_url="http://127.0.0.1:8000",
                max_steps=MAX_STEPS,
            )
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            result = {
                "task_id": task["task_id"],
                "language": language,
                "passed": False,
                "execution_errors": [f"{type(exc).__name__}: {exc}"],
            }
            print(f"    !! EXCEPTION: {type(exc).__name__}: {exc}")
            print(tb[-1500:])
            gc.collect()
            torch.cuda.empty_cache()

        gc.collect()
        torch.cuda.empty_cache()

        elapsed = _time.time() - t0
        result["elapsed_seconds"] = round(elapsed, 1)
        full_results.append(result)
        since_last_save += 1

        gpu_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        print(
            f"    passed={result.get('passed')} | steps={result.get('steps_taken', '?')} | "
            f"stopped={result.get('stopped_reason', '?')} | time={elapsed:.1f}s | GPU={gpu_gb:.2f}GB"
        )

        if since_last_save >= CHECKPOINT_EVERY:
            _save_checkpoint(full_results)
            since_last_save = 0
            print(f"    [checkpoint saved: {len(full_results)}/{len(all_pairs)} total runs]")

_save_checkpoint(full_results)  # حفظ نهائي حتى لو آخر دفعة أصغر من CHECKPOINT_EVERY
print()
print(f"Full run complete. Total runs recorded: {len(full_results)} / {len(all_pairs)}.")'''

MD_SUMMARY = "## السابع عشر: الملخّص النهائي (500 مهمة × 4 لغات) وتفصيل حسب اللغة"

CELL_SUMMARY = '''from app.evaluate import summarize_by_difficulty, cross_language_gap

final_summary = summarize(full_results)
final_summary["by_language"] = summarize_by_language(full_results)
final_summary["by_difficulty"] = summarize_by_difficulty(full_results)
final_summary["cross_language_gap"] = cross_language_gap(final_summary["by_language"])

with open(FULL_RUN_DIR / "full_run_summary.json", "w", encoding="utf-8") as f:
    json.dump(final_summary, f, ensure_ascii=False, indent=2)

print("=" * 60)
print("FULL RUN SUMMARY (500 tasks x 4 languages)")
print("=" * 60)
print(f"Total runs:            {final_summary['total_tasks']}")
print(f"Passed:                {final_summary['passed_tasks']}")
print(f"Task Success Rate:     {final_summary['task_success_rate']:.2f}%")
print(f"Safety Violation Rate: {final_summary['safety_violation_rate']:.2f}%")
print()
for lang, s in final_summary["by_language"].items():
    print(f"  {lang:6s} | success={s['task_success_rate']:.1f}% | state_match={s['state_match_rate']:.1f}% | "
          f"tool_acc={s['tool_selection_accuracy']:.1f}% | safety_violation={s['safety_violation_rate']:.1f}%")
print()
print("Cross-language gap (task_success_rate):", final_summary["cross_language_gap"])
print()
print("Saved to:", FULL_RUN_DIR / "full_run_summary.json")'''


def _md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def _code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    assert nb["cells"][38]["cell_type"] == "markdown", "cell 38 is expected to be the closing markdown note"

    new_cells = [
        _md_cell(MD_INTRO),
        _code_cell(CELL_SETUP),
        _code_cell(CELL_LOOP),
        _md_cell(MD_SUMMARY),
        _code_cell(CELL_SUMMARY),
    ]
    nb["cells"].extend(new_cells)

    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Appended {len(new_cells)} cells (39-43). Notebook now has {len(nb['cells'])} cells.")


if __name__ == "__main__":
    main()
