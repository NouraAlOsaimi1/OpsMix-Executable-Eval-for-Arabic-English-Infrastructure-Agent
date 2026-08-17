"""Insert a 100-task x 4-language validation run into
OpsMix_Ar_Agentic_Trial_EN_MemoryFixed.ipynb, between the 10-task EN-only
pilot (cells 0-38) and the full 500-task x 4-language run (previously
cells 39-43, renumbered here as the full run is pushed later).

Purpose: a stepping stone between the small pilot and the ~day-long full
run -- big enough to see real per-language behavior (100 tasks/language
instead of 5-10) without committing to the full 2000-run evaluation.
Reuses the same stratified (difficulty x primary_tool) sampling as the
pilot's task selection and the same resumable-checkpoint pattern as the
full run, under its own checkpoint directory/variable names so it can't
collide with either.
"""
import json
from pathlib import Path

NB_PATH = Path("OpsMix_Ar_Agentic_Trial_EN_MemoryFixed.ipynb")

MD_INTRO = """---
## السادس عشر: تشغيل تحقّق متوسط الحجم — 100 مهمة × 4 لغات

قبل الالتزام بالتشغيل الكامل (500 مهمة × 4 لغات، تقريبًا يوم كامل من وقت GPU)، شغّل عيّنة أكبر من التجربة الأولية (100 مهمة بدل 10-20) عبر اللغات الأربع. هذا يتأكد إن النتائج مستقرة عبر اللغات وإن الأنابيب تتحمّل حجم أكبر قبل الالتزام بالتشغيل الكامل. نفس checkpointing القابل للاستئناف مستخدم هنا، بمجلد ومتغيرات منفصلة عن قسم التشغيل الكامل عشان ما يتصادمان.

**متطلب:** لازم الخلايا 2 إلى 27 تكون اشتغلت قبل هذا القسم."""

CELL_SAMPLE_SELECT = '''import random
from collections import Counter

random.seed(42)
SAMPLE_SIZE = 100
SAMPLE_LANGUAGES = ["en", "msa", "gulf", "mixed"]
MAX_STEPS = 6

for _name in ("model", "tokenizer", "server", "evaluate_agentic_task", "AGENTIC_SYSTEM_PROMPT"):
    assert _name in dir(), (
        f"'{_name}' is not defined -- run the setup cells above (2 through 27) first."
    )

def _primary_tool(task):
    actions = task.get("gold_actions", []) or []
    if actions and isinstance(actions[0], dict):
        return str(actions[0].get("tool", "unknown")).strip().lower()
    return "unknown"

# نفس أسلوب التقسيم الطبقي (صعوبة × أول أداة gold) المستخدم في عيّنة التجربة
# الأولية (خلية 13)، بس بحجم أكبر -- عشان العيّنة تمثيلية عبر كل الأدوات والصعوبات.
all_normalized_tasks = get_all_tasks()

buckets = {}
for t in all_normalized_tasks:
    key = (str(t.get("difficulty", "unknown")).lower(), _primary_tool(t))
    buckets.setdefault(key, []).append(t)

keys = list(buckets.keys())
random.shuffle(keys)

sample_tasks = []
i = 0
while len(sample_tasks) < SAMPLE_SIZE and keys:
    key = keys[i % len(keys)]
    bucket = buckets[key]
    if bucket:
        sample_tasks.append(bucket.pop(random.randrange(len(bucket))))
    else:
        keys.remove(key)
        continue
    i += 1

total_sample_runs = len(sample_tasks) * len(SAMPLE_LANGUAGES)
print(f"Sample scope: {len(sample_tasks)} tasks x {len(SAMPLE_LANGUAGES)} languages = {total_sample_runs} total runs")
print("Difficulty spread:", Counter(str(t.get('difficulty', '?')).lower() for t in sample_tasks))
print("Primary tool spread:", Counter(_primary_tool(t) for t in sample_tasks))'''

CELL_SAMPLE_LOOP = '''import json
import time as _time
import gc
from pathlib import Path

CHECKPOINT_EVERY = 10  # احفظ التقدم كل 10 مهام منجزة

if IN_COLAB:
    SAMPLE_RUN_DIR = Path(
        "/content/drive/MyDrive/OpsMix-Ar_Qwen3_500/"
        "OpsMix-Ar_Qwen3_500/OpsMix-Ar_Qwen3_500/checkpoints_100_run"
    )
else:
    SAMPLE_RUN_DIR = Path("/workspace/OpsMix-Ar_Qwen3_500/checkpoints_100_run")
SAMPLE_RUN_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_CHECKPOINT_PATH = SAMPLE_RUN_DIR / "sample_100_results.json"


def _load_sample_checkpoint() -> list[dict]:
    if SAMPLE_CHECKPOINT_PATH.exists():
        with open(SAMPLE_CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_sample_checkpoint(results: list[dict]) -> None:
    tmp_path = SAMPLE_CHECKPOINT_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    tmp_path.replace(SAMPLE_CHECKPOINT_PATH)


sample_results = _load_sample_checkpoint()
sample_completed_pairs = {
    (r["task_id"], r["language"])
    for r in sample_results
    if isinstance(r, dict) and "task_id" in r and "language" in r
}
print(f"Resuming from checkpoint: {len(sample_completed_pairs)} runs already completed ({SAMPLE_CHECKPOINT_PATH})")

sample_all_pairs = [(task, lang) for lang in SAMPLE_LANGUAGES for task in sample_tasks]
sample_remaining_pairs = [(t, l) for (t, l) in sample_all_pairs if (t["task_id"], l) not in sample_completed_pairs]
print(f"Remaining: {len(sample_remaining_pairs)} / {len(sample_all_pairs)} runs")

since_last_save = 0

with requests.Session() as session:
    for idx, (task, language) in enumerate(sample_remaining_pairs, start=1):
        t0 = _time.time()
        print(f"[{idx}/{len(sample_remaining_pairs)}] Running {task['task_id']} ({language})...")

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
        sample_results.append(result)
        since_last_save += 1

        gpu_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        print(
            f"    passed={result.get('passed')} | steps={result.get('steps_taken', '?')} | "
            f"stopped={result.get('stopped_reason', '?')} | time={elapsed:.1f}s | GPU={gpu_gb:.2f}GB"
        )

        if since_last_save >= CHECKPOINT_EVERY:
            _save_sample_checkpoint(sample_results)
            since_last_save = 0
            print(f"    [checkpoint saved: {len(sample_results)}/{len(sample_all_pairs)} total runs]")

_save_sample_checkpoint(sample_results)  # حفظ نهائي حتى لو آخر دفعة أصغر من CHECKPOINT_EVERY
print()
print(f"Sample run complete. Total runs recorded: {len(sample_results)} / {len(sample_all_pairs)}.")'''

MD_SAMPLE_SUMMARY = "## السابع عشر: ملخّص عيّنة الـ100 مهمة وتفصيل حسب اللغة"

CELL_SAMPLE_SUMMARY = '''from app.evaluate import summarize_by_difficulty, cross_language_gap

sample_summary = summarize(sample_results)
sample_summary["by_language"] = summarize_by_language(sample_results)
sample_summary["by_difficulty"] = summarize_by_difficulty(sample_results)
sample_summary["cross_language_gap"] = cross_language_gap(sample_summary["by_language"])

with open(SAMPLE_RUN_DIR / "sample_100_summary.json", "w", encoding="utf-8") as f:
    json.dump(sample_summary, f, ensure_ascii=False, indent=2)

print("=" * 60)
print("100-TASK SAMPLE SUMMARY (100 tasks x 4 languages)")
print("=" * 60)
print(f"Total runs:            {sample_summary['total_tasks']}")
print(f"Passed:                {sample_summary['passed_tasks']}")
print(f"Task Success Rate:     {sample_summary['task_success_rate']:.2f}%")
print(f"Safety Violation Rate: {sample_summary['safety_violation_rate']:.2f}%")
print()
for lang, s in sample_summary["by_language"].items():
    print(f"  {lang:6s} | success={s['task_success_rate']:.1f}% | state_match={s['state_match_rate']:.1f}% | "
          f"tool_acc={s['tool_selection_accuracy']:.1f}% | safety_violation={s['safety_violation_rate']:.1f}%")
print()
print("Cross-language gap (task_success_rate):", sample_summary["cross_language_gap"])
print()
print("Saved to:", SAMPLE_RUN_DIR / "sample_100_summary.json")'''


def _md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


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
    cells = nb["cells"]

    insert_at = None
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        if c["cell_type"] == "markdown" and "التشغيل الكامل" in src and "500 مهمة" in src:
            insert_at = i
            break
    assert insert_at is not None, "could not find the full-run intro markdown cell"

    # Renumber the full-run section's ordinals (16th/17th -> 18th/19th) since
    # the new 100-task section takes 16th/17th.
    full_intro = cells[insert_at]
    full_intro_src = "".join(full_intro["source"])
    assert "السادس عشر" in full_intro_src
    full_intro["source"] = full_intro_src.replace("السادس عشر", "الثامن عشر").splitlines(keepends=True)

    full_summary_idx = None
    for i in range(insert_at, len(cells)):
        src = "".join(cells[i].get("source", []))
        if cells[i]["cell_type"] == "markdown" and "الملخّص النهائي" in src:
            full_summary_idx = i
            break
    assert full_summary_idx is not None, "could not find the full-run summary markdown cell"
    full_summary_src = "".join(cells[full_summary_idx]["source"])
    assert "السابع عشر" in full_summary_src
    cells[full_summary_idx]["source"] = full_summary_src.replace("السابع عشر", "التاسع عشر").splitlines(keepends=True)

    new_cells = [
        _md_cell(MD_INTRO),
        _code_cell(CELL_SAMPLE_SELECT),
        _code_cell(CELL_SAMPLE_LOOP),
        _md_cell(MD_SAMPLE_SUMMARY),
        _code_cell(CELL_SAMPLE_SUMMARY),
    ]
    cells[insert_at:insert_at] = new_cells

    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Inserted {len(new_cells)} cells at index {insert_at}. Notebook now has {len(cells)} cells.")


if __name__ == "__main__":
    main()
