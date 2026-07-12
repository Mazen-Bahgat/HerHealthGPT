# M2-vs-M3 English Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score M2 (Qwen3.5-9B base) vs M3 (Qwen3.5-9B + our English LoRA) on the 540-seed English benchmark via local generation, and emit one comparison table with the M3−M2 delta.

**Architecture:** A new local (no-server) inference runner loads each model via Unsloth and writes predictions in the exact JSONL schema the existing `scripts/evaluate.py` consumes (reusing the prompt + record builder from `scripts/run_inference.py`). `evaluate.py` scores each model; a new `scripts/compare_models.py` aggregates the two summaries into a markdown+JSON comparison.

**Tech Stack:** Python 3.11 (Ubuntu `ft-train-venv`, Unsloth/torch, GPU) for generation; Python 3.12 (Windows `.venv`, CPU) for scoring/comparison + pytest.

## Global Constraints

- **Benchmark:** `HerHealthGPT-LU_seed/seeds_en_v1.jsonl` — 540 rows, gold fields `category`/`gold_risk_level`/`requires_clarification`. Read-only.
- **Reuse (DRY), do not reimplement:** `FIXED_PROMPT_TEMPLATE`, `parse_model_content`, `build_output_record`, `select_input_text` from `scripts/run_inference.py`. Output records MUST be consumable by `scripts/evaluate.py` unchanged.
- **Controlled comparison:** both M2 and M3 load the **same 4-bit base** (`load_in_4bit=True`); M3 adds the LoRA adapter. **Greedy decoding** (`do_sample=False`), **thinking mode OFF** (`enable_thinking=False`), `language="en"`, `max_new_tokens=512`.
- **Base model (M2) local path:** `/home/sw2/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- **M3 adapter:** `models/qwen3.5-9b-herhealth-en-lora`
- **Generation env:** Ubuntu `ft-train-venv`, always with `HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1`.
- **Scoring/compare env:** Windows `.venv` (`.venv/Scripts/python.exe`), CPU.
- **Repo path in WSL:** `/mnt/d/Grad-Project/HerHealthGPT`.
- **Spec:** `docs/superpowers/specs/2026-07-12-m2-m3-english-evaluation-design.md`.

---

## File Structure

- `scripts/run_local_inference.py` — local generation runner (base or base+adapter) → predictions JSONL (Task 1).
- `tests/test_run_local_inference.py` — unit tests for the pure helpers (Task 1).
- `scripts/compare_models.py` — aggregate two `evaluate.py` summaries → comparison (Task 3).
- `tests/test_compare_models.py` — unit tests for the comparison math (Task 3).
- Generated (gitignored via `HerHealthGPT-LU_seed/inference/` already untracked; summaries under `HerHealthGPT-LU_seed/evaluation/`):
  - `HerHealthGPT-LU_seed/inference/M2_en.jsonl`, `M3_en.jsonl` (Task 2)
  - `HerHealthGPT-LU_seed/evaluation/M2_en_summary.json`, `M3_en_summary.json` (Task 3)
  - `HerHealthGPT-LU_seed/evaluation/M2_vs_M3_en.md` + `.json` (Task 3)
- Removed (superseded): `scripts/run_local_raw_baseline.py`, `tests/test_run_local_raw_baseline.py` (Task 1).

---

## Task 1: `run_local_inference.py` — runner + unit tests + GPU smoke

**Files:**
- Create: `scripts/run_local_inference.py`
- Create: `tests/test_run_local_inference.py`
- Remove: `scripts/run_local_raw_baseline.py`, `tests/test_run_local_raw_baseline.py`

**Interfaces:**
- Consumes: `run_inference.{FIXED_PROMPT_TEMPLATE, parse_model_content, build_output_record, select_input_text}`.
- Produces (used by Task 2/3):
  - `iter_benchmark(path: Path) -> list[dict]`
  - `record_for_row(row: dict, raw_response: str, model: str, label: str, row_number: int) -> dict` — a record with the keys `evaluate.py` scores.
  - `done_item_ids(path: Path) -> set[str]`
  - CLI: `python scripts/run_local_inference.py --label M2 --model <path> [--adapter <dir>] --benchmark <jsonl> --output <jsonl> [--limit N] [--max-new-tokens 512]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_run_local_inference.py`:
```python
import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
import run_local_inference as rli  # noqa: E402
import evaluate as ev  # noqa: E402


def _row():
    return {"seed_id": "menst-001", "style": "clinical", "category": "menstrual",
            "gold_risk_level": "see-doctor", "requires_clarification": "no",
            "style_text": "My periods stopped for 4 months and I'm not pregnant."}


VALID_RAW = ('{"predicted_category":"menstrual","interpreted_symptom":"amenorrhea",'
             '"predicted_risk":"see-doctor","recommended_action":"See a clinician",'
             '"asks_clarification":false,"clarifying_question":"",'
             '"unsafe_response":false,"response_text":"Please see a clinician."}')


def test_record_for_row_is_scorable_by_evaluate():
    rec = rli.record_for_row(_row(), VALID_RAW, "test-base", "M2", 1)
    assert rec["item_id"] == "menst-001_clinical_en"
    assert rec["model_label"] == "M2"
    assert rec["gold_category"] == "menstrual"
    assert rec["_parse_error"] == ""
    scored = ev.score_record(rec)
    assert scored["parse_ok"] is True
    assert scored["category_correct"] is True
    assert scored["risk_correct"] is True
    assert scored["clarification_correct"] is True


def test_record_for_row_marks_bad_json_unparsed():
    rec = rli.record_for_row(_row(), "not json at all", "test-base", "M3", 2)
    assert rec["_parse_error"]  # non-empty error kind
    assert ev.score_record(rec)["parse_ok"] is False


def test_done_item_ids_reads_existing_output(tmp_path):
    p = tmp_path / "out.jsonl"
    p.write_text(json.dumps({"item_id": "a_clinical_en"}) + "\n"
                 + json.dumps({"item_id": "b_layperson_en"}) + "\n", encoding="utf-8")
    assert rli.done_item_ids(p) == {"a_clinical_en", "b_layperson_en"}
    assert rli.done_item_ids(tmp_path / "missing.jsonl") == set()
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_run_local_inference.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'run_local_inference'`.

- [ ] **Step 3: Write the runner**

Create `scripts/run_local_inference.py`:
```python
"""Local (no-server) benchmark inference for HerHealthGPT-LU (M2/M3).

Loads a model locally via Unsloth (base, or base+adapter) and generates
structured predictions over the frozen English benchmark, writing JSONL in the
exact schema scripts/evaluate.py consumes. No server; reuses the prompt and
record schema from scripts/run_inference.py (DRY).

Heavy deps (torch/unsloth) are imported lazily inside LocalGenerator so the pure
helpers import cleanly on the Windows .venv for unit testing.

Run inside WSL (Ubuntu, ft-train-venv), offline:
  HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 /home/sw2/ft-train-venv/bin/python \
    scripts/run_local_inference.py --label M2 \
    --model /home/sw2/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a \
    --benchmark HerHealthGPT-LU_seed/seeds_en_v1.jsonl \
    --output HerHealthGPT-LU_seed/inference/M2_en.jsonl
  # M3 adds: --adapter models/qwen3.5-9b-herhealth-en-lora
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_inference as inf  # noqa: E402

MAX_SEQ = 2048
LANGUAGE = "en"


def iter_benchmark(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def done_item_ids(path: Path) -> set[str]:
    path = Path(path)
    if not path.exists():
        return set()
    ids = set()
    for line in path.open(encoding="utf-8"):
        if line.strip():
            ids.add(json.loads(line)["item_id"])
    return ids


def record_for_row(row: dict, raw_response: str, model: str, label: str, row_number: int) -> dict:
    parsed = inf.parse_model_content(raw_response)
    record = inf.build_output_record(row, parsed, raw_response, model, label, LANGUAGE, row_number)
    record["input_text"] = inf.select_input_text(row, None, LANGUAGE)
    return record


class LocalGenerator:
    def __init__(self, model_path: str, adapter: str | None, max_new_tokens: int) -> None:
        from unsloth import FastLanguageModel
        import torch
        self.torch = torch
        load_from = adapter or model_path
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=load_from, max_seq_length=MAX_SEQ, load_in_4bit=True, dtype=None)
        FastLanguageModel.for_inference(self.model)
        self.max_new_tokens = max_new_tokens

    def __call__(self, prompt: str) -> str:
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False,
            return_tensors="pt").to("cuda")
        with self.torch.inference_mode():
            out = self.model.generate(input_ids=inputs, max_new_tokens=self.max_new_tokens,
                                      do_sample=False, use_cache=True)
        return self.tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="M2 / M3 (paper label)")
    ap.add_argument("--model", required=True, help="base model path/id")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (M3)")
    ap.add_argument("--benchmark", type=Path, default=Path("HerHealthGPT-LU_seed/seeds_en_v1.jsonl"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    rows = iter_benchmark(args.benchmark)
    if args.limit:
        rows = rows[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    already = done_item_ids(args.output)
    print(f"{len(rows)} rows -> {args.label} ({args.adapter or args.model}); {len(already)} already done")

    gen = LocalGenerator(args.model, args.adapter, args.max_new_tokens)
    with args.output.open("a", encoding="utf-8") as out:
        for i, row in enumerate(rows, start=1):
            item_id = inf.build_item_id(row, i, LANGUAGE)
            if item_id in already:
                continue
            text = inf.select_input_text(row, None, LANGUAGE)
            try:
                raw = gen(inf.FIXED_PROMPT_TEMPLATE.format(text=text))
            except Exception as exc:  # keep going; mark the row
                raw = ""
                record = inf.build_output_record(
                    row, {"_error": str(exc), "_parse_error": "generation_error"},
                    raw, args.model, args.label, LANGUAGE, i)
            else:
                record = record_for_row(row, raw, args.model, args.label, i)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            if i % 25 == 0:
                print(f"[{i}/{len(rows)}] ...")
    print(f"done -> {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_run_local_inference.py -v
```
Expected: PASS (3 passed).

- [ ] **Step 5: Remove the superseded raw-baseline script**

Run (use `git rm` if tracked; `rm -f` if untracked):
```bash
git rm -f scripts/run_local_raw_baseline.py tests/test_run_local_raw_baseline.py 2>/dev/null || rm -f scripts/run_local_raw_baseline.py tests/test_run_local_raw_baseline.py
```
Then confirm nothing else imports it:
```bash
grep -rn "run_local_raw_baseline" scripts/ tests/ || echo "no references"
```
Expected: `no references`.

- [ ] **Step 6: GPU smoke — 4 items for M2 and M3**

Run:
```bash
wsl.exe -d Ubuntu -- bash -c 'cd /mnt/d/Grad-Project/HerHealthGPT && HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 /home/sw2/ft-train-venv/bin/python scripts/run_local_inference.py --label M2 --model /home/sw2/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a --output HerHealthGPT-LU_seed/inference/_smoke_M2.jsonl --limit 4'
wsl.exe -d Ubuntu -- bash -c 'cd /mnt/d/Grad-Project/HerHealthGPT && HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 /home/sw2/ft-train-venv/bin/python scripts/run_local_inference.py --label M3 --model /home/sw2/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a --adapter models/qwen3.5-9b-herhealth-en-lora --output HerHealthGPT-LU_seed/inference/_smoke_M3.jsonl --limit 4'
```
Expected: each prints `done -> …_smoke_*.jsonl`. Then verify 4 records each parse:
```bash
.venv/Scripts/python.exe -c "import json; [print(r['item_id'], r['_parse_error']) for r in map(json.loads, open('HerHealthGPT-LU_seed/inference/_smoke_M3.jsonl'))]"
```
Expected: 4 lines, `_parse_error` empty for well-formed outputs.

- [ ] **Step 7: Commit**

```bash
git add scripts/run_local_inference.py tests/test_run_local_inference.py
git rm -q scripts/run_local_raw_baseline.py tests/test_run_local_raw_baseline.py 2>/dev/null || true
git commit -m "feat: local benchmark inference runner (M2/M3); supersede raw baseline"
```

---

## Task 2: Full 540-seed inference for M2 and M3

**Files:**
- Generates: `HerHealthGPT-LU_seed/inference/M2_en.jsonl`, `M3_en.jsonl`

**Interfaces:**
- Consumes: `scripts/run_local_inference.py` (Task 1).
- Produces (used by Task 3): two 540-line prediction JSONLs.

- [ ] **Step 1: Run M2 (base) over all 540 seeds**

Run:
```bash
wsl.exe -d Ubuntu -- bash -c 'cd /mnt/d/Grad-Project/HerHealthGPT && HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 /home/sw2/ft-train-venv/bin/python scripts/run_local_inference.py --label M2 --model /home/sw2/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a --output HerHealthGPT-LU_seed/inference/M2_en.jsonl'
```
Expected: ends `done -> HerHealthGPT-LU_seed/inference/M2_en.jsonl`. (Resumable — re-run if interrupted; it skips done items.)

- [ ] **Step 2: Run M3 (base + adapter) over all 540 seeds**

Run:
```bash
wsl.exe -d Ubuntu -- bash -c 'cd /mnt/d/Grad-Project/HerHealthGPT && HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 /home/sw2/ft-train-venv/bin/python scripts/run_local_inference.py --label M3 --model /home/sw2/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a --adapter models/qwen3.5-9b-herhealth-en-lora --output HerHealthGPT-LU_seed/inference/M3_en.jsonl'
```
Expected: ends `done -> HerHealthGPT-LU_seed/inference/M3_en.jsonl`.

- [ ] **Step 3: Verify both files have 540 records**

Run:
```bash
.venv/Scripts/python.exe -c "import sys; [print(f, sum(1 for _ in open(f, encoding='utf-8'))) for f in ['HerHealthGPT-LU_seed/inference/M2_en.jsonl','HerHealthGPT-LU_seed/inference/M3_en.jsonl']]"
```
Expected: both report `540`.

(No commit — outputs live under `HerHealthGPT-LU_seed/inference/`, which is untracked scratch. If the team wants them versioned, that is a separate decision.)

---

## Task 3: `compare_models.py` — score + compare + report

**Files:**
- Create: `scripts/compare_models.py`
- Create: `tests/test_compare_models.py`
- Generates: `evaluation/M2_en_summary.json`, `M3_en_summary.json`, `evaluation/M2_vs_M3_en.md` + `.json`

**Interfaces:**
- Consumes: two `evaluate.py` summary JSONs (shape: `{"overall": {...}, "by_style": {k: {...}}, "by_gold_category": {k: {...}}, ...}` where each `{...}` has metric keys `parse_ok_rate`, `prediction_coverage`, `category_accuracy`, `risk_accuracy`, `clarification_accuracy`, `self_reported_unsafe_rate`).
- Produces:
  - `compare_summaries(baseline: dict, treatment: dict) -> dict`
  - `render_markdown(comparison: dict, baseline_label: str, treatment_label: str) -> str`

- [ ] **Step 1: Write the failing test**

Create `tests/test_compare_models.py`:
```python
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
import compare_models as cm  # noqa: E402


def _group(cat=0.5, risk=0.4, clar=0.6):
    return {"parse_ok_rate": 1.0, "prediction_coverage": 1.0,
            "category_accuracy": cat, "risk_accuracy": risk,
            "clarification_accuracy": clar, "self_reported_unsafe_rate": 0.1}


def _summary(cat, risk):
    return {"overall": _group(cat, risk),
            "by_style": {"clinical": _group(cat, risk)},
            "by_gold_category": {"menstrual": _group(cat, risk)}}


def test_compare_computes_delta():
    comp = cm.compare_summaries(_summary(0.5, 0.4), _summary(0.7, 0.6))
    assert comp["overall"]["category_accuracy"]["baseline"] == 0.5
    assert comp["overall"]["category_accuracy"]["treatment"] == 0.7
    assert round(comp["overall"]["category_accuracy"]["delta"], 3) == 0.2
    assert "clinical" in comp["by_style"]
    assert "menstrual" in comp["by_gold_category"]


def test_render_markdown_has_delta_and_labels():
    comp = cm.compare_summaries(_summary(0.5, 0.4), _summary(0.7, 0.6))
    md = cm.render_markdown(comp, "M2", "M3")
    assert "M2" in md and "M3" in md
    assert "category_accuracy" in md
    assert "+0.200" in md or "0.200" in md  # delta rendered
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_compare_models.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'compare_models'`.

- [ ] **Step 3: Write the comparison script**

Create `scripts/compare_models.py`:
```python
"""Compare two evaluate.py summaries (baseline vs treatment) into one table.

Emits per-metric baseline/treatment/delta at three levels: overall, per-style,
per-gold-category. Pure over the summary dicts; CPU-only.

Run:
  python scripts/compare_models.py \
    --baseline HerHealthGPT-LU_seed/evaluation/M2_en_summary.json --baseline-label M2 \
    --treatment HerHealthGPT-LU_seed/evaluation/M3_en_summary.json --treatment-label M3 \
    --out-md HerHealthGPT-LU_seed/evaluation/M2_vs_M3_en.md \
    --out-json HerHealthGPT-LU_seed/evaluation/M2_vs_M3_en.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = ["parse_ok_rate", "prediction_coverage", "category_accuracy",
           "risk_accuracy", "clarification_accuracy", "self_reported_unsafe_rate"]
LEVELS = ["by_style", "by_gold_category"]


def _cell(base_group: dict, treat_group: dict, metric: str) -> dict:
    b = base_group.get(metric)
    t = treat_group.get(metric)
    delta = (t - b) if isinstance(b, (int, float)) and isinstance(t, (int, float)) else None
    return {"baseline": b, "treatment": t, "delta": delta}


def _row(base_group: dict, treat_group: dict) -> dict:
    return {m: _cell(base_group, treat_group, m) for m in METRICS}


def compare_summaries(baseline: dict, treatment: dict) -> dict:
    out = {"overall": _row(baseline.get("overall", {}), treatment.get("overall", {}))}
    for level in LEVELS:
        out[level] = {}
        keys = sorted(set(baseline.get(level, {})) | set(treatment.get(level, {})))
        for k in keys:
            out[level][k] = _row(baseline.get(level, {}).get(k, {}),
                                  treatment.get(level, {}).get(k, {}))
    return out


def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "-"


def _fmt_delta(v) -> str:
    return f"{v:+.3f}" if isinstance(v, (int, float)) else "-"


def _table(title: str, rows: dict, bl: str, tl: str) -> str:
    lines = [f"### {title}", "", f"| metric | {bl} | {tl} | Δ({tl}−{bl}) |", "|---|---|---|---|"]
    for metric, cell in rows.items():
        lines.append(f"| {metric} | {_fmt(cell['baseline'])} | {_fmt(cell['treatment'])} | {_fmt_delta(cell['delta'])} |")
    return "\n".join(lines) + "\n"


def render_markdown(comparison: dict, baseline_label: str, treatment_label: str) -> str:
    out = [f"# {treatment_label} vs {baseline_label} — English benchmark\n"]
    out.append(_table("Overall", comparison["overall"], baseline_label, treatment_label))
    for level, heading in (("by_style", "By style"), ("by_gold_category", "By category")):
        for key, rows in comparison.get(level, {}).items():
            out.append(_table(f"{heading}: {key}", rows, baseline_label, treatment_label))
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--treatment", type=Path, required=True)
    ap.add_argument("--baseline-label", default="M2")
    ap.add_argument("--treatment-label", default="M3")
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    treatment = json.loads(args.treatment.read_text(encoding="utf-8"))
    comparison = compare_summaries(baseline, treatment)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(comparison, args.baseline_label, args.treatment_label), encoding="utf-8")
    args.out_json.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"wrote {args.out_md} and {args.out_json}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_compare_models.py -v
```
Expected: PASS (2 passed).

- [ ] **Step 5: Score M2 and M3 with the existing evaluate.py**

Run:
```bash
.venv/Scripts/python.exe scripts/evaluate.py --predictions HerHealthGPT-LU_seed/inference/M2_en.jsonl --summary HerHealthGPT-LU_seed/evaluation/M2_en_summary.json --expected-count 540
.venv/Scripts/python.exe scripts/evaluate.py --predictions HerHealthGPT-LU_seed/inference/M3_en.jsonl --summary HerHealthGPT-LU_seed/evaluation/M3_en_summary.json --expected-count 540
```
Expected: each prints its overall metrics line and writes the summary JSON.

- [ ] **Step 6: Build the comparison and eyeball it**

Run:
```bash
.venv/Scripts/python.exe scripts/compare_models.py --baseline HerHealthGPT-LU_seed/evaluation/M2_en_summary.json --baseline-label M2 --treatment HerHealthGPT-LU_seed/evaluation/M3_en_summary.json --treatment-label M3 --out-md HerHealthGPT-LU_seed/evaluation/M2_vs_M3_en.md --out-json HerHealthGPT-LU_seed/evaluation/M2_vs_M3_en.json
cat HerHealthGPT-LU_seed/evaluation/M2_vs_M3_en.md
```
Expected: an Overall table plus per-style/per-category tables, each with M2, M3, and Δ columns. Both models should show high `parse_ok_rate`; the Δ on `risk_accuracy`/`category_accuracy` is the headline finding (positive Δ = fine-tuning helped; a negative or zero Δ is still a valid, reportable result).

- [ ] **Step 7: Commit**

```bash
git add scripts/compare_models.py tests/test_compare_models.py
git commit -m "feat: model comparison (M2 vs M3) from evaluate.py summaries"
```

---

## Self-Review

**Spec coverage:** §5a runner → Task 1; §5b evaluate.py (unchanged, run) → Task 3 Step 5; §5c compare_models → Task 3; §2 inputs (benchmark, base path, adapter) → Global Constraints; §3 decisions (local gen, M2-vs-M3, greedy, thinking-off, 4-bit) → Global Constraints + Task 1 code; §6 metrics → reused from evaluate.py (Task 3 consumes them); §7 envs → Global Constraints + task commands; §8 testing (unit/smoke/verify) → Task 1 Steps 1-6, Task 3 Steps 1-6; §1 "supersede run_local_raw_baseline" → Task 1 Step 5; deferred items (M1/M4, LLM-judge, significance) → not in plan, matching spec. No gaps.

**Placeholder scan:** No TBD/TODO. The `except Exception` in the runner is a deliberate, spec-required resumable-error behavior (records the failure as `generation_error`), not a vague "handle errors" — the record shape is shown. All code steps show complete code.

**Type consistency:** `record_for_row(row, raw_response, model, label, row_number)` signature matches between the test (Task 1 Step 1) and the runner (Step 3). `iter_benchmark`/`done_item_ids` names match test and impl. `compare_summaries(baseline, treatment)` and `render_markdown(comparison, baseline_label, treatment_label)` match between the test (Task 3 Step 1) and impl (Step 3). The summary shape consumed by `compare_summaries` (`overall`/`by_style`/`by_gold_category` with the six metric keys) matches `evaluate.py`'s `summarize()` output verified in the codebase. `build_output_record`'s 7-arg signature matches the call in `record_for_row`.
