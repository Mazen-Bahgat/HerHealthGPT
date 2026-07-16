# New-Benchmark Metrics + EN / EN+FR+AR Fine-Tunes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce paper-ready zero-shot metrics on the team-revised benchmark, then QLoRA fine-tune Qwen3.5-9B on the new canonical QA splits (EN first, then EN+FR+AR).

**Architecture:** Stage 1 re-scores the existing M2 raw responses against the new gold labels (no GPU) by parameterizing the benchmark converter and adding a gold-patch script. Stages 2–3 build a leakage-cleaned chat-JSONL prep script (with a translated-CSV ingest path) and translation handoff templates. Stages 4–5 are GPU runs using the existing `train_qlora.py` / `run_local_inference.py` / `evaluate.py` / `safety_metrics.py` unchanged.

**Tech Stack:** Python 3.12 (Windows `.venv`) for data work + pytest; WSL `Ubuntu` / `/home/sw2/ft-train-venv` (Unsloth QLoRA) for GPU stages.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-15-new-benchmark-metrics-and-multilingual-ft-design.md`.
- **STAGE GATES:** after each stage (1–5), STOP, report results to Mazen, and get explicit approval before starting the next stage.
- New benchmark source of truth: `Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/gold_seeds_styled_labeled.csv` (540 rows; columns `Question,Answer,Topic,Keywords,Style,gold_condition,gold_risk_level,gold_action,requires_clarification`).
- FT hyperparameters: EN-only run = 3 epochs, lr 2e-4, batch 2, grad-accum 8; joint run = 2 epochs, same otherwise. Smoke gate (`--max-steps 5`) before every full GPU run.
- All CSV reads in new/modified code use `encoding="utf-8-sig"` (some source files carry a BOM).
- Windows test command prefix: `.venv/Scripts/python -m pytest`.
- Fail loudly: converters/patchers/ingesters exit non-zero with a message on any count or join mismatch — never silently skip rows.
- Existing test suite must stay green: `.venv/Scripts/python -m pytest` at every commit.
- Commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Parameterize the benchmark converter and carry `gold_action`

**Files:**
- Modify: `scripts/convert_gold_seeds_styled.py`
- Test: `tests/test_convert_gold_seeds_styled.py` (new)

**Interfaces:**
- Consumes: nothing new.
- Produces: CLI `python scripts/convert_gold_seeds_styled.py [--src PATH] [--out PATH]` (defaults = old Train_Val paths, so existing behavior is unchanged). Emitted JSONL rows gain a `"gold_action"` key (empty string when the source column is absent). Also exposes `convert(src: Path, out: Path) -> int` (returns row count) for tests and for Task 2's fixture building.

- [ ] **Step 1: Write the failing test**

Create `tests/test_convert_gold_seeds_styled.py`:

```python
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import convert_gold_seeds_styled as cgs

HEADER_NEW = ["Question", "Answer", "Topic", "Keywords", "Style",
              "gold_condition", "gold_risk_level", "gold_action",
              "requires_clarification"]
HEADER_OLD = ["Question", "Answer", "Topic", "Keywords", "Style",
              "gold_risk_level", "requires_clarification", "gold_condition"]


def _write_csv(path, header, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


def _row(header, q, answer, style, action=""):
    base = {"Question": q, "Answer": answer, "Topic": "PCOS",
            "Keywords": "k", "Style": style, "gold_condition": "PCOS",
            "gold_risk_level": "see-doctor", "requires_clarification": "no"}
    if "gold_action" in header:
        base["gold_action"] = action
    return {k: base[k] for k in header}


def test_convert_new_layout_carries_gold_action(tmp_path):
    src = tmp_path / "labeled.csv"
    out = tmp_path / "bench.jsonl"
    _write_csv(src, HEADER_NEW, [
        _row(HEADER_NEW, "Q1 canonical?", "Answer A", "canonical", "see a gp"),
        _row(HEADER_NEW, "Q1 clinical?", "Answer A", "clinical", "see a gp"),
    ])
    n = cgs.convert(src, out)
    assert n == 2
    rows = [json.loads(l) for l in out.open(encoding="utf-8")]
    assert all(r["gold_action"] == "see a gp" for r in rows)
    assert rows[0]["seed_id"] == "gss-000"
    assert {r["style"] for r in rows} == {"canonical", "clinical"}


def test_convert_old_layout_defaults_gold_action_empty(tmp_path):
    src = tmp_path / "labeled.csv"
    out = tmp_path / "bench.jsonl"
    _write_csv(src, HEADER_OLD, [
        _row(HEADER_OLD, "Q1 canonical?", "Answer A", "canonical"),
    ])
    cgs.convert(src, out)
    rows = [json.loads(l) for l in out.open(encoding="utf-8")]
    assert rows[0]["gold_action"] == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_convert_gold_seeds_styled.py -v`
Expected: FAIL with `AttributeError: module 'convert_gold_seeds_styled' has no attribute 'convert'`

- [ ] **Step 3: Refactor the converter**

In `scripts/convert_gold_seeds_styled.py`, replace `main()` with a `convert(src, out)` function plus an argparse `main()`. Keep the module docstring; change the body below the imports to:

```python
SRC = Path("Used_Datasets/Consolidated_Datasets/Train_Val_Dataset/gold_seeds_styled_labeled.csv")
OUT = Path("Used_Datasets/Consolidated_Datasets/Train_Val_Dataset/gold_seeds_styled.jsonl")


def convert(src: Path, out: Path) -> int:
    rows = list(csv.DictReader(src.open(encoding="utf-8-sig")))
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["Answer"]].append(r)

    records = []
    for i, (answer, group) in enumerate(sorted(groups.items())):
        seed_id = f"gss-{i:03d}"
        for r in group:
            records.append({
                "seed_id": seed_id,
                "style": r["Style"].strip().lower(),
                "category": inf.normalize_category(r["Topic"]),
                "topic_raw": r["Topic"],
                "style_text": r["Question"].strip(),
                "gold_answer": r["Answer"].strip(),
                "keywords": r["Keywords"],
                "gold_risk_level": r["gold_risk_level"],
                "requires_clarification": r["requires_clarification"],
                "gold_condition": r["gold_condition"],
                "gold_action": r.get("gold_action", ""),
            })

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(records)} rows -> {out} ({len(groups)} seed groups)")
    return len(records)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    convert(args.src, args.out)


if __name__ == "__main__":
    main()
```

Add `import argparse` to the imports.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_convert_gold_seeds_styled.py -v`
Expected: 2 PASS

- [ ] **Step 5: Run the full suite, then commit**

Run: `.venv/Scripts/python -m pytest`
Expected: all green.

```bash
git add scripts/convert_gold_seeds_styled.py tests/test_convert_gold_seeds_styled.py
git commit -m "feat: parameterize benchmark converter (--src/--out) and carry gold_action

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Gold-patch script for existing predictions

**Files:**
- Create: `scripts/patch_predictions_gold.py`
- Test: `tests/test_patch_predictions_gold.py`

**Interfaces:**
- Consumes: benchmark JSONL rows from Task 1's converter (keys `seed_id`, `style`, `style_text`, `category`, `gold_risk_level`, `gold_action`, `gold_condition`, `requires_clarification`); prediction records in the `run_inference.build_output_record` schema (join key `item_id` = `{seed_id}_{style}_{language}`).
- Produces: CLI `python scripts/patch_predictions_gold.py --benchmark B.jsonl --predictions P.jsonl --output OUT.jsonl`. Exposes `patch_records(preds: list[dict], bench_rows: list[dict], language: str = "en") -> list[dict]` which returns new records with the five gold fields overwritten and everything else (raw_response, parsed prediction fields) byte-identical. Raises `ValueError` on: any prediction `item_id` missing from the benchmark, any benchmark item missing from predictions, or `input_text != style_text` for a joined pair.

- [ ] **Step 1: Write the failing test**

Create `tests/test_patch_predictions_gold.py`:

```python
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import patch_predictions_gold as ppg


def _bench_row(seed_id="gss-000", style="canonical", q="How is PCOS diagnosed?"):
    return {"seed_id": seed_id, "style": style, "style_text": q,
            "category": "pcos", "gold_risk_level": "routine",
            "gold_action": "self-care ok", "gold_condition": "PCOS",
            "requires_clarification": "no"}


def _pred(seed_id="gss-000", style="canonical", q="How is PCOS diagnosed?"):
    return {"item_id": f"{seed_id}_{style}_en", "seed_id": seed_id,
            "style": style, "language": "en", "input_text": q,
            "gold_category": "pcos", "gold_risk_level": "see-doctor",
            "gold_action": "", "gold_condition": "old", "requires_clarification": "yes",
            "raw_response": "{...}", "predicted_risk": "routine"}


def test_patch_overwrites_gold_preserves_predictions():
    out = ppg.patch_records([_pred()], [_bench_row()])
    assert out[0]["gold_risk_level"] == "routine"
    assert out[0]["gold_action"] == "self-care ok"
    assert out[0]["gold_condition"] == "PCOS"
    assert out[0]["requires_clarification"] == "no"
    assert out[0]["raw_response"] == "{...}"
    assert out[0]["predicted_risk"] == "routine"


def test_patch_fails_on_unknown_item_id():
    with pytest.raises(ValueError, match="gss-999"):
        ppg.patch_records([_pred(seed_id="gss-999")], [_bench_row()])


def test_patch_fails_on_missing_prediction():
    bench = [_bench_row(), _bench_row(style="clinical", q="Clinical Q?")]
    with pytest.raises(ValueError, match="clinical"):
        ppg.patch_records([_pred()], bench)


def test_patch_fails_on_question_text_mismatch():
    with pytest.raises(ValueError, match="input_text"):
        ppg.patch_records([_pred(q="Different question?")], [_bench_row()])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_patch_predictions_gold.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'patch_predictions_gold'`

- [ ] **Step 3: Implement the script**

Create `scripts/patch_predictions_gold.py`:

```python
"""Patch gold-label fields of an existing inference JSONL from a benchmark JSONL.

Zero-shot predictions are independent of gold labels, so when only the gold
labels of a benchmark change (same 540 questions), the already-paid-for
generations can be re-scored by overwriting the gold fields in each record —
no GPU re-run. Join is by item_id, with input_text verified against the
benchmark style_text so a silent misjoin is impossible.

Usage:
    python scripts/patch_predictions_gold.py \
        --benchmark Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/gold_seeds_styled.jsonl \
        --predictions Used_Datasets/Consolidated_Datasets/Train_Val_Dataset/M2_gss_en.jsonl \
        --output Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/M2_gss_en.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_inference as inf  # noqa: E402

GOLD_FIELDS = ["gold_category", "gold_risk_level", "gold_action",
               "gold_condition", "requires_clarification"]


def patch_records(preds: list[dict], bench_rows: list[dict],
                  language: str = "en") -> list[dict]:
    bench_by_id = {f"{b['seed_id']}_{b['style']}_{language}": b for b in bench_rows}
    missing_preds = sorted(set(bench_by_id) - {p["item_id"] for p in preds})
    if missing_preds:
        raise ValueError(f"benchmark items with no prediction: {missing_preds[:5]} "
                         f"({len(missing_preds)} total)")
    out = []
    for p in preds:
        b = bench_by_id.get(p["item_id"])
        if b is None:
            raise ValueError(f"prediction item_id not in benchmark: {p['item_id']}")
        if p.get("input_text", "").strip() != b["style_text"].strip():
            raise ValueError(f"input_text mismatch for {p['item_id']}: "
                             f"{p.get('input_text')!r} != {b['style_text']!r}")
        patched = dict(p)
        patched["gold_category"] = inf.normalize_category(b.get("category"))
        patched["gold_risk_level"] = inf.normalize_risk(b.get("gold_risk_level"))
        patched["gold_action"] = b.get("gold_action", "")
        patched["gold_condition"] = b.get("gold_condition", "")
        patched["requires_clarification"] = (b.get("requires_clarification") or "").strip().lower()
        out.append(patched)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, type=Path)
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--language", default="en")
    args = ap.parse_args()

    bench = [json.loads(l) for l in args.benchmark.open(encoding="utf-8") if l.strip()]
    preds = [json.loads(l) for l in args.predictions.open(encoding="utf-8") if l.strip()]
    patched = patch_records(preds, bench, args.language)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in patched:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(patched)} records patched -> {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_patch_predictions_gold.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/patch_predictions_gold.py tests/test_patch_predictions_gold.py
git commit -m "feat: patch gold-label fields onto existing predictions (no GPU re-run)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Run Stage 1 — re-score M2 on the new gold (STAGE 1 GATE)

**Files:**
- Create (generated): `Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/gold_seeds_styled.jsonl`, `M2_gss_en.jsonl`, `M2_gss_summary.json`, `M2_gss_scored.csv`, `safety_M2_gss.json`, `safety_M2_gss.md` (all in that folder)

**Interfaces:**
- Consumes: Task 1 CLI, Task 2 CLI, existing `scripts/evaluate.py` and `scripts/safety_metrics.py` CLIs.
- Produces: the paper's zero-shot baseline numbers on the new benchmark.

- [ ] **Step 1: Convert the new labeled CSV to benchmark JSONL**

```bash
.venv/Scripts/python scripts/convert_gold_seeds_styled.py \
  --src Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/gold_seeds_styled_labeled.csv \
  --out Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/gold_seeds_styled.jsonl
```

Expected: `540 rows -> ... (90 seed groups)`. Anything else → stop and investigate.

- [ ] **Step 2: Patch the existing M2 predictions onto the new gold**

```bash
.venv/Scripts/python scripts/patch_predictions_gold.py \
  --benchmark Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/gold_seeds_styled.jsonl \
  --predictions Used_Datasets/Consolidated_Datasets/Train_Val_Dataset/M2_gss_en.jsonl \
  --output Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/M2_gss_en.jsonl
```

Expected: `540 records patched -> ...`. A ValueError here means the item_id/question join broke — stop and report.

- [ ] **Step 3: Evaluate + safety metrics**

```bash
.venv/Scripts/python scripts/evaluate.py \
  --predictions Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/M2_gss_en.jsonl \
  --summary Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/M2_gss_summary.json \
  --scored-csv Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/M2_gss_scored.csv \
  --expected-count 540
.venv/Scripts/python scripts/safety_metrics.py \
  --predictions M2gss=Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/M2_gss_en.jsonl \
  --out-md Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/safety_M2_gss.md \
  --out-json Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/safety_M2_gss.json \
  --expected-count 540
```

Expected: both exit 0. Sanity checks on the summary JSON: `parse_ok_rate` = 1.0 (raw responses unchanged); risk majority baseline ≈ 348/540 = 0.644.

- [ ] **Step 4: Commit the Stage 1 artifacts**

```bash
git add Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/
git commit -m "results: M2 zero-shot re-scored against revised gold labels (new benchmark)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 5: STAGE 1 GATE — report and wait**

Report to Mazen: headline metrics (risk_accuracy, under_triage_rate, clarification recall/specificity, majority baselines) side-by-side with the previous gold's numbers (risk_acc 0.637, under-triage 0.750). **Do not start Task 4 until approved.**

---

### Task 4: `prepare_ft_data_v2.py` — EN path (leakage filter, dedup, chat records)

**Files:**
- Create: `scripts/prepare_ft_data_v2.py`
- Test: `tests/test_prepare_ft_data_v2.py`

**Interfaces:**
- Consumes: canonical CSVs with columns `Question,Answer,Topic,Keywords`; the benchmark labeled CSV (for the leakage question set).
- Produces: CLI `python scripts/prepare_ft_data_v2.py --lang en [--train PATH --val PATH --benchmark PATH --out-dir PATH]`. Emits `data/ft/{lang}_v2/train.jsonl`, `val.jsonl` (records `{"messages": [system, user, assistant], "category": str}`), and `leakage_log.csv` (columns `split,reason,Question`). Exposes pure functions used by tests and Task 5/6: `norm_q(q: str) -> str` (strip + casefold), `clean_splits(train_rows, val_rows, bench_questions) -> tuple[list, list, list[dict]]` (returns cleaned train, cleaned val, log rows), `to_chat_record(row: dict) -> dict`, and the constant `SYSTEM_PROMPT`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_prepare_ft_data_v2.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_ft_data_v2 as prep


def _row(q, a="An answer.", topic="PCOS"):
    return {"Question": q, "Answer": a, "Topic": topic, "Keywords": "k"}


def test_leaked_questions_dropped_from_both_splits():
    bench = {prep.norm_q("Leaky train Q?"), prep.norm_q("Leaky val Q?")}
    train = [_row("Leaky train Q?"), _row("Clean train Q?")]
    val = [_row("Leaky val Q?"), _row("Clean val Q?")]
    ctrain, cval, log = prep.clean_splits(train, val, bench)
    assert [r["Question"] for r in ctrain] == ["Clean train Q?"]
    assert [r["Question"] for r in cval] == ["Clean val Q?"]
    reasons = {(l["split"], l["reason"]) for l in log}
    assert ("train", "benchmark_leak") in reasons
    assert ("val", "benchmark_leak") in reasons


def test_train_val_dup_dropped_from_train_val_wins():
    train = [_row("Shared Q?"), _row("Train only Q?")]
    val = [_row("shared q?")]  # case-insensitive match
    ctrain, cval, log = prep.clean_splits(train, val, set())
    assert [r["Question"] for r in ctrain] == ["Train only Q?"]
    assert [r["Question"] for r in cval] == ["shared q?"]
    assert log[0]["reason"] == "train_val_dup"
    assert log[0]["split"] == "train"


def test_chat_record_shape():
    rec = prep.to_chat_record(_row("Q?", "A."))
    roles = [m["role"] for m in rec["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert rec["messages"][0]["content"] == prep.SYSTEM_PROMPT
    assert rec["messages"][1]["content"] == "Q?"
    assert rec["messages"][2]["content"] == "A."
    assert rec["category"] == "pcos"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_prepare_ft_data_v2.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prepare_ft_data_v2'`

- [ ] **Step 3: Implement the script**

Create `scripts/prepare_ft_data_v2.py`:

```python
"""Prepare the 200_Seed_Dataset canonical QA splits for Qwen QLoRA training.

Reads train_canonical.csv / validation_canonical.csv (plain Question/Answer
QA), drops rows whose Question appears verbatim (case-insensitive) in the
styled benchmark, dedups questions shared between train and val (val wins so
the validation set stays untouched), and emits Qwen chat-message JSONL in the
same record shape as prepare_ft_data.py. Every dropped row is written to
leakage_log.csv with its reason.

FR/AR: pass --lang fr --train <returned fr.csv> --val <same file> where the
returned handoff CSV carries Question_translated / Answer_translated columns
(see build_translation_handoff_v2.py); rows are mapped onto Question/Answer
before the same cleaning, and split membership comes from the row_id prefix.

Run (EN):
    python scripts/prepare_ft_data_v2.py --lang en
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_inference as inf  # noqa: E402

DATA_DIR = Path("Used_Datasets/Consolidated_Datasets/200_Seed_Dataset")
DEFAULT_TRAIN = DATA_DIR / "train_canonical.csv"
DEFAULT_VAL = DATA_DIR / "validation_canonical.csv"
DEFAULT_BENCHMARK = DATA_DIR / "gold_seeds_styled_labeled.csv"

SYSTEM_PROMPT = (
    "You are HerHealthGPT, a women's-health assistant. Answer the user's "
    "question about menstrual health, PCOS, or fertility accurately, clearly, "
    "and safely, and advise seeing a doctor when appropriate."
)


def norm_q(q: str) -> str:
    return (q or "").strip().casefold()


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))


def apply_translation(rows: list[dict]) -> list[dict]:
    """Map *_translated columns onto Question/Answer; fail on empty cells."""
    out = []
    bad = []
    for r in rows:
        qt = (r.get("Question_translated") or "").strip()
        at = (r.get("Answer_translated") or "").strip()
        if not qt or not at:
            bad.append(r.get("row_id") or r.get("Question", "?"))
            continue
        mapped = dict(r)
        mapped["Question"], mapped["Answer"] = qt, at
        out.append(mapped)
    if bad:
        raise ValueError(f"{len(bad)} rows with empty translations, e.g. {bad[:5]}")
    return out


def clean_splits(train_rows: list[dict], val_rows: list[dict],
                 bench_questions: set[str]) -> tuple[list[dict], list[dict], list[dict]]:
    log: list[dict] = []

    def drop_leaks(rows: list[dict], split: str) -> list[dict]:
        kept = []
        for r in rows:
            if norm_q(r["Question"]) in bench_questions:
                log.append({"split": split, "reason": "benchmark_leak",
                            "Question": r["Question"]})
            else:
                kept.append(r)
        return kept

    train_rows = drop_leaks(train_rows, "train")
    val_rows = drop_leaks(val_rows, "val")

    val_qs = {norm_q(r["Question"]) for r in val_rows}
    kept_train = []
    for r in train_rows:
        if norm_q(r["Question"]) in val_qs:
            log.append({"split": "train", "reason": "train_val_dup",
                        "Question": r["Question"]})
        else:
            kept_train.append(r)
    return kept_train, val_rows, log


def to_chat_record(row: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["Question"].strip()},
            {"role": "assistant", "content": row["Answer"].strip()},
        ],
        "category": inf.normalize_category(row.get("Topic")),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(to_chat_record(r), ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["en", "fr", "ar"])
    ap.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    ap.add_argument("--val", type=Path, default=DEFAULT_VAL)
    ap.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or Path(f"data/ft/{args.lang}_v2")

    train_rows = read_csv(args.train)
    val_rows = read_csv(args.val)
    if args.lang != "en":
        if args.train == DEFAULT_TRAIN:
            raise SystemExit("--lang fr/ar requires --train pointing at the returned handoff CSV")
        if args.train == args.val:
            # single returned handoff file: split membership from row_id prefix
            all_rows = apply_translation(read_csv(args.train))
            train_rows = [r for r in all_rows if r["row_id"].startswith("train-")]
            val_rows = [r for r in all_rows if r["row_id"].startswith("val-")]
        else:
            train_rows = apply_translation(train_rows)
            val_rows = apply_translation(val_rows)

    bench_questions = {norm_q(r["Question"]) for r in read_csv(args.benchmark)}
    ctrain, cval, log = clean_splits(train_rows, val_rows, bench_questions)

    write_jsonl(out_dir / "train.jsonl", ctrain)
    write_jsonl(out_dir / "val.jsonl", cval)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "leakage_log.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "reason", "Question"])
        w.writeheader()
        w.writerows(log)

    # hard guarantee: no benchmark question survives into the output
    for r in ctrain + cval:
        assert norm_q(r["Question"]) not in bench_questions
    print(f"lang={args.lang} train={len(ctrain)} val={len(cval)} "
          f"dropped={len(log)} -> {out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_prepare_ft_data_v2.py -v`
Expected: 3 PASS

- [ ] **Step 5: Run on the real EN data**

```bash
.venv/Scripts/python scripts/prepare_ft_data_v2.py --lang en
```

Expected: `lang=en train=475 val=118 dropped=7 -> data\ft\en_v2` (±1 if a leaked question is also a dup — read `data/ft/en_v2/leakage_log.csv` and record the exact counts for the gate report). Note: `data/ft/` is gitignored — the JSONL is regenerable; commit only code.

- [ ] **Step 6: Full suite, then commit**

Run: `.venv/Scripts/python -m pytest`
Expected: all green.

```bash
git add scripts/prepare_ft_data_v2.py tests/test_prepare_ft_data_v2.py
git commit -m "feat: leakage-cleaned chat-JSONL prep for 200_Seed canonical splits (v2)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 7: STAGE 2 GATE — report and wait**

Report: exact train/val counts, every dropped row with its reason (from the log). **Do not start Task 5 until approved.**

---

### Task 5: Translation handoff templates (STAGE 3 GATE)

**Files:**
- Create: `scripts/build_translation_handoff_v2.py`
- Create (generated): `Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/translation_handoff_v2/{fr,ar}.csv`, `translation_handoff_v2/README.md`
- Test: `tests/test_build_translation_handoff_v2.py`

**Interfaces:**
- Consumes: `prepare_ft_data_v2.read_csv`, `clean_splits`, `norm_q`, `apply_translation` (imported — the handoff must contain exactly the rows that survive cleaning, so translators never translate a row we'd drop).
- Produces: CLI `python scripts/build_translation_handoff_v2.py`. Handoff CSV columns: `row_id,split,Question,Answer,Topic,Keywords,Question_translated,Answer_translated` where `row_id` = `train-0007` / `val-0002` (split + zero-padded source row index). Exposes `build_handoff_rows(train_rows, val_rows, bench_questions) -> list[dict]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_build_translation_handoff_v2.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_translation_handoff_v2 as handoff
import prepare_ft_data_v2 as prep


def _row(q, a="An answer.", topic="PCOS"):
    return {"Question": q, "Answer": a, "Topic": topic, "Keywords": "k"}


def test_handoff_contains_only_cleaned_rows_with_stable_ids():
    bench = {prep.norm_q("Leaky Q?")}
    train = [_row("Leaky Q?"), _row("Keep train Q?")]
    val = [_row("Keep val Q?")]
    rows = handoff.build_handoff_rows(train, val, bench)
    ids = [r["row_id"] for r in rows]
    assert ids == ["train-0001", "val-0000"]  # index from the ORIGINAL file
    assert all(r["Question_translated"] == "" == r["Answer_translated"] for r in rows)
    assert {r["split"] for r in rows} == {"train", "val"}


def test_roundtrip_filled_template_ingests_as_translation():
    bench = set()
    train = [_row("Keep train Q?")]
    rows = handoff.build_handoff_rows(train, [], bench)
    rows[0]["Question_translated"] = "Question traduite ?"
    rows[0]["Answer_translated"] = "Une réponse."
    mapped = prep.apply_translation(rows)
    assert mapped[0]["Question"] == "Question traduite ?"
    assert mapped[0]["Answer"] == "Une réponse."
    rec = prep.to_chat_record(mapped[0])
    assert rec["messages"][1]["content"] == "Question traduite ?"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_build_translation_handoff_v2.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'build_translation_handoff_v2'`

- [ ] **Step 3: Implement the script**

Create `scripts/build_translation_handoff_v2.py`:

```python
"""Build FR/AR translation handoff templates for the v2 FT corpus.

Exports the leakage-cleaned train+val rows (exactly the rows that
prepare_ft_data_v2.py keeps — translators never see a row we'd drop) to one
CSV per language with empty Question_translated / Answer_translated columns.
row_id is stable (split + zero-padded index in the ORIGINAL canonical file)
so returned files join back deterministically.

Run: python scripts/build_translation_handoff_v2.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_ft_data_v2 as prep  # noqa: E402

OUT_DIR = prep.DATA_DIR / "translation_handoff_v2"
LANGS = ["fr", "ar"]
FIELDS = ["row_id", "split", "Question", "Answer", "Topic", "Keywords",
          "Question_translated", "Answer_translated"]

README = """# Translation handoff v2 (FR / AR)

- Fill ONLY `Question_translated` and `Answer_translated` for every row.
- Do NOT edit or reorder any other column; `row_id` must come back untouched.
- Translate meaning and register (layperson tone stays layperson).
- Keep the file CSV, UTF-8 encoded. Topic/Keywords are metadata — leave in English.
- Return one completed file per language (fr.csv, ar.csv).
"""


def build_handoff_rows(train_rows: list[dict], val_rows: list[dict],
                       bench_questions: set[str]) -> list[dict]:
    # tag original indices BEFORE cleaning so row_id survives future edits
    for i, r in enumerate(train_rows):
        r["row_id"], r["split"] = f"train-{i:04d}", "train"
    for i, r in enumerate(val_rows):
        r["row_id"], r["split"] = f"val-{i:04d}", "val"
    ctrain, cval, _log = prep.clean_splits(train_rows, val_rows, bench_questions)
    out = []
    for r in ctrain + cval:
        out.append({
            "row_id": r["row_id"], "split": r["split"],
            "Question": r["Question"], "Answer": r["Answer"],
            "Topic": r.get("Topic", ""), "Keywords": r.get("Keywords", ""),
            "Question_translated": "", "Answer_translated": "",
        })
    return out


def main() -> None:
    train_rows = prep.read_csv(prep.DEFAULT_TRAIN)
    val_rows = prep.read_csv(prep.DEFAULT_VAL)
    bench_questions = {prep.norm_q(r["Question"])
                       for r in prep.read_csv(prep.DEFAULT_BENCHMARK)}
    rows = build_handoff_rows(train_rows, val_rows, bench_questions)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for lang in LANGS:
        path = OUT_DIR / f"{lang}.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"{len(rows)} rows -> {path}")
    (OUT_DIR / "README.md").write_text(README, encoding="utf-8")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_build_translation_handoff_v2.py -v`
Expected: 2 PASS

- [ ] **Step 5: Generate the real templates and commit**

```bash
.venv/Scripts/python scripts/build_translation_handoff_v2.py
```

Expected: two `N rows -> ...` lines where N = Stage-2 train+val total (~593).

```bash
.venv/Scripts/python -m pytest
git add scripts/build_translation_handoff_v2.py tests/test_build_translation_handoff_v2.py \
  Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/translation_handoff_v2/
git commit -m "feat: FR/AR translation handoff templates for v2 FT corpus

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 6: STAGE 3 GATE — hand off and wait**

Report: template paths + row count; remind the team the ingest command is
`prepare_ft_data_v2.py --lang fr --train <returned fr.csv> --val <returned fr.csv>`.
**Do not start Task 6 until approved.**

---

### Task 6: Stage 4 — EN-only QLoRA fine-tune + eval (STAGE 4 GATE)

**Files:**
- Create (generated, gitignored): `models/qwen3.5-9b-herhealth-en-v2-lora/`
- Create (generated, committed): `Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/M3en_v2_en.jsonl`, `M3en_v2_summary.json`, `safety_M2_vs_M3en_v2.{md,json}`, `compare_M2_vs_M3en_v2.{md,json}`

**Interfaces:**
- Consumes: `data/ft/en_v2/{train,val}.jsonl` (Task 4), existing `train_qlora.py`, `run_local_inference.py`, `evaluate.py`, `safety_metrics.py`, `compare_models.py` — all unchanged.
- Produces: EN-v2 adapter + fine-tuned-vs-M2 comparison on the new benchmark.

All GPU commands run inside WSL distro `Ubuntu` as user `sw2`, from the repo at `/mnt/d/Grad-Project/HerHealthGPT`, with the pinned env (`docs/ENVIRONMENTS.md`). Prefix used below:

```bash
wsl -d Ubuntu -u sw2 --cd /mnt/d/Grad-Project/HerHealthGPT -- \
  env HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 /home/sw2/ft-train-venv/bin/python ...
```

- [ ] **Step 1: Smoke gate (5 steps)**

```bash
wsl -d Ubuntu -u sw2 --cd /mnt/d/Grad-Project/HerHealthGPT -- \
  env HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 /home/sw2/ft-train-venv/bin/python \
  scripts/train_qlora.py --data-dir data/ft/en_v2 --max-steps 5 --output models/_smoke_en_v2
```

Expected: completes without error, loss finite. Delete `models/_smoke_en_v2` after.

- [ ] **Step 2: Full EN run (~25–30 min)**

```bash
wsl -d Ubuntu -u sw2 --cd /mnt/d/Grad-Project/HerHealthGPT -- \
  env HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 /home/sw2/ft-train-venv/bin/python \
  scripts/train_qlora.py --data-dir data/ft/en_v2 --epochs 3 --lr 2e-4 --batch 2 \
  --output models/qwen3.5-9b-herhealth-en-v2-lora
```

Expected: run manifest JSON written next to the adapter (epochs=3, lr=2e-4, git_sha). Record final train loss.

- [ ] **Step 3: Inference on the new benchmark (~25 min)**

```bash
wsl -d Ubuntu -u sw2 --cd /mnt/d/Grad-Project/HerHealthGPT -- \
  env HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 /home/sw2/ft-train-venv/bin/python \
  scripts/run_local_inference.py --label M3en-v2 \
  --model /home/sw2/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --adapter models/qwen3.5-9b-herhealth-en-v2-lora \
  --benchmark Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/gold_seeds_styled.jsonl \
  --output Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/M3en_v2_en.jsonl
```

Expected: 540 records. If parse failures appear, run `scripts/reparse_inference.py --input ... --in-place` first (known M3 close-brace quirk) and report the repair rate.

- [ ] **Step 4: Evaluate + compare vs Stage-1 M2**

```bash
.venv/Scripts/python scripts/evaluate.py \
  --predictions Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/M3en_v2_en.jsonl \
  --summary Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/M3en_v2_summary.json \
  --expected-count 540
.venv/Scripts/python scripts/safety_metrics.py \
  --predictions M2gss=Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/M2_gss_en.jsonl \
  --predictions M3env2=Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/M3en_v2_en.jsonl \
  --out-md Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/safety_M2_vs_M3en_v2.md \
  --out-json Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/safety_M2_vs_M3en_v2.json \
  --expected-count 540
.venv/Scripts/python scripts/compare_models.py \
  --baseline Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/M2_gss_en.jsonl \
  --treatment Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/M3en_v2_en.jsonl \
  --baseline-label M2 --treatment-label M3en-v2 \
  --out-md Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/compare_M2_vs_M3en_v2.md \
  --out-json Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/compare_M2_vs_M3en_v2.json
```

Expected: all exit 0; safety_metrics runs McNemar pair tests (two predictions given).

- [ ] **Step 5: Commit artifacts**

```bash
git add Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/
git commit -m "results: M3en-v2 (EN-only QLoRA on v2 corpus) vs M2 on new benchmark

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 6: STAGE 4 GATE — report and wait**

Report: comparison table (risk_acc, under-triage, clarification, parse_ok, McNemar p-values), post-FT parse_ok explicitly (a material drop below 1.0 is a finding, not something to absorb silently). **Do not start Task 7 until approved.**

---

### Task 7: Stage 5 — joint EN+FR+AR fine-tune + eval (STAGE 5 GATE; blocked on translations)

**Files:**
- Create: `scripts/merge_ft_langs.py`
- Test: `tests/test_merge_ft_langs.py`
- Create (generated): `data/ft/enfrar_v2/{train,val}.jsonl` (gitignored), `models/qwen3.5-9b-herhealth-enfrar-lora/` (gitignored), eval artifacts `M3ml_v2_en.jsonl`, `M3ml_v2_summary.json`, `safety_M2_vs_M3ml_v2.{md,json}`, `compare_M2_vs_M3ml_v2.{md,json}` in `200_Seed_Dataset/`

**Interfaces:**
- Consumes: `data/ft/{en,fr,ar}_v2/{train,val}.jsonl` — FR/AR produced by `prepare_ft_data_v2.py --lang fr|ar --train <returned csv> --val <returned csv>` once the team returns the templates.
- Produces: `merge_shuffle(paths: list[Path], seed: int) -> list[dict]` and CLI `python scripts/merge_ft_langs.py --langs en,fr,ar --split train --seed 3407` (same for `val`), writing `data/ft/enfrar_v2/{split}.jsonl`.

- [ ] **Step 0: Precondition** — team translations returned and ingested:

```bash
.venv/Scripts/python scripts/prepare_ft_data_v2.py --lang fr \
  --train Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/translation_handoff_v2/fr.csv \
  --val   Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/translation_handoff_v2/fr.csv
.venv/Scripts/python scripts/prepare_ft_data_v2.py --lang ar \
  --train Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/translation_handoff_v2/ar.csv \
  --val   Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/translation_handoff_v2/ar.csv
```

Expected: counts matching the EN splits. If translations are not back, stop here — Stages 1–4 stand alone.

- [ ] **Step 1: Write the failing test**

Create `tests/test_merge_ft_langs.py`:

```python
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import merge_ft_langs as mfl


def _write(path, texts):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"messages": [{"role": "user", "content": t}]}) + "\n")


def test_merge_is_deterministic_and_complete(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write(a, ["a1", "a2"])
    _write(b, ["b1"])
    merged1 = mfl.merge_shuffle([a, b], seed=3407)
    merged2 = mfl.merge_shuffle([a, b], seed=3407)
    assert merged1 == merged2
    contents = {r["messages"][0]["content"] for r in merged1}
    assert contents == {"a1", "a2", "b1"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_merge_ft_langs.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

Create `scripts/merge_ft_langs.py`:

```python
"""Merge per-language FT chat JSONL files into one shuffled corpus.

Run:
    python scripts/merge_ft_langs.py --langs en,fr,ar --split train --seed 3407
    python scripts/merge_ft_langs.py --langs en,fr,ar --split val --seed 3407
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def merge_shuffle(paths: list[Path], seed: int) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        rows.extend(json.loads(l) for l in p.open(encoding="utf-8") if l.strip())
    random.Random(seed).shuffle(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="en,fr,ar")
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--out-dir", type=Path, default=Path("data/ft/enfrar_v2"))
    args = ap.parse_args()

    paths = [Path(f"data/ft/{lang}_v2/{args.split}.jsonl")
             for lang in args.langs.split(",")]
    for p in paths:
        if not p.exists():
            raise SystemExit(f"missing input: {p}")
    rows = merge_shuffle(paths, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.split}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} rows ({args.langs}) -> {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test, merge real data, commit code**

```bash
.venv/Scripts/python -m pytest tests/test_merge_ft_langs.py -v
.venv/Scripts/python scripts/merge_ft_langs.py --langs en,fr,ar --split train
.venv/Scripts/python scripts/merge_ft_langs.py --langs en,fr,ar --split val
git add scripts/merge_ft_langs.py tests/test_merge_ft_langs.py
git commit -m "feat: deterministic per-language FT corpus merge

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

Expected merged train ≈ 3 × EN train count (~1,425).

- [ ] **Step 5: Smoke gate, then full joint run (2 epochs, ~1 h)**

```bash
wsl -d Ubuntu -u sw2 --cd /mnt/d/Grad-Project/HerHealthGPT -- \
  env HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 /home/sw2/ft-train-venv/bin/python \
  scripts/train_qlora.py --data-dir data/ft/enfrar_v2 --max-steps 5 --output models/_smoke_ml
wsl -d Ubuntu -u sw2 --cd /mnt/d/Grad-Project/HerHealthGPT -- \
  env HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 /home/sw2/ft-train-venv/bin/python \
  scripts/train_qlora.py --data-dir data/ft/enfrar_v2 --epochs 2 --lr 2e-4 --batch 2 \
  --output models/qwen3.5-9b-herhealth-enfrar-lora
```

- [ ] **Step 6: Inference + eval on the new benchmark**

```bash
wsl -d Ubuntu -u sw2 --cd /mnt/d/Grad-Project/HerHealthGPT -- \
  env HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 /home/sw2/ft-train-venv/bin/python \
  scripts/run_local_inference.py --label M3ml-v2 \
  --model /home/sw2/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --adapter models/qwen3.5-9b-herhealth-enfrar-lora \
  --benchmark Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/gold_seeds_styled.jsonl \
  --output Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/M3ml_v2_en.jsonl
.venv/Scripts/python scripts/evaluate.py \
  --predictions Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/M3ml_v2_en.jsonl \
  --summary Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/M3ml_v2_summary.json \
  --expected-count 540
.venv/Scripts/python scripts/safety_metrics.py \
  --predictions M2gss=Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/M2_gss_en.jsonl \
  --predictions M3mlv2=Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/M3ml_v2_en.jsonl \
  --out-md Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/safety_M2_vs_M3ml_v2.md \
  --out-json Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/safety_M2_vs_M3ml_v2.json \
  --expected-count 540
.venv/Scripts/python scripts/compare_models.py \
  --baseline Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/M2_gss_en.jsonl \
  --treatment Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/M3ml_v2_en.jsonl \
  --baseline-label M2 --treatment-label M3ml-v2 \
  --out-md Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/compare_M2_vs_M3ml_v2.md \
  --out-json Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/compare_M2_vs_M3ml_v2.json
```

Expected: 540 records; all eval commands exit 0. If parse failures appear, run `scripts/reparse_inference.py --input ... --in-place` first and report the repair rate.

- [ ] **Step 7: Commit artifacts + STAGE 5 GATE report**

```bash
git add Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/
git commit -m "results: M3ml-v2 (joint EN+FR+AR QLoRA) vs M2 on new benchmark

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

Report: three-way M2 / M3en-v2 / M3ml-v2 comparison; note the English-benchmark-only evaluation limitation for the paper.
