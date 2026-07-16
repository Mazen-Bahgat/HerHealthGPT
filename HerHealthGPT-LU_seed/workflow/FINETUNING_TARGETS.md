# FINETUNING_TARGETS.md — HerHealthGPT-LU

**Status:** Canonical / frozen · **Scope:** Seed counts, repository hierarchy, and file-to-phase declaration for the FemSympQA seed pipeline · **Supersedes:** ad-hoc target numbers mentioned in earlier planning docs

---

## 1. Seed numbers (agreed)

### 1.1 Current benchmark seeds (v1) — frozen

| Category | Seeds | Styles per seed | Rows |
|---|---|---|---|
| Menstrual | 30 | 6 | 180 |
| PCOS | 30 | 6 | 180 |
| Fertility | 30 | 6 | 180 |
| **Total** | **90** | 6 | **540** |

This is the frozen benchmark/eval table (`seeds_en_v1.csv` / `.jsonl`). It does not change when fine-tuning targets change.

### 1.2 Fine-tuning targets — new finalized target

| Metric | Value |
|---|---|
| Fine-tune seeds per category (range) | 800–1,000 |
| Total fine-tune seeds (range, 3 categories) | 2,400–3,000 |
| **Recommended default** | **900 per category → 2,700 total** |
| Balanced maximum from current eligible pool (ceiling) | 1,154 per category → 3,462 total |

**Reading order for these numbers:** the range (800–1,000/category) is the agreed target band; 900/category is the working default unless a category's eligible pool can't support it; 1,154/category is the hard ceiling set by what the deduped candidate pool (`phase2_candidates_deduped.csv`) can actually supply today — it is not a target, it's a constraint.

---

## 2. Repository hierarchy (official declaration)

```
HerHealthGPT-LU_seed/
│
├── README.md
│   ← Start here: explains scope, schema, and how all files connect.
│
├── FINAL_for_team/
│   ├── seeds_en_v1.csv
│   │   ← Main frozen benchmark table (90 seeds × 6 styles = 540 rows).
│   ├── seeds_en_v1.jsonl
│   │   ← Same benchmark content in JSONL for pipelines.
│   ├── stats_summary.md
│   │   ← Final measured counts/stats + planning note.
│   ├── decisions_log.md
│   │   ← Why targets and filtering choices were made.
│   ├── leakage_note.md
│   │   ← Source rows reserved for benchmark; MUST be excluded from training.
│   ├── borderline_bucket.csv
│   │   ← Borderline candidates for later manual promotion/rejection.
│   └── FINETUNING_TARGETS.md
│       ← Canonical fine-tuning plan (this file):
│          800–1,000/category (2,400–3,000 total),
│          default 900/category (2,700 total).
│
├── DIRECTION_docs/
│   ├── LUHME_direction_workflow.md
│   │   ← Full 11-phase operational plan for LUHME.
│   └── LUHME_direction_summary.md
│       ← One-page summary for meeting/Overleaf.
│
├── PIPELINE/
│   └── build_seed.py
│       ← Reproducible script that builds seed artifacts.
│
└── INTERMEDIATES_audit/
    ├── phase1_candidates_raw.csv
    │   ← Raw keyword hits before final filtering.
    ├── phase2_candidates_deduped.csv
    │   ← Deduplicated candidate pool (main source for train/val creation).
    ├── phase2_dedup_drops.jsonl
    │   ← Audit trail of duplicate removals.
    ├── phase3_proposed_targets.json
    │   ← Proposed per-category counts at selection time.
    ├── phase3_selected_seeds.json
    │   ← Selected seeds before final export format.
    └── style_variants.json
        ← Generated variants before merge into final seed table.
```

This structure is confirmed correct — use it as the official declaration going forward.

---

## 3. Which files are used in the fine-tuning process

**Used directly**

- `INTERMEDIATES_audit/phase2_candidates_deduped.csv` → build balanced train/val data from here.
- `FINAL_for_team/leakage_note.md` → exclude these `source_dataset` + `source_row_id` pairs from train/val.
- `FINAL_for_team/FINETUNING_TARGETS.md` → official target numbers for split planning (this file).

**Used for evaluation (not training)**

- `FINAL_for_team/seeds_en_v1.csv` (or `.jsonl`) → frozen benchmark/eval set.

**Used for governance/reproducibility**

- `PIPELINE/build_seed.py`
- `FINAL_for_team/decisions_log.md`
- `FINAL_for_team/stats_summary.md`

**Simple train/eval rule**

- Train/Val: from `phase2_candidates_deduped.csv`
- Eval benchmark: `seeds_en_v1.*`
- No overlap allowed: enforce with `leakage_note.md`

---

## 4. Dataset Artifact Structure (ready-to-paste table)

| File | Purpose | Used-in-Phase | Used-for-Train | Used-for-Evaluate |
|---|---|---|---|---|
| `HerHealthGPT-LU_seed/README.md` | Entry-point documentation (scope, schema, workflow) | Documentation | No | No |
| `HerHealthGPT-LU_seed/seeds_en_v1.csv` | Final English seed benchmark table (canonical + style variants) | Benchmark freeze | No (holdout only) | Yes |
| `HerHealthGPT-LU_seed/seeds_en_v1.jsonl` | JSONL equivalent of final benchmark table | Benchmark freeze | No (holdout only) | Yes |
| `HerHealthGPT-LU_seed/stats_summary.md` | Final counts by category/source/confidence; dedup stats + planning note | Reporting | No | Indirect (audit/reporting) |
| `HerHealthGPT-LU_seed/decisions_log.md` | Rationale for target sizes and filtering choices | Governance | No | Indirect (method transparency) |
| `HerHealthGPT-LU_seed/leakage_note.md` | Explicit source row IDs reserved for benchmark; must be excluded from train corpora | Leakage control | Yes (as exclusion list) | Yes (integrity control) |
| `HerHealthGPT-LU_seed/borderline_bucket.csv` | Borderline candidates for manual review/future benchmark expansion | Curation | Optional (after review) | Optional (future versions) |
| `HerHealthGPT-LU_seed/FINETUNING_TARGETS.md` | Canonical fine-tuning target: 800–1,000/category (2,400–3,000 total), default 900/category | Split planning | Yes (planning reference) | Indirect |
| `HerHealthGPT-LU_seed/build_seed.py` | Reproducible seed-construction pipeline | Reproducibility | No (data builder) | No |
| `HerHealthGPT-LU_seed/phase1_candidates_raw.csv` | Raw keyword-matched candidates before final filtering | Candidate extraction | Yes (after filtering) | No |
| `HerHealthGPT-LU_seed/phase2_candidates_deduped.csv` | Deduplicated candidate pool with provenance | Post-dedup candidate pool | Yes (primary source for train/val split) | No |
| `HerHealthGPT-LU_seed/phase2_dedup_drops.jsonl` | Audit trail of dropped near-duplicate clusters | Dedup audit | No | No |
| `HerHealthGPT-LU_seed/phase3_proposed_targets.json` | Proposed per-category target counts | Seed planning | No | Indirect |
| `HerHealthGPT-LU_seed/phase3_selected_seeds.json` | Selected seed objects before final export formatting | Seed finalization | No (unless reversioned) | Indirect |
| `HerHealthGPT-LU_seed/style_variants.json` | Generated variant texts before final merge/export | Variant generation | Optional (if explicitly included) | Yes (if style-based evaluation is used) |
| `HerHealthGPT-LU_seed/LUHME_direction_workflow.md` | Full 11-phase project execution plan | Planning | No | No |
| `HerHealthGPT-LU_seed/LUHME_direction_summary.md` | One-page planning summary for meetings/quick reference | Planning | No | No |

---

## 5. Train/Eval Separation Rule (critical)

| Rule | Enforcement File | Outcome |
|---|---|---|
| No benchmark example may appear in training/validation | `leakage_note.md` | Prevents leakage |
| Match by provenance key | `source_dataset` + `source_row_id` | Deterministic exclusion |
| Benchmark is frozen holdout | `seeds_en_v1.csv` / `seeds_en_v1.jsonl` | Used only for evaluation |
| Fine-tuning built from non-benchmark pool | `phase2_candidates_deduped.csv` (after exclusion) | Clean train/val splits |
| Fine-tuning size policy | `FINETUNING_TARGETS.md` | Balanced target (800–1,000/category; default 900/category) |
