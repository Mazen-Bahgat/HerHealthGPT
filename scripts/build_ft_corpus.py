"""HerHealthGPT-LU §2C — build the silver fine-tuning corpus (steps 1-7).

Implements the join procedure from
docs/superpowers/specs/2026-07-06-herhealthgpt-lu-design.md §2C, reusing the
category classifier already tuned and frozen for benchmark seed selection in
HerHealthGPT-LU_seed/build_seed.py (assign_best_category / clean_text) so the
FT corpus and the benchmark seeds share one definition of "PCOS" / "fertility"
/ "menstrual".

Dual leakage key (spec §8, "highest-priority correction"):
  1. (source_dataset, source_row_id) — exact originating row of every seed.
  2. seed_answer_hash — sha1(normalized MENST Answer text). One MENST answer
     has ~13 paraphrase-augmented siblings under different row IDs; the row-id
     key alone lets those siblings leak into the FT corpus. Only MENST rows
     get a seed_answer_hash (this is a MENST-specific augmentation artifact;
     iCliniq/HCM answers are not paraphrase-duplicated the same way).

Run: python scripts/build_ft_corpus.py [--sample N]

Outputs (all under HerHealthGPT-LU_seed/):
  ft_corpus_v1.jsonl   Llama-Factory {"instruction","input","output"} format
  leakage_log.csv      audit trail: every row excluded during this build + why
  leakage_note.md       regenerated block-list (adds seed_answer_hash column)
  ft_corpus_stats.md    category balance + step-by-step yield report
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED_DIR = REPO_ROOT / "HerHealthGPT-LU_seed"
DATASET_DIR = REPO_ROOT / "Dataset"
MENST_DIR = DATASET_DIR / "MENST"
HCM_PATH = DATASET_DIR / "HealthCareMagic-100k-Chat-Format-en" / "HealthCareMagic-100k-en.jsonl"
ICLINIQ_DIR = DATASET_DIR / "ChatDoctor-iCliniq" / "data"

SEEDS_CSV = SEED_DIR / "seeds_en_v1.csv"
FT_CORPUS_OUT = SEED_DIR / "ft_corpus_v1.jsonl"
LEAKAGE_LOG_OUT = SEED_DIR / "leakage_log.csv"
LEAKAGE_NOTE_OUT = SEED_DIR / "leakage_note.md"
STATS_OUT = SEED_DIR / "ft_corpus_stats.md"

sys.path.insert(0, str(SEED_DIR))
import build_seed as bs  # noqa: E402  (reuse frozen category classifier)

CATEGORIES = bs.CATEGORIES  # ("menstrual", "pcos", "fertility")
TARGET_PER_CATEGORY = 900  # decisions_log.md recommended default
SYSTEM_INSTRUCTION = (
    "You are a supportive women's-health information assistant. Answer the "
    "patient's question clearly and safely, and recommend seeing a clinician "
    "for diagnosis or treatment decisions."
)


@dataclass
class QAPair:
    qa_id: str
    source_dataset: str
    source_row_id: str
    category: str
    raw_question: str
    raw_answer: str
    answer_hash: str = ""
    meta: dict = field(default_factory=dict)


def sha1_normalized(text: str) -> str:
    norm = re.sub(r"\s+", " ", text.strip().lower())
    norm = re.sub(r"[^\w\s]", "", norm)
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Step 1 — normalize each source to {qa_id, source, raw_question, raw_answer}
# ---------------------------------------------------------------------------


def load_menst_qa() -> list[QAPair]:
    pairs: list[QAPair] = []
    for fname, tag in (("training2K.csv", "MENST_training2K"), ("train24K.csv", "MENST_train24K")):
        df = pd.read_csv(MENST_DIR / fname)
        df.columns = [c.strip() for c in df.columns]
        for idx, row in df.iterrows():
            s = int(row["Set"]) if pd.notna(row["Set"]) else -1
            if s not in (1, 2):
                continue  # Set 3 excluded per decisions_log.md; matches seed policy
            q = bs.clean_text(row["Question"])
            a = bs.clean_text(row.get("Answer", ""))
            if not q or not a:
                continue
            row_id = f"{fname.replace('.csv', '')}:{idx}"
            pairs.append(
                QAPair(
                    qa_id=f"menst:{row_id}",
                    source_dataset=tag,
                    source_row_id=row_id,
                    category="",
                    raw_question=q,
                    raw_answer=a,
                    answer_hash=sha1_normalized(a),
                    meta={"topic": bs.clean_text(row.get("Topic", "")), "menst_set": str(s)},
                )
            )
    return pairs


def load_icliniq_qa() -> list[QAPair]:
    pq = next(ICLINIQ_DIR.glob("*.parquet"))
    df = pd.read_parquet(pq)
    pairs: list[QAPair] = []
    for idx, row in df.iterrows():
        q = bs.clean_text(row["input"])
        # answer_icliniq only — avoid GPT-style leakage from answer_chatgpt/answer_chatdoctor
        a = bs.clean_text(row.get("answer_icliniq", ""))
        if not q or not a:
            continue
        pairs.append(
            QAPair(
                qa_id=f"icliniq:{idx}",
                source_dataset="ChatDoctor-iCliniq",
                source_row_id=str(idx),
                category="",
                raw_question=q,
                raw_answer=a,
            )
        )
    return pairs


def load_hcm_qa() -> list[QAPair]:
    pairs: list[QAPair] = []
    with open(HCM_PATH, encoding="utf-8", errors="replace") as f:
        for line_i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text", "")
            q = bs.extract_human_hcm(text)
            a = ""
            if "<bot>:" in text:
                a = bs.clean_text(text.split("<bot>:", 1)[1])
            if not q or not a:
                continue
            pairs.append(
                QAPair(
                    qa_id=f"hcm:{line_i}",
                    source_dataset="HealthCareMagic-100k",
                    source_row_id=str(line_i),
                    category="",
                    raw_question=q,
                    raw_answer=a,
                )
            )
    return pairs


# ---------------------------------------------------------------------------
# Step 2 — domain filter (reuses build_seed.py's frozen classifier)
# ---------------------------------------------------------------------------


def categorize(pairs: list[QAPair]) -> list[QAPair]:
    out = []
    for p in pairs:
        for cat, tier, _notes in bs.assign_best_category(p.raw_question):
            if tier == "Reject":
                continue
            out.append(
                QAPair(
                    qa_id=f"{p.qa_id}:{cat}",
                    source_dataset=p.source_dataset,
                    source_row_id=p.source_row_id,
                    category=cat,
                    raw_question=p.raw_question,
                    raw_answer=p.raw_answer,
                    answer_hash=p.answer_hash,
                    meta=p.meta,
                )
            )
    return out


# ---------------------------------------------------------------------------
# Step 4 — quality/precision filter
# ---------------------------------------------------------------------------

JUNK_ANSWER = re.compile(
    r"^\s*(?:please\s+)?(?:consult|see|visit)\s+(?:a|your)\s+(?:doctor|gynecologist|gp|physician)\.?\s*$",
    re.I,
)
URL_ONLY = re.compile(r"^\s*https?://\S+\s*$")


def _is_mostly_ascii(text: str) -> bool:
    if not text:
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / len(text) > 0.9


def quality_filter(pairs: list[QAPair]) -> tuple[list[QAPair], Counter]:
    drop_reasons: Counter = Counter()
    kept: list[QAPair] = []
    seen_qa: set[str] = set()
    for p in pairs:
        a = p.raw_answer
        if not (20 <= len(a) <= 1500):
            drop_reasons["answer_length"] += 1
            continue
        if JUNK_ANSWER.match(a) or URL_ONLY.match(a):
            drop_reasons["junk_answer"] += 1
            continue
        if not _is_mostly_ascii(p.raw_question) or not _is_mostly_ascii(a):
            drop_reasons["non_english_heuristic"] += 1
            continue
        key = bs.normalize_for_dedup(p.raw_question)[:220] + "|" + bs.normalize_for_dedup(a)[:220]
        if key in seen_qa:
            drop_reasons["exact_qa_dup"] += 1
            continue
        seen_qa.add(key)
        kept.append(p)
    return kept, drop_reasons


# ---------------------------------------------------------------------------
# Step 6 — leakage exclusion (dual key)
# ---------------------------------------------------------------------------


def load_seed_leakage_keys() -> tuple[list[dict], set[tuple[str, str]], dict[str, str]]:
    """Return (seed rows, {(source_dataset, source_row_id)}, {source_row_id: answer_hash for MENST seeds})."""
    seeds: list[dict] = []
    with open(SEEDS_CSV, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            seeds.append(row)
    # one row per (seed_id, style) in the CSV; collapse to distinct seeds
    distinct = {}
    for row in seeds:
        distinct[row["seed_id"]] = row
    row_id_keys = {(row["source_dataset"], row["source_row_id"]) for row in distinct.values()}

    # look up MENST seed answers to compute seed_answer_hash
    menst_answers: dict[tuple[str, str], str] = {}
    for fname, tag in (("training2K.csv", "MENST_training2K"), ("train24K.csv", "MENST_train24K")):
        df = pd.read_csv(MENST_DIR / fname)
        df.columns = [c.strip() for c in df.columns]
        for idx, row in df.iterrows():
            menst_answers[(tag, f"{fname.replace('.csv', '')}:{idx}")] = bs.clean_text(row.get("Answer", ""))

    seed_answer_hash: dict[str, str] = {}  # seed_id -> hash (empty for non-MENST)
    for seed_id, row in distinct.items():
        key = (row["source_dataset"], row["source_row_id"])
        if row["source_dataset"].startswith("MENST") and key in menst_answers and menst_answers[key]:
            seed_answer_hash[seed_id] = sha1_normalized(menst_answers[key])
        else:
            seed_answer_hash[seed_id] = ""

    return list(distinct.values()), row_id_keys, seed_answer_hash


def write_leakage_note(distinct_seeds: list[dict], seed_answer_hash: dict[str, str]) -> None:
    lines = [
        "# Leakage exclusion list — seeds_en_v1\n\n",
        "Exclude these from fine-tuning pulls. Dual key: a row is excluded if it "
        "matches EITHER `(source_dataset, source_row_id)` OR `seed_answer_hash` "
        "(MENST paraphrase-family key — see build_ft_corpus.py docstring).\n\n",
        "| seed_id | category | source_dataset | source_row_id | seed_answer_hash |\n",
        "|---|---|---|---|---|\n",
    ]
    for row in sorted(distinct_seeds, key=lambda r: r["seed_id"]):
        h = seed_answer_hash.get(row["seed_id"], "") or "—"
        lines.append(
            f"| {row['seed_id']} | {row['category']} | {row['source_dataset']} | "
            f"{row['source_row_id']} | `{h}` |\n"
        )
    LEAKAGE_NOTE_OUT.write_text("".join(lines), encoding="utf-8")


def apply_leakage_exclusion(
    pairs: list[QAPair], row_id_keys: set[tuple[str, str]], answer_hashes: set[str]
) -> tuple[list[QAPair], list[dict]]:
    kept: list[QAPair] = []
    excluded: list[dict] = []
    for p in pairs:
        key = (p.source_dataset, p.source_row_id)
        if key in row_id_keys:
            excluded.append(
                {"qa_id": p.qa_id, "source_dataset": p.source_dataset, "source_row_id": p.source_row_id,
                 "exclusion_reason": "source_row_id_match"}
            )
            continue
        if p.answer_hash and p.answer_hash in answer_hashes:
            excluded.append(
                {"qa_id": p.qa_id, "source_dataset": p.source_dataset, "source_row_id": p.source_row_id,
                 "exclusion_reason": "answer_hash_match"}
            )
            continue
        kept.append(p)
    return kept, excluded


# ---------------------------------------------------------------------------
# Step 7 — balance + format for Llama-Factory
# ---------------------------------------------------------------------------


def balance_and_format(pairs: list[QAPair], target_per_category: int) -> list[dict]:
    by_cat: dict[str, list[QAPair]] = defaultdict(list)
    for p in pairs:
        by_cat[p.category].append(p)
    formatted: list[dict] = []
    for cat in CATEGORIES:
        pool = by_cat.get(cat, [])
        pool.sort(key=lambda p: (0 if p.source_dataset.startswith("MENST") else 1, p.source_row_id))
        take = pool[:target_per_category]
        for p in take:
            formatted.append(
                {
                    "instruction": SYSTEM_INSTRUCTION,
                    "input": p.raw_question,
                    "output": p.raw_answer,
                    "category": p.category,
                    "source_dataset": p.source_dataset,
                    "source_row_id": p.source_row_id,
                }
            )
    return formatted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None, help="cap rows per source for a fast dry run")
    args = parser.parse_args()

    print("Step 1: loading + normalizing sources...")
    menst = load_menst_qa()
    icliniq = load_icliniq_qa()
    hcm = load_hcm_qa()
    if args.sample:
        menst, icliniq, hcm = menst[: args.sample], icliniq[: args.sample], hcm[: args.sample]
    print(f"  MENST {len(menst)}  iCliniq {len(icliniq)}  HCM {len(hcm)}")
    all_pairs = menst + icliniq + hcm

    print("Step 2: domain categorization (reusing build_seed.py's frozen classifier)...")
    categorized = categorize(all_pairs)
    print(f"  {len(categorized)} (question, category) pairs across {len(CATEGORIES)} categories")
    print("  by category:", Counter(p.category for p in categorized))

    print("Step 4: quality/precision filter...")
    filtered, drop_reasons = quality_filter(categorized)
    print(f"  kept {len(filtered)}, dropped {sum(drop_reasons.values())}: {dict(drop_reasons)}")

    print("Step 5/6: loading frozen seed leakage keys + computing seed_answer_hash...")
    distinct_seeds, row_id_keys, seed_answer_hash = load_seed_leakage_keys()
    answer_hashes = {h for h in seed_answer_hash.values() if h}
    print(f"  {len(distinct_seeds)} seeds, {len(row_id_keys)} row-id keys, {len(answer_hashes)} answer hashes (MENST)")
    write_leakage_note(distinct_seeds, seed_answer_hash)

    clean, excluded = apply_leakage_exclusion(filtered, row_id_keys, answer_hashes)
    print(f"  leakage-excluded {len(excluded)} rows; {len(clean)} remain")
    with open(LEAKAGE_LOG_OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["qa_id", "source_dataset", "source_row_id", "exclusion_reason"])
        w.writeheader()
        w.writerows(excluded)

    print("Step 7: balancing to target/category + formatting for Llama-Factory...")
    formatted = balance_and_format(clean, TARGET_PER_CATEGORY)
    with open(FT_CORPUS_OUT, "w", encoding="utf-8") as f:
        for row in formatted:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    final_counts = Counter(r["category"] for r in formatted)
    stats = [
        "# FT corpus v1 (English) — build stats\n\n",
        f"**Total after §2C steps 1-7:** {len(formatted)}  \n",
        f"**Target:** {TARGET_PER_CATEGORY}/category ({TARGET_PER_CATEGORY * 3} total)\n\n",
        "## Yield per stage\n\n",
        "| Stage | Count |\n|---|---:|\n",
        f"| Raw normalized (Step 1) | {len(all_pairs)} |\n",
        f"| Domain-categorized (Step 2) | {len(categorized)} |\n",
        f"| Quality-filtered (Step 4) | {len(filtered)} |\n",
        f"| Leakage-excluded (Step 6) | {len(excluded)} |\n",
        f"| Final FT corpus (Step 7, balanced) | {len(formatted)} |\n\n",
        "## Final category balance\n\n| Category | Count |\n|---|---:|\n",
    ]
    for cat in CATEGORIES:
        note = "" if final_counts.get(cat, 0) >= TARGET_PER_CATEGORY else " — **below target, see eligible-pool note below**"
        stats.append(f"| {cat} | {final_counts.get(cat, 0)}{note} |\n")
    stats.append("\n## Quality-filter drop reasons\n\n| Reason | Count |\n|---|---:|\n")
    for reason, n in drop_reasons.most_common():
        stats.append(f"| {reason} | {n} |\n")
    stats.append(
        f"\n## Leakage\n\n- Dual key applied: `(source_dataset, source_row_id)` + `seed_answer_hash` "
        f"(MENST-only). {len(excluded)} rows excluded, see `leakage_log.csv`.\n"
        f"- Zero rows in `ft_corpus_v1.jsonl` share a `seed_answer_hash` or "
        f"`(source_dataset, source_row_id)` with any of the 90 frozen benchmark seeds.\n"
    )
    STATS_OUT.write_text("".join(stats), encoding="utf-8")
    print(f"\nWrote {len(formatted)} FT pairs -> {FT_CORPUS_OUT}")
    print(f"Wrote leakage log -> {LEAKAGE_LOG_OUT}")
    print(f"Wrote stats -> {STATS_OUT}")


if __name__ == "__main__":
    main()
