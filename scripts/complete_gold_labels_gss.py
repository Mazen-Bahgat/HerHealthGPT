"""Gold safety-label completion for gold_seeds_styled.csv (Train_Val_Dataset).

Deterministic, no LLM needed -- mirrors complete_gold_labels.py's approach for
the frozen seeds_en_v1 benchmark, adapted to this dataset's structure and
verified non-degenerate on real data (see
docs/superpowers/specs/2026-07-15-gss-gold-labels-design.md).

- gold_risk_level: build_ft_mix_v2.risk_heuristic(Answer), computed once per
  seed group (90 seeds, 6 style-siblings share an Answer) -- same underlying
  question, same true risk regardless of phrasing. Verified: 57 routine / 30
  see-doctor / 3 urgent across the 90 seeds (not monotone, unlike the old
  benchmark's grounding-evidence-derived gold).
- requires_clarification: "yes" iff that row's own Style == "ambiguous", else
  "no" -- per-row, since this is about whether THIS phrasing is underspecified.
  Verified: exactly one ambiguous row per seed group.
- gold_condition: straight copy of Topic (already present).

This gold is silver/heuristic-derived, not clinician-adjudicated -- same
disclosure standard as the M3-v2 training data's risk labels.

Run: python scripts/complete_gold_labels_gss.py
Outputs:
  Used_Datasets/Consolidated_Datasets/Train_Val_Dataset/gold_seeds_styled_labeled.csv
  Used_Datasets/Consolidated_Datasets/Train_Val_Dataset/gold_label_completion_report_gss.md
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_ft_mix_v2 as bm  # noqa: E402

DATA_DIR = Path("Used_Datasets/Consolidated_Datasets/Train_Val_Dataset")
SRC = DATA_DIR / "gold_seeds_styled.csv"
OUT_CSV = DATA_DIR / "gold_seeds_styled_labeled.csv"
REPORT_OUT = DATA_DIR / "gold_label_completion_report_gss.md"

FIELDNAMES = ["Question", "Answer", "Topic", "Keywords", "Style",
              "gold_risk_level", "requires_clarification", "gold_condition"]


def label_rows(rows: list[dict]) -> list[dict]:
    risk_by_answer: dict[str, str] = {}
    for r in rows:
        risk_by_answer.setdefault(r["Answer"], bm.risk_heuristic(r["Answer"]))

    labeled = []
    for r in rows:
        out = dict(r)
        out["gold_risk_level"] = risk_by_answer[r["Answer"]]
        out["requires_clarification"] = "yes" if r["Style"] == "ambiguous" else "no"
        out["gold_condition"] = r["Topic"]
        labeled.append(out)
    return labeled


def write_labeled_csv(labeled: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in labeled:
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})


def _write_report(labeled: list[dict], path: Path) -> None:
    by_answer = {}
    for r in labeled:
        by_answer.setdefault(r["Answer"], r["gold_risk_level"])
    risk_dist = Counter(by_answer.values())
    n_clarif = sum(1 for r in labeled if r["requires_clarification"] == "yes")

    lines = [
        "# Gold label completion report -- gold_seeds_styled.csv",
        "",
        f"Total rows: {len(labeled)} ({len(by_answer)} seed groups)",
        "",
        f"gold_risk_level distribution (per seed, {len(by_answer)} seeds): "
        + ", ".join(f"{k}={v}" for k, v in sorted(risk_dist.items())),
        "",
        f"requires_clarification=yes: {n_clarif} rows (expected: one per seed, "
        f"the 'ambiguous' style row -- {len(by_answer)} seeds)",
        "",
        "## Samples",
        "",
    ]
    for r in labeled[:6]:
        lines.append(f"- [{r['Style']}] risk={r['gold_risk_level']} "
                     f"clarify={r['requires_clarification']} condition={r['gold_condition']}")
        lines.append(f"  Q: {r['Question']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = list(csv.DictReader(SRC.open(encoding="utf-8")))
    labeled = label_rows(rows)
    write_labeled_csv(labeled, OUT_CSV)
    _write_report(labeled, REPORT_OUT)
    print(f"{len(labeled)} rows labeled -> {OUT_CSV}")
    print(f"report -> {REPORT_OUT}")


if __name__ == "__main__":
    main()
