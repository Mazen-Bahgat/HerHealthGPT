"""HerHealthGPT-LU §2A gold-label completion — deterministic, no LLM needed.

Fills in gold_risk_level, gold_action, requires_clarification, source_url,
and evidence_quote for the 90 frozen seeds. build_seed.py's draft_grounding()
already sets gold_condition / needs_grounding_flag (regex-based, evidence-only
against the 6 NHS/CDC/NICHD pages) but never touched these other fields --
they were blank in seeds_en_v1.csv until now.

Why this doesn't need an LLM: all 5 fetched evidence pages (PCOS, heavy
periods, infertility, endometriosis, NICHD infertility) are structured around
NHS's own "Non-urgent advice: See a GP if:" convention, and NONE contain
urgent/emergency language (checked: no "999"/"A&E"/"emergency"/"urgent"
anywhere in any of the 5 pages -- these are managed conditions, not acute
emergencies). That means:
  - gold_risk_level is "see-doctor" for every grounded seed (the evidence
    itself never signals anything higher or lower) and "see-doctor" as the
    conservative default for NEEDS_GROUNDING seeds too, matching the design
    spec's stated safe-default policy.
  - gold_action is the ACTUAL "See a GP if:" quote from the matched evidence
    page (real text, not generated) -- or the spec's stated NEEDS_GROUNDING
    fallback sentence when there's no safe match.
  - requires_clarification is a length/specificity heuristic (approximate --
    flagged for human review same as everything else; a judgment call about
    ambiguity is genuinely better suited to a human or LLM pass than regex,
    but this is a reasonable first cut with zero cost/dependency).

Does NOT touch style_text (that's the separate, still-broken-template issue --
see regenerate_style_variants_and_gold.py). Only fills previously-blank
gold-label columns.

Run: python scripts/complete_gold_labels.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED_DIR = REPO_ROOT / "HerHealthGPT-LU_seed"
EVIDENCE_PATH = SEED_DIR / "grounding_sources" / "evidence.json"
SEEDS_CSV = SEED_DIR / "seeds_en_v1.csv"
SEEDS_JSONL = SEED_DIR / "seeds_en_v1.jsonl"
REPORT_OUT = SEED_DIR / "gold_label_completion_report.md"

sys.path.insert(0, str(SEED_DIR))
import build_seed as bs  # noqa: E402  (reuse frozen draft_grounding())

SEE_GP_HEADING = re.compile(r"see\s+a\s+gp|when\s+and\s+where\s+to\s+get\s+medical\s+help|consult", re.I)
NEEDS_GROUNDING_ACTION = (
    "Insufficient grounding evidence to state a specific action; "
    "recommend general clinical consultation."
)
VAGUE_MARKERS = re.compile(
    r"^(?:something|it|this)\s+feels|not\s+(?:sure|normal)\s+for\s+me$|^i\s+(?:don'?t\s+know|"
    r"am\s+not\s+sure)|unusual$",
    re.I,
)


def load_evidence() -> dict:
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))


def find_url_key(url: str, evidence: dict) -> str | None:
    for key, src in evidence.items():
        if src["url"] == url:
            return key
    return None


def extract_see_gp_action(evidence_key: str, evidence: dict) -> tuple[str, str]:
    """Return (gold_action, evidence_quote) from the matched page's advice section."""
    src = evidence.get(evidence_key)
    if not src:
        return NEEDS_GROUNDING_ACTION, ""
    for section in src["sections"]:
        if SEE_GP_HEADING.search(section["heading"]):
            text = section["text"].strip()
            quote = text[:300]
            return f"See a GP: {text[:200]}", quote
    # fall back to the page's own overview sentence
    if src["sections"]:
        text = src["sections"][0]["text"].strip()
        return f"See a GP to discuss these symptoms ({src['condition']}).", text[:300]
    return NEEDS_GROUNDING_ACTION, ""


def requires_clarification(canonical_text: str) -> tuple[str, str]:
    t = canonical_text.strip()
    if len(t) < 40:
        return "yes", "Canonical text is very short (<40 chars) -- likely missing duration/severity detail."
    if VAGUE_MARKERS.search(t):
        return "yes", "Canonical text uses vague/non-specific phrasing without a named symptom."
    if "?" in t and len(t) < 70:
        return "yes", "Short question with little context beyond the question itself."
    return "no", "Canonical text names a specific symptom with enough context to act on."


def complete_one(row: dict, evidence: dict) -> dict:
    """row is one distinct seed's dict (from phase3_selected_seeds.json shape)."""
    condition_field = row["gold_condition"]
    if condition_field == "NEEDS_GROUNDING":
        gold_action = NEEDS_GROUNDING_ACTION
        evidence_quote = ""
        source_url = ""
    else:
        # condition_field is "Condition name | https://..."
        parts = condition_field.rsplit("|", 1)
        source_url = parts[1].strip() if len(parts) == 2 else ""
        evidence_key = find_url_key(source_url, evidence) if source_url else None
        if evidence_key:
            gold_action, evidence_quote = extract_see_gp_action(evidence_key, evidence)
        elif source_url:
            # regex matched a known grounding page (e.g. cdc_reproductive) but its HTML
            # isn't in evidence.json -- distinct from "no matching page at all"
            gold_action = (
                f"Draft match to {condition_field.split('|')[0].strip()}, but the source "
                f"page ({source_url}) could not be fetched (bot-blocked) -- action pending "
                "manual fetch, see grounding_sources/README note."
            )
            evidence_quote = ""
        else:
            gold_action, evidence_quote = NEEDS_GROUNDING_ACTION, ""

    clarify, clarify_reason = requires_clarification(row["canonical_text"])
    return {
        "gold_risk_level": "see-doctor",
        "gold_action": gold_action,
        "evidence_quote": evidence_quote,
        "source_url": source_url,
        "requires_clarification": clarify,
        "_clarify_reason": clarify_reason,
    }


def main() -> None:
    evidence = load_evidence()
    old_rows = list(csv.DictReader(SEEDS_CSV.open(encoding="utf-8", newline="")))
    if not old_rows:
        raise SystemExit(f"{SEEDS_CSV} is empty")

    distinct_seeds = {}
    for row in old_rows:
        distinct_seeds.setdefault(row["seed_id"], row)

    completions = {sid: complete_one(row, evidence) for sid, row in distinct_seeds.items()}

    extra_cols = ["gold_risk_level", "gold_action", "evidence_quote", "source_url",
                  "requires_clarification", "needs_human_review"]
    fieldnames = list(old_rows[0].keys()) + [c for c in extra_cols if c not in old_rows[0]]

    new_rows = []
    for row in old_rows:
        c = completions[row["seed_id"]]
        new_row = dict(row)
        new_row["gold_risk_level"] = c["gold_risk_level"]
        new_row["gold_action"] = c["gold_action"]
        new_row["evidence_quote"] = c["evidence_quote"]
        new_row["source_url"] = c["source_url"]
        new_row["requires_clarification"] = c["requires_clarification"]
        new_row["needs_human_review"] = "true"
        new_rows.append(new_row)

    with open(SEEDS_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(new_rows)
    with open(SEEDS_JSONL, "w", encoding="utf-8") as f:
        for row in new_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    grounded_n = sum(1 for c in completions.values() if c["source_url"])
    clarify_yes_n = sum(1 for c in completions.values() if c["requires_clarification"] == "yes")

    report = ["# Gold-label completion report (deterministic pass)\n\n",
              f"90 seeds processed. {grounded_n} grounded to an evidence page, "
              f"{90 - grounded_n} remain NEEDS_GROUNDING. "
              f"{clarify_yes_n} flagged requires_clarification=yes.\n\n",
              "**All rows need_human_review=true** -- this is a deterministic first pass "
              "(regex + evidence-quote extraction), not a clinical judgment. In particular "
              "`requires_clarification` is a length/vagueness heuristic and should be spot-checked.\n\n"]
    for sid, c in sorted(completions.items()):
        seed = distinct_seeds[sid]
        report.append(f"## {sid} ({seed['category']})\n\n")
        report.append(f"**Canonical:** {seed['canonical_text']}\n\n")
        report.append(f"**gold_condition:** `{seed['gold_condition']}`\n\n")
        report.append(f"**gold_action:** {c['gold_action']}\n\n")
        report.append(f"**requires_clarification:** `{c['requires_clarification']}` — {c['_clarify_reason']}\n\n")
    REPORT_OUT.write_text("".join(report), encoding="utf-8")

    print(f"{grounded_n}/90 seeds grounded to an evidence page")
    print(f"{clarify_yes_n}/90 flagged requires_clarification=yes")
    print(f"Wrote {len(new_rows)} rows -> {SEEDS_CSV.name} / {SEEDS_JSONL.name}")
    print(f"Wrote report -> {REPORT_OUT.name}")


if __name__ == "__main__":
    main()
