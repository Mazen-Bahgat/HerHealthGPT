"""HerHealthGPT-LU §9 item 2 — regenerate broken style variants + complete gold labels.

WHY THIS EXISTS: build_seed.py's generate_styles() is a fixed-template
rewriter (f"I have {claims}." for every menstrual seed, regardless of what
the canonical text actually says). Verified on real rows: menst-001
("How long will I have irregular periods after a miscarriage?") and
menst-002 ("Do irregular periods influence the ability to get pregnant?")
both collapse to the IDENTICAL clinical variant "I have irregular periods."
— the miscarriage context and the fertility question are silently dropped.
This is flagged as a known issue in the design spec (Sec 2A) and was
scheduled for Day-2 regeneration; this script performs that regeneration
plus finishes the gold-label fields the spec's item schema calls for
(gold_risk_level, gold_action, requires_clarification) that build_seed.py's
draft_grounding() never set.

One Claude API call per seed (90 total) does BOTH jobs together, because
gold labels must be written from the canonical text (spec: "Gold labels are
written from the canonical text, not the variant") and the same evidence
context the model needs for grounding is naturally available at the same
call. NO temperature/top_p/top_k (rejected by claude-opus-4-8). Idempotent:
per-seed raw API responses are cached before parsing, so a crash never loses
paid output and a re-run only calls for seeds not yet cached.

Requires ANTHROPIC_API_KEY in .env (see .env.example) or the environment —
not present when this script was written; run once a key is available:

    python scripts/regenerate_style_variants_and_gold.py

Output (all under HerHealthGPT-LU_seed/):
  seeds_en_v1.csv / .jsonl   style_text replaced (non-canonical rows only);
                             + gold_risk_level, gold_action, source_url,
                             requires_clarification, needs_human_review columns
  style_variants.json       regenerated with LLM output + rationale
  regeneration_report.md    old vs new variant per seed, for human review
  regenerate_cache/         raw per-seed API responses (idempotency)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Literal

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED_DIR = REPO_ROOT / "HerHealthGPT-LU_seed"
GROUNDING_DIR = SEED_DIR / "grounding_sources"
CACHE_DIR = SEED_DIR / "regenerate_cache"

PHASE3_SEEDS = SEED_DIR / "phase3_selected_seeds.json"
SEEDS_CSV = SEED_DIR / "seeds_en_v1.csv"
SEEDS_JSONL = SEED_DIR / "seeds_en_v1.jsonl"
STYLE_VARIANTS_OUT = SEED_DIR / "style_variants.json"
REPORT_OUT = SEED_DIR / "regeneration_report.md"

MODEL = "claude-opus-4-8"
STYLES = ("clinical", "layperson", "indirect_cultural", "ambiguous", "emotionally_concerned")

# Which grounding_sources/evidence.json keys are admissible evidence per category.
# Mirrors build_seed.py's draft_grounding() routing. cdc_reproductive is currently
# unavailable (bot-blocked on fetch) -- menstrual seeds needing it (endometriosis/
# fibroid mentions without heavy-bleeding language) correctly stay NEEDS_GROUNDING.
CATEGORY_EVIDENCE_KEYS = {
    "menstrual": ["nhs_heavy_periods"],
    "pcos": ["nhs_pcos_symptoms"],
    "fertility": ["nhs_infertility", "nichd_endo", "nichd_infertility"],
}

RUBRIC = """You are preparing gold-standard data for a published multilingual health-\
communication research benchmark (HerHealthGPT-LU / LUHME 2026). Precision and honesty \
matter more than completeness -- this data will be evaluated against real medical sources.

TASK 1 -- Style variants (rewrite register only, never content):
Given the canonical patient text below, write exactly 5 register variants:
- clinical: how a clinician would phrase the same concern in a chart note
- layperson: everyday non-medical phrasing
- indirect_cultural: indirect/euphemistic phrasing a patient might use out of discomfort or \
cultural norms, without naming the condition
- ambiguous: vague enough that a real triage nurse would need to ask a follow-up question
- emotionally_concerned: same content, visibly worried tone

HARD RULES for every variant:
- Preserve EVERY clinical claim, qualifier, and context in the canonical text -- timing \
("after a miscarriage"), the actual question being asked (e.g. does X affect fertility?), \
severity words, duration, all of it. Losing any of these is a failure.
- Do not invent symptoms, history, or context not present in the canonical text.
- Do not name a specific diagnosis (e.g. "PCOS") unless the canonical text itself already does.
- Each variant must be a different string from the canonical text and from each other variant \
-- do not reuse a fixed phrase across different seeds regardless of their content.
- A factual/informational question (e.g. "Does X affect fertility?") must stay a QUESTION in \
every variant, not collapse into a flat symptom statement.

TASK 2 -- Gold labels (evidence-only, never invent):
Below the canonical text you will find zero or more EVIDENCE PASSAGES from NHS/CDC/NICHD \
pages already approved as grounding sources for this category. Using ONLY those passages:
- gold_condition: the approved page's condition name if (and only if) a passage genuinely \
covers what the canonical text describes. If no passage safely covers it, output exactly \
"NEEDS_GROUNDING" -- do not guess or extrapolate beyond what the evidence says.
- evidence_quote: the exact sentence from the evidence passages that supports gold_condition. \
Empty string if gold_condition is NEEDS_GROUNDING.
- gold_risk_level: "routine" (self-care, no urgency signaled), "see-doctor" (evidence says see \
a GP/doctor but not urgently), or "urgent" (evidence signals urgent/emergency care) -- derived \
from the SAME evidence passage's own urgency language, not from your general medical knowledge. \
If gold_condition is NEEDS_GROUNDING, output "see-doctor" as the safe conservative default.
- gold_action: the recommended action from the evidence passage, paraphrased faithfully. If \
gold_condition is NEEDS_GROUNDING, output "Insufficient grounding evidence to state a specific \
action; recommend general clinical consultation."
- requires_clarification: "yes" if the canonical text is too vague/ambiguous for a clinician to \
act on without follow-up questions (missing duration, severity, or which specific symptom), \
otherwise "no".
- clarification_rationale: one sentence explaining the requires_clarification decision.
"""


class StyleVariants(BaseModel):
    clinical: str
    layperson: str
    indirect_cultural: str
    ambiguous: str
    emotionally_concerned: str


class GoldLabels(BaseModel):
    gold_condition: str
    evidence_quote: str
    gold_risk_level: Literal["routine", "see-doctor", "urgent"]
    gold_action: str
    requires_clarification: Literal["yes", "no"]
    clarification_rationale: str


class SeedRegeneration(BaseModel):
    style_variants: StyleVariants
    gold: GoldLabels


def load_evidence_bundle() -> dict:
    if not GROUNDING_DIR.joinpath("evidence.json").exists():
        raise FileNotFoundError(
            f"{GROUNDING_DIR / 'evidence.json'} missing -- run "
            "scripts/scrape_grounding_sources.py first"
        )
    return json.loads(GROUNDING_DIR.joinpath("evidence.json").read_text(encoding="utf-8"))


def evidence_text_for_category(category: str, evidence: dict) -> str:
    keys = CATEGORY_EVIDENCE_KEYS.get(category, [])
    blocks = []
    for key in keys:
        if key not in evidence:
            continue
        src = evidence[key]
        blocks.append(f"### {src['condition']} ({src['url']})")
        for section in src["sections"]:
            blocks.append(f"[{section['heading']}] {section['text']}")
    if not blocks:
        return "(no evidence passages available for this category)"
    return "\n".join(blocks)


def build_prompt(seed: dict, evidence_text: str) -> str:
    return (
        RUBRIC
        + f"\nCATEGORY: {seed['category']}\n"
        + f"CANONICAL PATIENT TEXT: \"{seed['canonical_text']}\"\n"
        + f"CURRENT DRAFT GOLD_CONDITION (unverified regex guess, may be wrong): "
        + f"{seed['gold_condition']}\n\n"
        + f"EVIDENCE PASSAGES:\n{evidence_text}\n"
    )


def regenerate_one(client: anthropic.Anthropic, seed: dict, evidence: dict) -> SeedRegeneration:
    cache_file = CACHE_DIR / f"{seed['seed_id']}.json"
    if cache_file.exists():
        return SeedRegeneration.model_validate_json(cache_file.read_text(encoding="utf-8"))

    prompt = build_prompt(seed, evidence_text_for_category(seed["category"], evidence))
    response = client.messages.parse(
        model=MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
        output_format=SeedRegeneration,
    )
    result: SeedRegeneration = response.parsed_output
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result


def distinct_string_ratio(variants_by_seed: dict[str, StyleVariants]) -> float:
    all_texts = [v for sv in variants_by_seed.values() for v in sv.model_dump().values()]
    return len(set(all_texts)) / len(all_texts) if all_texts else 0.0


def main() -> None:
    load_dotenv()
    if not PHASE3_SEEDS.exists():
        print(f"ERROR: {PHASE3_SEEDS} not found", file=sys.stderr)
        sys.exit(1)

    seeds = json.loads(PHASE3_SEEDS.read_text(encoding="utf-8"))
    evidence = load_evidence_bundle()
    print(f"{len(seeds)} seeds loaded; regenerating style variants + gold labels...")

    client = anthropic.Anthropic()
    results: dict[str, SeedRegeneration] = {}
    for i, seed in enumerate(seeds, start=1):
        print(f"[{i}/{len(seeds)}] {seed['seed_id']} ({seed['category']})...")
        results[seed["seed_id"]] = regenerate_one(client, seed, evidence)

    variants_only = {sid: r.style_variants for sid, r in results.items()}
    ratio = distinct_string_ratio(variants_only)
    print(f"distinct-string ratio across all regenerated variants: {ratio:.2%}")
    if ratio < 0.9:
        print("WARNING: ratio below 90% -- inspect regeneration_report.md before freezing")

    # --- rewrite seeds_en_v1.csv / .jsonl -----------------------------------
    old_rows = list(csv.DictReader(SEEDS_CSV.open(encoding="utf-8", newline="")))
    seed_meta = {s["seed_id"]: s for s in seeds}
    new_cols = old_rows[0].keys() if old_rows else []
    extra_cols = [
        "gold_risk_level", "gold_action", "requires_clarification",
        "evidence_quote", "needs_human_review",
    ]
    fieldnames = list(new_cols) + [c for c in extra_cols if c not in new_cols]

    new_rows = []
    report_lines = ["# Style-variant regeneration report\n\n"]
    for seed_id, reg in results.items():
        meta = seed_meta[seed_id]
        report_lines.append(f"## {seed_id} ({meta['category']})\n\n")
        report_lines.append(f"**Canonical:** {meta['canonical_text']}\n\n")
        report_lines.append(
            f"**Gold:** `{reg.gold.gold_condition}` | risk=`{reg.gold.gold_risk_level}` | "
            f"clarify=`{reg.gold.requires_clarification}`\n"
            f"> {reg.gold.evidence_quote or '(no evidence quote -- NEEDS_GROUNDING)'}\n\n"
        )
        for old_row in old_rows:
            if old_row["seed_id"] != seed_id:
                continue
            style = old_row["style"]
            new_row = dict(old_row)
            if style != "canonical":
                old_text = old_row["style_text"]
                new_text = getattr(reg.style_variants, style)
                new_row["style_text"] = new_text
                report_lines.append(f"- **{style}**\n  - old: {old_text}\n  - new: {new_text}\n")
            new_row["gold_condition"] = reg.gold.gold_condition
            new_row["needs_grounding_flag"] = str(reg.gold.gold_condition == "NEEDS_GROUNDING")
            new_row["gold_risk_level"] = reg.gold.gold_risk_level
            new_row["gold_action"] = reg.gold.gold_action
            new_row["requires_clarification"] = reg.gold.requires_clarification
            new_row["evidence_quote"] = reg.gold.evidence_quote
            new_row["needs_human_review"] = "true"
            new_rows.append(new_row)
        report_lines.append("\n")

    with open(SEEDS_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(new_rows)
    with open(SEEDS_JSONL, "w", encoding="utf-8") as f:
        for row in new_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    style_variants_out = {
        sid: {**reg.style_variants.model_dump(), "_gold": reg.gold.model_dump()}
        for sid, reg in results.items()
    }
    STYLE_VARIANTS_OUT.write_text(
        json.dumps(style_variants_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    REPORT_OUT.write_text("".join(report_lines), encoding="utf-8")

    needs_grounding_n = sum(1 for r in results.values() if r.gold.gold_condition == "NEEDS_GROUNDING")
    print(f"\nWrote {len(new_rows)} rows -> {SEEDS_CSV.name} / {SEEDS_JSONL.name}")
    print(f"Wrote regenerated variants -> {STYLE_VARIANTS_OUT.name}")
    print(f"Wrote human-review report -> {REPORT_OUT.name}")
    print(f"NEEDS_GROUNDING after LLM pass: {needs_grounding_n}/{len(results)}")
    print("\nNEXT: a human (Hassan/Hana/Mariam) must review regeneration_report.md and each "
          "row's needs_human_review flag before the benchmark freezes -- this is an AI first "
          "pass, not a substitute for the meaning-preservation review the spec requires.")


if __name__ == "__main__":
    main()
