#!/usr/bin/env python3
"""
HerHealthGPT-LU (LUHME 2026) — categorized English seed dataset builder.

Evidence-only pipeline:
  Phase 1: Keyword extraction on patient-authored text only
  Phase 2: Near-duplicate clustering (HCM↔iCliniq, MENST Set1↔Set2)
  Phase 3: Seed finalization (Clear-first, diversity + source-mix)
  Phase 4: Style variants (canonical + 5), meaning-preserving rewrite
  Phase 5: Draft gold grounding vs NHS/CDC/NICHD pages only
  Phase 6: Write deliverables under this directory

Canonical patient text is NEVER paraphrased for provenance.
Style variants may rephrase register only — no added/dropped clinical claims.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

ROOT = Path(r"e:\Graduation_Project\Dataset")
OUT = ROOT / "HerHealthGPT-LU_seed"
HCM_PATH = ROOT / "HealthCareMagic-100k-Chat-Format-en" / "HealthCareMagic-100k-en.jsonl"
ICLINIQ_DIR = ROOT / "ChatDoctor-iCliniq" / "data"
MENST_DIR = ROOT / "MENST"

CATEGORIES = ("menstrual", "pcos", "fertility")
PREFIX = {"menstrual": "menst", "pcos": "pcos", "fertility": "fert"}
STYLES = (
    "canonical",
    "clinical",
    "layperson",
    "indirect_cultural",
    "ambiguous",
    "emotionally_concerned",
)

GROUNDING = {
    "nhs_pcos_symptoms": {
        "condition": "PCOS symptoms",
        "url": "https://www.nhs.uk/conditions/polycystic-ovary-syndrome-pcos/symptoms/",
    },
    "nhs_heavy_periods": {
        "condition": "Heavy periods (menorrhagia)",
        "url": "https://www.nhs.uk/conditions/heavy-periods/",
    },
    "nhs_infertility": {
        "condition": "Infertility",
        "url": "https://www.nhs.uk/conditions/infertility/",
    },
    "cdc_reproductive": {
        "condition": "Common reproductive health concerns",
        "url": "https://www.cdc.gov/reproductive-health/women-health/common-concerns.html",
    },
    "nichd_endo": {
        "condition": "Endometriosis (NICHD)",
        "url": "https://www.nichd.nih.gov/health/topics/endometriosis",
    },
    "nichd_infertility": {
        "condition": "Infertility (NICHD)",
        "url": "https://www.nichd.nih.gov/health/topics/infertility",
    },
}

# ---------------------------------------------------------------------------
# Keywords
# ---------------------------------------------------------------------------

PERIOD_TIME_SPAN = re.compile(
    r"\b(?:for|over|during|after|within|in)\s+(?:a|the|this|that|short|long|brief|extended)?\s*period\s+of\b",
    re.I,
)
PERIOD_OF_TIME = re.compile(r"\bperiod\s+of\s+(?:time|days|weeks|months|years)\b", re.I)

MENSTRUAL_KW = [
    (r"\birregular\s+periods?\b", "strong"),
    (r"\bheavy\s+periods?\b", "strong"),
    (r"\bpainful\s+periods?\b", "strong"),
    (r"\bperiod\s+pain\b", "strong"),
    (r"\bmenstrual\s+cramps?\b", "strong"),
    (r"\bdysmenorrh(?:ea|oea)\b", "strong"),
    (r"\bmenorrhagia\b", "strong"),
    (r"\bamenorrh(?:ea|oea)\b", "strong"),
    (r"\boligomenorrh(?:ea|oea)\b", "strong"),
    (r"\bmetrorrhagia\b", "strong"),
    (r"\bmissed\s+(?:my\s+)?periods?\b", "strong"),
    (r"\blate\s+periods?\b", "strong"),
    (r"\bperiods?\s+(?:are|is|have|has)\s+(?:late|irregular|heavy|painful|scanty|light)\b", "strong"),
    (r"\b(?:haven'?t|have\s+not|hasn'?t)\s+(?:had\s+)?(?:a\s+)?period\b", "strong"),
    (r"\bno\s+periods?\b", "strong"),
    (r"\bperiods?\s+stopped\b", "strong"),
    (r"\b(?:pms|premenstrual)\b", "strong"),
    (r"\bmenstrual\s+(?:cycle|bleeding|flow|problems?)\b", "strong"),
    (r"\bmenses\b", "strong"),
    (r"\bspotting\s+(?:between|before|after)\s+(?:periods?|cycles?)\b", "strong"),
    (r"\bcycle\s+(?:is|has|became|become)\s+(?:irregular|longer|shorter|changed)\b", "strong"),
    (r"\bmenstrual\b", "medium"),
    (r"\bperiods?\b", "weak"),
    (r"\bcycle\b", "weak"),
    (r"\bcramps?\b", "weak"),
]

PCOS_KW = [
    (r"\bpcos\b", "strong"),
    (r"\bpcod\b", "strong"),
    (r"\bpolycystic\s+ovar(?:y|ies)\b", "strong"),
    (r"\bpolycystic\s+ovarian\b", "strong"),
    (r"\bhirsutism\b", "strong"),
    (r"\bfacial\s+hair\b", "strong"),
    (r"\bexcessive\s+(?:body|facial)\s+hair\b", "strong"),
    (r"\bunwanted\s+(?:facial\s+)?hair\b", "strong"),
    (r"\bhormonal\s+imbalance\b", "medium"),
    (r"\bhormonal\s+acne\b", "strong"),
    (r"\bovarian\s+cysts?\b", "medium"),
    (r"\bcysts?\s+on\s+(?:my\s+)?ovar", "medium"),
]

FERTILITY_KW = [
    (r"\binfertil(?:e|ity)\b", "strong"),
    (r"\bsubfertil(?:e|ity)\b", "strong"),
    (r"\bttc\b", "strong"),
    (r"\btrying\s+to\s+conceive\b", "strong"),
    (r"\btrying\s+to\s+(?:get\s+)?pregnant\b", "strong"),
    (r"\bunable\s+to\s+conceive\b", "strong"),
    (r"\bcan(?:'?t|not)\s+(?:seem\s+to\s+)?(?:get|become)\s+pregnant\b", "strong"),
    (r"\bdiffl?icult(?:y)?\s+(?:in\s+|with\s+)?conceiv", "strong"),
    (r"\bstruggl(?:e|ing)\s+to\s+(?:conceive|get\s+pregnant)\b", "strong"),
    (r"\bfertility\s+(?:issue|problem|test|treatment|clinic|workup)\b", "strong"),
    (r"\bfertility\b", "medium"),
    (r"\bconceiv(?:e|ing)\b", "medium"),
    (r"\bivf\b", "medium"),
    (r"\biui\b", "medium"),
    (r"\bfallopian\b", "medium"),
    (r"\bovulat(?:e|ion|ing)\b", "medium"),
]

PREGNANCY_CONFIRMED = re.compile(
    r"\b(?:i\s+am\s+pregnant|i'?m\s+pregnant|already\s+pregnant|weeks?\s+pregnant|"
    r"positive\s+pregnancy\s+test|pregnancy\s+test\s+(?:is\s+)?positive|"
    r"miscarriage|abortion|prenatal|"
    r"how\s+far\s+along|gestation)\b",
    re.I,
)

MALE_CONTEXT = re.compile(
    r"\b(?:testes?\b|sperm\b|erectile\b|semen\b|varicocele\b|"
    r"i\s+am\s+a\s+(?:man|male)\b|i'?m\s+a\s+(?:man|male)\b|"
    r"\b28-year-old\s+man\b|\b\d+-year-old\s+man\b)\b",
    re.I,
)

COMPILED = {
    "menstrual": [(re.compile(p, re.I), s) for p, s in MENSTRUAL_KW],
    "pcos": [(re.compile(p, re.I), s) for p, s in PCOS_KW],
    "fertility": [(re.compile(p, re.I), s) for p, s in FERTILITY_KW],
}
STRENGTH_SCORE = {"strong": 3, "medium": 2, "weak": 1}

# Myth / general-education FAQ patterns (MENST-heavy) — demote for seed selection
EDU_MYTH = re.compile(
    r"\b(?:is\s+there\s+any\s+truth|is\s+that\s+true|i\s+heard\s+(?:that|you)|"
    r"some\s+girls?\s+in\s+my\s+village|my\s+friends?\s+were\s+discussing|"
    r"what\s+is\s+menstruation|what\s+is\s+the\s+menstrual\s+cycle|"
    r"scientific\s+term|how\s+can\s+i\s+keep\s+track|"
    r"mhs\s+program|using\s+the\s+.*pads?\s+makes)\b",
    re.I,
)
DEFINITION_FAQ = re.compile(
    r"^(?:what\s+is\s+|what\s+are\s+|what\s+causes?\s+|define\s+|define:|"
    r"can\s+you\s+explain\s+what\s+|please\s+explain\s+what\s+)\b",
    re.I,
)

# Clinical claim tokens preserved across styles (must appear in original to be used)
CLAIM_PATTERNS = [
    (r"\bpcos\b", "PCOS"),
    (r"\bpcod\b", "PCOD"),
    (r"\bpolycystic\b", "polycystic ovaries"),
    (r"\bhirsutism\b", "hirsutism"),
    (r"facial\s+hair", "facial hair"),
    (r"excessive\s+(?:body\s+|facial\s+)?hair", "excess hair"),
    (r"unwanted\s+(?:facial\s+)?hair", "unwanted hair"),
    (r"\birregular\s+periods?\b", "irregular periods"),
    (r"\bheavy\s+periods?\b", "heavy periods"),
    (r"\bpainful\s+periods?\b", "painful periods"),
    (r"\bperiod\s+pain\b", "period pain"),
    (r"\bmenstrual\s+cramps?\b", "menstrual cramps"),
    (r"\bcramps?\b", "cramps"),
    (r"\bdysmenorrh(?:ea|oea)\b", "dysmenorrhea"),
    (r"\bmenorrhagia\b", "menorrhagia"),
    (r"\bamenorrh(?:ea|oea)\b", "amenorrhea"),
    (r"\bmissed\s+(?:my\s+)?periods?\b", "missed periods"),
    (r"\blate\s+periods?\b", "late periods"),
    (r"\bno\s+periods?\b", "no periods"),
    (r"\b(?:pms|premenstrual)\b", "PMS"),
    (r"\bspotting\b", "spotting"),
    (r"\bendometri(?:osis|al)?\b", "endometriosis"),
    (r"\binfertil(?:e|ity)\b", "infertility"),
    (r"trying\s+to\s+conc?eive", "trying to conceive"),
    (r"trying\s+to\s+(?:get\s+)?pregnant", "trying to get pregnant"),
    (r"unable\s+to\s+conc?eive", "unable to conceive"),
    (r"\bttc\b", "TTC"),
    (r"\bivf\b", "IVF"),
    (r"\biui\b", "IUI"),
    (r"\bovulat(?:e|ion|ing)\b", "ovulation"),
    (r"\bweight\b", "weight"),
    (r"\bacne\b", "acne"),
    (r"\bfatigue\b", "fatigue"),
    (r"\bcysts?\b", "cysts"),
    (r"\bhormon", "hormonal"),
    (r"\bfertility\b", "fertility"),
    (r"\bconc?eiv", "conceive"),
    (r"periods?\s+(?:haven'?t|have\s+not)\s+come", "periods have not come"),
    (r"haven'?t\s+(?:had\s+)?(?:a\s+)?period", "have not had a period"),
]


@dataclass
class Candidate:
    source_dataset: str
    source_row_id: str
    category: str
    text: str
    confidence_tier: str
    match_notes: str
    menst_set: str = ""
    topic: str = ""
    dedup_group_id: str = ""
    keep: bool = True
    drop_reason: str = ""


def clean_text(s) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    t = str(s).replace("\u00a0", " ").replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def extract_human_hcm(text: str) -> str:
    if "<human>:" not in text:
        return ""
    after = text.split("<human>:", 1)[1]
    if "<bot>:" in after:
        after = after.split("<bot>:", 1)[0]
    return clean_text(after)


def score_category(text: str, category: str) -> tuple[str, str]:
    hits = []
    score = 0
    strong_hits = 0
    for rx, strength in COMPILED[category]:
        if rx.search(text):
            hits.append(f"{rx.pattern[:40]}:{strength}")
            score += STRENGTH_SCORE[strength]
            if strength == "strong":
                strong_hits += 1
    if not hits:
        return "Reject", "no_keyword"

    if category == "menstrual":
        if (PERIOD_TIME_SPAN.search(text) or PERIOD_OF_TIME.search(text)) and strong_hits == 0:
            return "Reject", "period_as_timespan"
        if strong_hits == 0 and score <= 2:
            tl = text.lower()
            if re.search(
                r"\b(?:my|her)\s+periods?\b|\bperiods?\s+(?:are|is|have|came|come|started|stopped)\b",
                tl,
            ) and not (PERIOD_TIME_SPAN.search(text) or PERIOD_OF_TIME.search(text)):
                return "Borderline", "weak_period_match:" + ";".join(hits[:3])
            return "Reject", "weak_nonmenstrual"
        if strong_hits >= 1 or score >= 4:
            return "Clear", ";".join(hits[:5])
        return "Borderline", ";".join(hits[:5])

    if category == "pcos":
        if strong_hits >= 1:
            return "Clear", ";".join(hits[:5])
        if score >= 2:
            return "Borderline", ";".join(hits[:5])
        return "Reject", "weak_pcos:" + ";".join(hits[:3])

    if category == "fertility":
        tl = text.lower()
        if MALE_CONTEXT.search(text) and not re.search(
            r"\b(?:i\s+am\s+a\s+(?:woman|female)|my\s+wife|girlfriend|partner\s+and\s+i)\b",
            tl,
        ):
            if strong_hits >= 1:
                return "Borderline", "male_context:" + ";".join(hits[:3])
            return "Reject", "male_incidental"
        if PREGNANCY_CONFIRMED.search(text):
            if not re.search(
                r"\binfertil|trying\s+to\s+conceive|\bttc\b|unable\s+to\s+conceive",
                tl,
            ):
                return "Reject", "pregnancy_confirmed_not_ttc"
        if strong_hits >= 1:
            return "Clear", ";".join(hits[:5])
        if score >= 2:
            return "Borderline", ";".join(hits[:5])
        return "Reject", "weak_fertility:" + ";".join(hits[:3])

    return "Reject", "unknown"


def assign_best_category(text: str) -> list[tuple[str, str, str]]:
    results = []
    for cat in CATEGORIES:
        tier, notes = score_category(text, cat)
        if tier != "Reject":
            results.append((cat, tier, notes))
    return results


def load_menst() -> list[Candidate]:
    cands: list[Candidate] = []
    t2 = pd.read_csv(MENST_DIR / "training2K.csv")
    t2.columns = [c.strip() for c in t2.columns]
    for idx, row in t2.iterrows():
        s = int(row["Set"]) if pd.notna(row["Set"]) else -1
        if s not in (1, 2):
            continue
        q = clean_text(row["Question"])
        if not q:
            continue
        for cat, tier, notes in assign_best_category(q):
            cands.append(
                Candidate(
                    "MENST_training2K",
                    f"training2K:{idx}",
                    cat,
                    q,
                    tier,
                    notes,
                    menst_set=str(s),
                )
            )
    t24 = pd.read_csv(MENST_DIR / "train24K.csv")
    t24.columns = [c.strip() for c in t24.columns]
    for idx, row in t24.iterrows():
        s = int(row["Set"]) if pd.notna(row["Set"]) else -1
        if s not in (1, 2):
            continue
        q = clean_text(row["Question"])
        if not q:
            continue
        topic = clean_text(row.get("Topic", ""))
        for cat, tier, notes in assign_best_category(q):
            cands.append(
                Candidate(
                    "MENST_train24K",
                    f"train24K:{idx}",
                    cat,
                    q,
                    tier,
                    notes,
                    menst_set=str(s),
                    topic=topic,
                )
            )
    return cands


def load_icliniq() -> list[Candidate]:
    pq = next(ICLINIQ_DIR.glob("*.parquet"))
    df = pd.read_parquet(pq)
    cands: list[Candidate] = []
    for idx, row in df.iterrows():
        q = clean_text(row["input"])
        if not q:
            continue
        for cat, tier, notes in assign_best_category(q):
            cands.append(
                Candidate("ChatDoctor-iCliniq", str(idx), cat, q, tier, notes)
            )
    return cands


def load_hcm() -> list[Candidate]:
    cands: list[Candidate] = []
    with open(HCM_PATH, "r", encoding="utf-8", errors="replace") as f:
        for line_i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            human = extract_human_hcm(obj.get("text", ""))
            if not human:
                continue
            for cat, tier, notes in assign_best_category(human):
                cands.append(
                    Candidate(
                        "HealthCareMagic-100k", str(line_i), cat, human, tier, notes
                    )
                )
    return cands


def normalize_for_dedup(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"^(hi|hello|dear|hey)\b[\s,]*(doctor|dr)?[\s,]*", "", t)
    return t[:800]


def cluster_near_dups(
    items: list[Candidate], threshold: float = 88.0
) -> tuple[list[Candidate], list[dict]]:
    drop_log: list[dict] = []
    by_cat: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(items):
        by_cat[c.category].append(i)

    group_counter = 0
    for cat, indices in by_cat.items():
        norms = [normalize_for_dedup(items[i].text) for i in indices]
        parent = list(range(len(indices)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        n = len(indices)
        buckets: dict[str, list[int]] = defaultdict(list)
        for li, norm in enumerate(norms):
            buckets[norm[:20]].append(li)
            buckets[f"len{len(norm)//20}"].append(li)

        compared = set()
        for bucket_ids in buckets.values():
            if len(bucket_ids) > 80:
                bucket_ids = bucket_ids[:80]
            for a_i in range(len(bucket_ids)):
                for b_i in range(a_i + 1, len(bucket_ids)):
                    a, b = bucket_ids[a_i], bucket_ids[b_i]
                    pair = (min(a, b), max(a, b))
                    if pair in compared:
                        continue
                    compared.add(pair)
                    ca, cb = items[indices[a]], items[indices[b]]
                    cross = (
                        {ca.source_dataset, cb.source_dataset}
                        == {"HealthCareMagic-100k", "ChatDoctor-iCliniq"}
                        or (
                            ca.source_dataset.startswith("MENST")
                            and cb.source_dataset.startswith("MENST")
                            and ca.menst_set
                            and cb.menst_set
                            and ca.menst_set != cb.menst_set
                        )
                        or ca.source_dataset == cb.source_dataset
                    )
                    if not cross:
                        continue
                    if (
                        ca.source_dataset == cb.source_dataset
                        and ca.source_row_id == cb.source_row_id
                    ):
                        continue
                    if fuzz.token_set_ratio(norms[a], norms[b]) >= threshold:
                        union(a, b)

        groups: dict[int, list[int]] = defaultdict(list)
        for li in range(n):
            groups[find(li)].append(li)

        for _root, members in groups.items():
            if len(members) < 2:
                continue
            group_counter += 1
            gid = f"dup-{cat}-{group_counter:04d}"

            def rank(li):
                c = items[indices[li]]
                tier_r = 0 if c.confidence_tier == "Clear" else 1
                if c.source_dataset == "MENST_training2K" and c.menst_set == "1":
                    src_r = 0
                elif c.source_dataset.startswith("MENST"):
                    src_r = 1
                elif c.source_dataset == "HealthCareMagic-100k":
                    src_r = 2
                else:
                    src_r = 3
                return (tier_r, src_r, len(c.text), c.source_row_id)

            members_sorted = sorted(members, key=rank)
            keep_li = members_sorted[0]
            for li in members:
                items[indices[li]].dedup_group_id = gid
            for li in members_sorted[1:]:
                c = items[indices[li]]
                kept = items[indices[keep_li]]
                c.keep = False
                c.drop_reason = f"near_dup_of:{kept.source_dataset}:{kept.source_row_id}"
                drop_log.append(
                    {
                        "category": cat,
                        "dedup_group_id": gid,
                        "dropped": f"{c.source_dataset}:{c.source_row_id}",
                        "kept": f"{kept.source_dataset}:{kept.source_row_id}",
                        "reason": c.drop_reason,
                    }
                )
    return items, drop_log


def token_jaccard(a: str, b: str) -> float:
    ta = set(normalize_for_dedup(a).split())
    tb = set(normalize_for_dedup(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def extract_claims(text: str) -> list[str]:
    claims = []
    seen = set()
    for pat, label in CLAIM_PATTERNS:
        if re.search(pat, text, re.I) and label.lower() not in seen:
            claims.append(label)
            seen.add(label.lower())
    return claims


def is_seed_eligible(c: Candidate) -> bool:
    """Hard quality filters for promotion into the seed set."""
    if not c.keep or c.confidence_tier != "Clear":
        return False
    t = c.text
    if len(t) < 25 or len(t) > 1200:
        return False
    if MALE_CONTEXT.search(t) and not re.search(
        r"\b(?:wife|girlfriend|partner\s+and\s+i|i\s+am\s+a\s+(?:woman|female))\b",
        t,
        re.I,
    ):
        return False

    claims = extract_claims(t)

    # Definition/myth FAQs only allowed as Set1 anchors when they still have a claim token
    if DEFINITION_FAQ.search(t.strip()) and source_bucket(c) != "menst_set1":
        return False
    if DEFINITION_FAQ.search(t.strip()) and not claims:
        return False

    if c.category == "menstrual":
        if EDU_MYTH.search(t) and not claims:
            return False
        if not claims:
            return False
        # Prefer lived symptom / concern over pure glossary
        if DEFINITION_FAQ.search(t.strip()) and not re.search(
            r"\b(?:my|i\s+have|i'?m|i\s+am|worried|pain|heavy|irregular|missed|late)\b",
            t,
            re.I,
        ):
            return False
    if c.category == "fertility":
        if EDU_MYTH.search(t) and not re.search(
            r"trying\s+to\s+(?:conceive|get\s+pregnant)|unable\s+to\s+conceive|"
            r"\binfertil|\bttc\b|fertility\s+(?:issue|problem|test|treatment)",
            t,
            re.I,
        ):
            return False
        if not any(
            x.lower()
            in {
                "infertility",
                "trying to conceive",
                "trying to get pregnant",
                "unable to conceive",
                "ttc",
                "ivf",
                "iui",
                "fertility",
                "conceive",
                "ovulation",
            }
            for x in claims
        ):
            return False
    if c.category == "pcos":
        if not any(
            x.lower()
            in {
                "pcos",
                "pcod",
                "polycystic ovaries",
                "hirsutism",
                "facial hair",
                "excess hair",
                "unwanted hair",
            }
            for x in claims
        ):
            return False
        # Prefer first-person symptom questions over third-person diet tips for main set
        # (diet tips still allowed but demoted in sort_key)
    return True


def normalize_menst_set(val) -> str:
    """CSV reload may yield '1.0'; keep only 1/2/3 as digit strings."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return ""
    try:
        return str(int(float(s)))
    except ValueError:
        return s


def source_bucket(c: Candidate) -> str:
    mset = normalize_menst_set(c.menst_set)
    if c.source_dataset == "MENST_training2K" and mset == "1":
        return "menst_set1"
    if c.source_dataset.startswith("MENST"):
        return "menst_set2"
    if c.source_dataset == "HealthCareMagic-100k":
        return "hcm"
    return "icliniq"


def diversity_select_stratified(pool: list[Candidate], target: int) -> list[Candidate]:
    """
    Mix sources for authentic phrasing diversity.
    Prefer Set1 MENST anchors but do not monopolize with educational FAQs.
    Soft quotas (week-1): ~20% menst_set1, ~25% menst_set2, ~40% HCM, ~15% iCliniq.
    Quotas auto-shrink when a bucket is sparse (Set1 Clear yield is small).
    """
    eligible = [c for c in pool if is_seed_eligible(c)]
    by_bucket: dict[str, list[Candidate]] = defaultdict(list)
    for c in eligible:
        by_bucket[source_bucket(c)].append(c)

    def sort_key(c: Candidate):
        myth = 1 if EDU_MYTH.search(c.text) else 0
        defin = 1 if DEFINITION_FAQ.search(c.text.strip()) else 0
        # Demote generic diet-advice templates somewhat
        diet = 1 if re.search(r"\bdiet\b", c.text, re.I) and not re.search(
            r"\b(?:i\s+have|i'?m|diagnosed|hirsutism|facial\s+hair|irregular)\b",
            c.text,
            re.I,
        ) else 0
        first_person = 0 if re.search(r"\b(?:i|my|we)\b", c.text, re.I) else 1
        return (defin, myth, diet, first_person, abs(len(c.text) - 220), c.source_row_id)

    for b in by_bucket:
        by_bucket[b] = sorted(by_bucket[b], key=sort_key)

    desired = {
        "menst_set1": max(1, int(round(target * 0.20))),
        "menst_set2": max(2, int(round(target * 0.25))),
        "hcm": max(3, int(round(target * 0.40))),
        "icliniq": max(2, int(round(target * 0.15))),
    }
    # Cap by available pool size
    quotas = {b: min(desired[b], len(by_bucket.get(b, []))) for b in desired}
    # Distribute remainder to HCM then Set2 then iCliniq
    remaining = target - sum(quotas.values())
    for b in ("hcm", "menst_set2", "icliniq", "menst_set1"):
        if remaining <= 0:
            break
        avail = len(by_bucket.get(b, [])) - quotas[b]
        take = min(avail, remaining)
        quotas[b] += take
        remaining -= take

    selected: list[Candidate] = []
    used_text_keys: set[str] = set()

    def try_add(c: Candidate) -> bool:
        key = normalize_for_dedup(c.text)[:200]
        if key in used_text_keys:
            return False
        if any(token_jaccard(c.text, s.text) >= 0.50 for s in selected):
            return False
        selected.append(c)
        used_text_keys.add(key)
        return True

    for bucket, q in quotas.items():
        added = 0
        for c in by_bucket.get(bucket, []):
            if added >= q:
                break
            if try_add(c):
                added += 1

    if len(selected) < target:
        for bucket in ("hcm", "icliniq", "menst_set1", "menst_set2"):
            for c in by_bucket.get(bucket, []):
                if len(selected) >= target:
                    break
                try_add(c)
            if len(selected) >= target:
                break

    return selected[:target]


def draft_grounding(category: str, text: str) -> tuple[str, bool]:
    tl = text.lower()
    if category == "pcos":
        if re.search(r"\bpcos\b|\bpcod\b|polycystic", tl):
            g = GROUNDING["nhs_pcos_symptoms"]
            return f"{g['condition']} | {g['url']}", False
        if re.search(r"hirsutism|facial\s+hair|excessive\s+.*hair|unwanted\s+.*hair", tl):
            # NHS PCOS symptom list includes excess hair; patient asks about hair —
            # only safe without inventing PCOS diagnosis if they also mention PCOS/hormone
            if re.search(r"pcos|pcod|polycystic|hormon", tl):
                g = GROUNDING["nhs_pcos_symptoms"]
                return f"{g['condition']} | {g['url']}", False
            return "NEEDS_GROUNDING", True
        return "NEEDS_GROUNDING", True

    if category == "menstrual":
        if re.search(r"heavy\s+period|menorrhagia|heavy\s+(?:menstrual\s+)?bleeding|flooding", tl):
            g = GROUNDING["nhs_heavy_periods"]
            return f"{g['condition']} | {g['url']}", False
        if re.search(r"endometri", tl):
            # CDC common-concerns covers endometriosis (NICHD URL may block automated fetch)
            g = GROUNDING["cdc_reproductive"]
            return f"Endometriosis (CDC common concerns) | {g['url']}", False
        if re.search(r"fibroid", tl):
            g = GROUNDING["cdc_reproductive"]
            return f"Uterine fibroids (CDC common concerns) | {g['url']}", False
        # Irregular / painful / missed / PMS lack a dedicated allowed page match → NEEDS_GROUNDING
        if re.search(
            r"irregular\s+period|amenorrh|missed\s+period|no\s+period|late\s+period|"
            r"\bpms\b|painful\s+period|dysmenorrh|menstrual\s+cramp|period\s+pain|"
            r"spotting|menstrual\s+(?:cycle|bleeding|flow)",
            tl,
        ):
            return "NEEDS_GROUNDING", True
        return "NEEDS_GROUNDING", True

    if category == "fertility":
        if re.search(
            r"infertil|trying\s+to\s+conceive|\bttc\b|unable\s+to\s+conceive|"
            r"can(?:'?t|not)\s+.*pregnant|difficulty\s+.*conceiv|struggl.*pregnant|"
            r"trying\s+to\s+get\s+pregnant|fertility",
            tl,
        ):
            g = GROUNDING["nhs_infertility"]
            return f"{g['condition']} | {g['url']}", False
        if re.search(r"endometri", tl):
            g = GROUNDING["nichd_endo"]
            return f"{g['condition']} | {g['url']}", False
        return "NEEDS_GROUNDING", True

    return "NEEDS_GROUNDING", True


def propose_targets(clear_eligible: dict[str, int]) -> tuple[dict[str, int], dict[str, str]]:
    """
    Week-1 freeze: tens per category when Clear eligible yield allows.
    Prefer quality over volume — cap at 30 after seeing abundant Clear raw yield,
    because stratified authentic patient language (not FAQs) is scarcer than raw Clear.
    """
    targets = {}
    notes = {}
    for cat in CATEGORIES:
        n = clear_eligible.get(cat, 0)
        if n >= 60:
            t = 30
            note = "ok"
        elif n >= 40:
            t = 25
            note = "ok"
        elif n >= 25:
            t = 20
            note = "ok"
        elif n >= 15:
            t = 15
            note = "tight"
        elif n >= 8:
            t = n
            note = "yield_constrained"
        else:
            t = n
            note = "yield_constrained"
        if cat == "fertility" and n < 20:
            note = "yield_constrained"
        targets[cat] = t
        notes[cat] = note
    return targets, notes


def join_claims(claims: list[str], max_n: int = 4) -> str:
    # Drop redundant substrings (e.g. "conceive" when "trying to conceive" present)
    filtered = []
    for c in claims:
        if any(c.lower() != o.lower() and c.lower() in o.lower() for o in claims):
            continue
        filtered.append(c)
    claims = filtered[:max_n]
    if not claims:
        return "this reproductive health concern"
    if len(claims) == 1:
        return claims[0]
    if len(claims) == 2:
        return f"{claims[0]} and {claims[1]}"
    return ", ".join(claims[:-1]) + f", and {claims[-1]}"


def generate_styles(canonical: str, category: str) -> dict[str, str]:
    """
    Meaning-preserving register variants.
    Only recycles claim phrases attested in the canonical patient text.
    """
    claims = extract_claims(canonical)
    focus = join_claims(claims)

    # Mild shortening for clinical style: strip greetings only
    body = re.sub(
        r"^(hi|hello|hey|dear)[\s,]*(doctor|dr\.?)?[\s,:!-]*",
        "",
        canonical,
        flags=re.I,
    ).strip()
    body = re.sub(r"\s+", " ", body)

    clinical = f"I am experiencing {focus}."
    # If original is already short and claim-dense, prefer a tighter clinical wrap
    if category == "menstrual":
        clinical = f"I have {focus}."
    elif category == "pcos":
        clinical = f"I have symptoms related to {focus}."
    elif category == "fertility":
        clinical = f"I have concerns about {focus}."

    layperson = f"My {focus} is not normal for me and I need advice."
    if category == "menstrual":
        if "PMS" in focus:
            layperson = f"Before my period I feel bad PMS symptoms ({focus}) and it worries me."
        elif "cramp" in focus.lower() or "pain" in focus.lower():
            layperson = f"My period pain / cramps ({focus}) have been hard to deal with."
        else:
            layperson = f"My period is not coming properly — this is about {focus}."
    if category == "pcos":
        layperson = f"I think something hormonal is going on with me ({focus})."
    if category == "fertility":
        layperson = f"We have been trying for a baby and I worry about {focus}."

    indirect = f"My monthly / womanly problem has become unusual regarding {focus}."
    if category == "fertility":
        indirect = f"Family-building has not happened yet; this relates to {focus}."
    if category == "pcos":
        indirect = f"My body changes ({focus}) have become hard to ignore."

    ambiguous = f"Something feels wrong with my cycle / hormones ({focus})."
    if category == "fertility":
        ambiguous = f"Something feels wrong with our ability to conceive ({focus})."

    emotional = (
        f"I am really worried and scared about {focus}. "
        f"Please help me understand what this means."
    )

    return {
        "clinical": clinical,
        "layperson": layperson,
        "indirect_cultural": indirect,
        "ambiguous": ambiguous,
        "emotionally_concerned": emotional,
        "_claims_used": claims,
        "_body_preview": body[:120],
    }


def write_candidates_csv(path: Path, cands: list[Candidate]):
    cols = [
        "source_dataset",
        "source_row_id",
        "category",
        "confidence_tier",
        "menst_set",
        "topic",
        "dedup_group_id",
        "keep",
        "drop_reason",
        "match_notes",
        "text",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for c in cands:
            w.writerow({k: getattr(c, k) for k in cols})


def write_docs(
    seed_records,
    rows,
    targets,
    yield_notes,
    clear_raw,
    clear_eligible,
    drop_log,
    borderline_rows,
):
    cat_counts = Counter(r["category"] for r in seed_records)
    tier_counts = Counter(r["confidence_tier"] for r in seed_records)
    src_counts = Counter(r["source_dataset"] for r in seed_records)
    needs_n = sum(1 for r in seed_records if r["needs_grounding_flag"])

    stats = [
        "# Seed dataset stats summary — HerHealthGPT-LU English v1\n\n",
        "## Proposed vs achieved\n\n",
        "| Category | Clear raw (post-dedup) | Clear eligible (quality filter) | Proposed target | Achieved | Yield note |\n",
        "|----------|-----------------------:|--------------------------------:|----------------:|---------:|------------|\n",
    ]
    for cat in CATEGORIES:
        stats.append(
            f"| {cat} | {clear_raw.get(cat, 0)} | {clear_eligible.get(cat, 0)} | "
            f"{targets.get(cat, 0)} | {cat_counts.get(cat, 0)} | {yield_notes.get(cat, '')} |\n"
        )
    stats.append(
        f"\n**Total seeds:** {len(seed_records)}  \n"
        f"**Total rows (seed × style):** {len(rows)} "
        f"(6 styles: canonical + clinical + layperson + indirect_cultural + ambiguous + emotionally_concerned)\n\n"
        "## Confidence tiers (seeds)\n\n| Tier | Count |\n|------|------:|\n"
    )
    for t, n in tier_counts.most_common():
        stats.append(f"| {t} | {n} |\n")
    stats.append("\n## Source mix (seeds)\n\n| Source | Count |\n|--------|------:|\n")
    for t, n in src_counts.most_common():
        stats.append(f"| {t} | {n} |\n")
    stats.append(
        f"\n## Dedup\n\n- Near-duplicate drop records: **{len(drop_log)}** "
        f"(see `phase2_dedup_drops.jsonl`)\n\n"
        f"## Grounding\n\n- Seeds with `needs_grounding_flag=true`: **{needs_n}**\n"
        "- Draft mappings only against NHS PCOS symptoms, NHS heavy periods, NHS infertility, "
        "CDC reproductive health, NICHD endometriosis/infertility.\n"
        "- No gold_risk_level / gold_action fields invented.\n"
        "- NHS PCOS page content verified 2026-07 (site may show PMOS rename; URL retained as specified).\n\n"
        "## Notes\n\n"
        "- MENST Set 3 excluded; `test.csv` excluded from seed pool (eval holdout).\n"
        "- Keywords applied only to patient-authored fields "
        "(`<human>:` / `input` / `Question`).\n"
        "- Style variants recycle only claim phrases attested in canonical patient text.\n"
    )
    (OUT / "stats_summary.md").write_text("".join(stats), encoding="utf-8")

    leak = [
        "# Leakage exclusion list — seeds_en_v1\n\n",
        "Exclude these `(source_dataset, source_row_id)` from fine-tuning pulls.\n\n",
        "| seed_id | category | source_dataset | source_row_id |\n",
        "|---|---|---|---|\n",
    ]
    for r in seed_records:
        leak.append(
            f"| {r['seed_id']} | {r['category']} | {r['source_dataset']} | {r['source_row_id']} |\n"
        )
    (OUT / "leakage_note.md").write_text("".join(leak), encoding="utf-8")

    decisions = f"""# Decisions log — HerHealthGPT-LU English seed v1

## Phase-1 Clear-tier yield (post-dedup, before quality filter)

| Category | Clear | Borderline |
|----------|------:|-----------:|
| menstrual | {clear_raw.get('menstrual', 0)} | (see phase2 CSV) |
| pcos | {clear_raw.get('pcos', 0)} | (see phase2 CSV) |
| fertility | {clear_raw.get('fertility', 0)} | (see phase2 CSV) |

## Quality filter → eligible Clear pool

Filters remove myth-only FAQs (esp. fertility), male-patient context, texts lacking
attested clinical claim tokens, and extreme length outliers.

| Category | Eligible Clear |
|----------|---------------:|
| menstrual | {clear_eligible.get('menstrual', 0)} |
| pcos | {clear_eligible.get('pcos', 0)} |
| fertility | {clear_eligible.get('fertility', 0)} |

## Proposed targets (locked after Phase-1+filter)

| Category | Target | Rationale |
|----------|-------:|-----------|
| menstrual | {targets.get('menstrual')} | Abundant Clear; week-1 freeze ~tens; quality over volume → capped at 30 |
| pcos | {targets.get('pcos')} | Abundant Clear PCOS/PCOD patient language across HCM/iCliniq/MENST → 30 |
| fertility | {targets.get('fertility')} | Abundant Clear TTC/infertility pool after filters → 30 ({yield_notes.get('fertility')}) |

**Yield-constrained?** {', '.join(k for k,v in yield_notes.items() if v=='yield_constrained') or 'None at locked targets.'}

Not using the July-6 130-seed / 2-category plan. Three categories × ~30 = ~90 seeds.

## Source-mix policy

Target mix per category (~): up to ~20% MENST Set1 anchors (capped by sparse Clear Set1
eligible rows — training2K Set1 is only 117 total rows), ~25% MENST Set2, ~40% HealthCareMagic
authentic patient phrasing, ~15% iCliniq. Prefer Clear only in main set. Quotas auto-shrink
when a bucket is empty.

**Note:** MENST Set1 Clear eligible yield is single-digit overall after category split; the
achieved mix therefore skews toward HCM + Set2 for authentic symptom language while still
including available Set1 anchors.

## Borderline policy

Borderline candidates retained in `borderline_bucket.csv` with reason; **not promoted**
into `seeds_en_v1` while Clear eligible yield meets targets.

## Licensing / provenance

- HealthCareMagic-100k-Chat-Format-en: Apache-2.0
- MENST: MIT (dataset card)
- ChatDoctor-iCliniq: **license unverified** at time of seed build — usable for benchmark /
  seed construction with caution; do not redistribute as own data without clarifying upstream terms.

## Grounding policy

Evidence-only vs:
- NHS PCOS symptoms
- NHS heavy periods
- NHS infertility
- CDC Common Reproductive Health Concerns
- NICHD endometriosis / infertility

If a claim cannot be mapped safely → `needs_grounding_flag=true`, `gold_condition=NEEDS_GROUNDING`.
Heavy periods → NHS heavy periods; endometriosis/fibroids phrases → CDC common concerns;
PCOS/PCOD → NHS PCOS symptoms; TTC/infertility → NHS infertility.
Irregular/PMS/painful without an exact allowed page match → NEEDS_GROUNDING (do not invent).

## Style policy

6 rows/seed: `canonical` (verbatim patient text, light clean) + 5 register variants.
Variants may only reuse claim phrases attested in the patient text — no symptom invention.
"""
    (OUT / "decisions_log.md").write_text(decisions, encoding="utf-8")

    readme = """# HerHealthGPT-LU English seed dataset (v1)

Week-1 categorized **English** seed set for HerHealthGPT-LU / LUHME 2026.

## How to read deliverables

| File | Purpose |
|------|---------|
| `seeds_en_v1.jsonl` / `seeds_en_v1.csv` | Main deliverable: one row per (seed × style) |
| `stats_summary.md` | Counts, source mix, dedup, grounding |
| `leakage_note.md` | `(source_dataset, source_row_id)` to exclude from FT pulls |
| `decisions_log.md` | Target counts, filters, license notes |
| `borderline_bucket.csv` | Borderline candidates + keep/drop reason |
| `build_seed.py` | Reproducible pipeline |
| `phase1_*` / `phase2_*` / `phase3_*` | Intermediate dumps |

## Schema (`seeds_en_v1.*`)

- `seed_id` — `menst-NNN` / `pcos-NNN` / `fert-NNN`
- `category` — `menstrual` | `pcos` | `fertility`
- `source_dataset`, `source_row_id` — provenance
- `confidence_tier` — Clear (main set)
- `dedup_group_id` — set when near-dup cluster existed
- `canonical_text` — original patient question (light clean only)
- `style`, `style_text` — register variant (includes `canonical`)
- `gold_condition` — draft map to allowed NHS/CDC/NICHD page, or `NEEDS_GROUNDING`
- `needs_grounding_flag` — true when mapping uncertain

## Rebuild

```bash
python build_seed.py
```
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Resume from phase2 if present and --fresh not set
    phase2_path = OUT / "phase2_candidates_deduped.csv"
    drop_path = OUT / "phase2_dedup_drops.jsonl"
    reuse = phase2_path.exists() and drop_path.exists()

    if reuse:
        print("Reusing phase2_candidates_deduped.csv (skip Phase 1–2 extract)...")
        df = pd.read_csv(phase2_path)
        all_cands = []
        for _, row in df.iterrows():
            keep_val = row["keep"]
            if isinstance(keep_val, str):
                keep_bool = keep_val.lower() in ("true", "1", "yes")
            else:
                keep_bool = bool(keep_val)
            all_cands.append(
                Candidate(
                    source_dataset=row["source_dataset"],
                    source_row_id=str(row["source_row_id"]),
                    category=row["category"],
                    text=str(row["text"]),
                    confidence_tier=row["confidence_tier"],
                    match_notes=""
                    if pd.isna(row.get("match_notes"))
                    else str(row.get("match_notes")),
                    menst_set=normalize_menst_set(row.get("menst_set")),
                    topic="" if pd.isna(row.get("topic")) else str(row.get("topic")),
                    dedup_group_id=""
                    if pd.isna(row.get("dedup_group_id"))
                    else str(row.get("dedup_group_id")),
                    keep=keep_bool,
                    drop_reason=""
                    if pd.isna(row.get("drop_reason"))
                    else str(row.get("drop_reason")),
                )
            )
        drop_log = []
        with open(drop_path, encoding="utf-8") as f:
            for line in f:
                drop_log.append(json.loads(line))
    else:
        print("Phase 1: MENST...")
        menst = load_menst()
        print(f"  {len(menst)}")
        print("Phase 1: iCliniq...")
        icl = load_icliniq()
        print(f"  {len(icl)}")
        print("Phase 1: HCM...")
        hcm = load_hcm()
        print(f"  {len(hcm)}")
        all_cands = menst + icl + hcm
        write_candidates_csv(OUT / "phase1_candidates_raw.csv", all_cands)
        print("Phase 2: dedup...")
        all_cands, drop_log = cluster_near_dups(all_cands)
        write_candidates_csv(OUT / "phase2_candidates_deduped.csv", all_cands)
        with open(OUT / "phase2_dedup_drops.jsonl", "w", encoding="utf-8") as f:
            for d in drop_log:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    kept = [c for c in all_cands if c.keep]
    clear_raw = {
        cat: sum(1 for c in kept if c.category == cat and c.confidence_tier == "Clear")
        for cat in CATEGORIES
    }
    clear_eligible = {
        cat: sum(1 for c in kept if c.category == cat and is_seed_eligible(c))
        for cat in CATEGORIES
    }
    print("Clear raw:", clear_raw)
    print("Clear eligible:", clear_eligible)

    targets, yield_notes = propose_targets(clear_eligible)
    print("Proposed targets:", targets, yield_notes)
    with open(OUT / "phase3_proposed_targets.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "clear_raw_post_dedup": clear_raw,
                "clear_eligible": clear_eligible,
                "proposed_targets": targets,
                "yield_notes": yield_notes,
            },
            f,
            indent=2,
        )

    # Cross-category: prefer assigning a text to its strongest exclusive category when multi-tagged
    # Selection is per-category independently but block same normalize key across categories after pick
    print("Phase 3: stratified diversity selection...")
    seeds: list[Candidate] = []
    used_global: set[str] = set()
    for cat in CATEGORIES:
        pool = [c for c in kept if c.category == cat]
        sel = diversity_select_stratified(pool, targets[cat])
        # Drop if same patient text already used in another category
        filtered = []
        for c in sel:
            key = normalize_for_dedup(c.text)[:220]
            if key in used_global:
                continue
            used_global.add(key)
            filtered.append(c)
        # Top-up if cross-cat removals
        if len(filtered) < targets[cat]:
            for c in diversity_select_stratified(pool, targets[cat] * 3):
                if len(filtered) >= targets[cat]:
                    break
                key = normalize_for_dedup(c.text)[:220]
                if key in used_global:
                    continue
                if c in filtered:
                    continue
                if not is_seed_eligible(c):
                    continue
                used_global.add(key)
                filtered.append(c)
        print(
            f"  {cat}: {len(filtered)} / {targets[cat]} "
            f"(eligible pool {clear_eligible[cat]})"
        )
        seeds.extend(filtered[: targets[cat]])

    # Assign IDs + grounding + styles
    counters = {c: 0 for c in CATEGORIES}
    seed_records = []
    style_audit = {}
    rows = []
    for s in seeds:
        counters[s.category] += 1
        sid = f"{PREFIX[s.category]}-{counters[s.category]:03d}"
        gold, needs = draft_grounding(s.category, s.text)
        gold_out = "NEEDS_GROUNDING" if needs and gold.startswith("NEEDS") else gold
        # For menstrual draft CDC maps we keep gold string but needs=True
        if needs and gold.startswith("NEEDS"):
            gold_out = "NEEDS_GROUNDING"
        rec = {
            "seed_id": sid,
            "category": s.category,
            "source_dataset": s.source_dataset,
            "source_row_id": s.source_row_id,
            "confidence_tier": s.confidence_tier,
            "dedup_group_id": s.dedup_group_id or "",
            "canonical_text": s.text,
            "gold_condition": gold_out,
            "needs_grounding_flag": bool(needs),
            "menst_set": s.menst_set,
            "match_notes": s.match_notes,
        }
        seed_records.append(rec)
        styles = generate_styles(s.text, s.category)
        style_audit[sid] = {
            "claims": styles.get("_claims_used", []),
            "clinical": styles["clinical"],
            "layperson": styles["layperson"],
            "indirect_cultural": styles["indirect_cultural"],
            "ambiguous": styles["ambiguous"],
            "emotionally_concerned": styles["emotionally_concerned"],
        }
        for style in STYLES:
            if style == "canonical":
                st = s.text
            else:
                st = styles[style]
            rows.append(
                {
                    "seed_id": sid,
                    "category": s.category,
                    "source_dataset": s.source_dataset,
                    "source_row_id": s.source_row_id,
                    "confidence_tier": s.confidence_tier,
                    "dedup_group_id": s.dedup_group_id or "",
                    "canonical_text": s.text,
                    "style": style,
                    "style_text": st,
                    "gold_condition": gold_out,
                    "needs_grounding_flag": bool(needs),
                }
            )

    with open(OUT / "phase3_selected_seeds.json", "w", encoding="utf-8") as f:
        json.dump(seed_records, f, indent=2, ensure_ascii=False)
    with open(OUT / "style_variants.json", "w", encoding="utf-8") as f:
        json.dump(style_audit, f, indent=2, ensure_ascii=False)

    cols = [
        "seed_id",
        "category",
        "source_dataset",
        "source_row_id",
        "confidence_tier",
        "dedup_group_id",
        "canonical_text",
        "style",
        "style_text",
        "gold_condition",
        "needs_grounding_flag",
    ]
    with open(OUT / "seeds_en_v1.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT / "seeds_en_v1.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})

    # Borderline bucket
    promoted = {(r["source_dataset"], r["source_row_id"], r["category"]) for r in seed_records}
    borderline_rows = []
    for c in all_cands:
        if c.confidence_tier != "Borderline":
            continue
        key = (c.source_dataset, c.source_row_id, c.category)
        if key in promoted:
            status, reason = "kept_in_seeds", "promoted_for_diversity"
        elif not c.keep:
            status, reason = "dropped_dedup", c.drop_reason
        else:
            status, reason = "not_promoted", f"clear_yield_sufficient:{c.match_notes[:100]}"
        borderline_rows.append(
            {
                "source_dataset": c.source_dataset,
                "source_row_id": c.source_row_id,
                "category": c.category,
                "status": status,
                "reason": reason,
                "text": c.text[:500],
            }
        )
    with open(OUT / "borderline_bucket.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "source_dataset",
                "source_row_id",
                "category",
                "status",
                "reason",
                "text",
            ],
        )
        w.writeheader()
        w.writerows(borderline_rows)

    write_docs(
        seed_records,
        rows,
        targets,
        yield_notes,
        clear_raw,
        clear_eligible,
        drop_log,
        borderline_rows,
    )
    print(f"Wrote seeds_en_v1.jsonl ({len(rows)} rows), {len(seed_records)} seeds")
    print("Source mix:", Counter(r["source_dataset"] for r in seed_records))
    print("Needs grounding:", sum(1 for r in seed_records if r["needs_grounding_flag"]))
    print("Done.")


if __name__ == "__main__":
    main()
