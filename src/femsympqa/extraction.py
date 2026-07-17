import re
from functools import reduce

from pydantic import BaseModel

from femsympqa.html_clean import Section
from femsympqa.risk_mapping import map_urgency_to_risk, merge_risk
from femsympqa.schemas import BaseEntry, Evidence


class ExtractedSymptom(BaseModel):
    symptom: str
    canonical_query: str
    recommended_action: str
    urgency_quote: str


class ExtractionResult(BaseModel):
    entries: list[ExtractedSymptom]


class RawEntry(BaseModel):
    condition: str
    source_url: str
    symptom: str
    canonical_query: str
    recommended_action: str
    urgency_quote: str


def build_extract_prompt(condition: str, url: str, sections: list[Section], template: str) -> str:
    rendered = "\n\n".join(f"## {s.heading}\n{s.text}" for s in sections)
    return template.format(condition=condition, url=url, sections=rendered)


def _normalize_symptom(symptom: str) -> str:
    return re.sub(r"\s+", " ", symptom.strip().lower())


def merge_entries(raw: list[RawEntry], condition_slug: str) -> list[BaseEntry]:
    grouped: dict[str, list[RawEntry]] = {}
    for item in raw:
        grouped.setdefault(_normalize_symptom(item.symptom), []).append(item)

    entries: list[BaseEntry] = []
    for idx, key in enumerate(sorted(grouped), start=1):
        items = grouped[key]
        risks = [map_urgency_to_risk(f"{i.urgency_quote} {i.recommended_action}") for i in items]
        final_risk = reduce(merge_risk, risks)
        conflict = None
        if len({r.value for r in risks}) > 1:
            tiers = ", ".join(sorted({r.value for r in risks}))
            conflict = f"sources disagree on urgency ({tiers}); higher tier applied"
        # action from the (first) highest-risk source
        best = max(zip(risks, items), key=lambda pair: ["low", "moderate", "high"].index(pair[0].value))
        entries.append(BaseEntry(
            base_id=f"{condition_slug}-s{idx:02d}",
            condition=items[0].condition,
            symptom=items[0].symptom,
            canonical_query=items[0].canonical_query,
            risk_level=final_risk,
            recommended_action=best[1].recommended_action,
            evidence=[Evidence(source_url=i.source_url, quote=i.urgency_quote) for i in items],
            conflict_note=conflict,
        ))
    return entries
