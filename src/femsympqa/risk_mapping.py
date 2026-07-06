"""3-tier risk taxonomy mapped from clinical sources' own urgency language.

Rule (from the spec): high beats moderate beats low. Marker lists are
lowercase substrings matched against lowercased input.
"""
from femsympqa.schemas import RiskLevel

HIGH_MARKERS = [
    "999",
    "a&e",
    "emergency",
    "urgent gp",
    "urgent appointment",
    "immediately",
    "straight away",
]

MODERATE_MARKERS = [
    "see a gp",
    "see your gp",
    "speak to a gp",
    "speak to your gp",
    "gp appointment",
    "see a doctor",
    "talk to your doctor",
    "talk to a doctor",
    "contact a gp",
    "medical review",
]

_RISK_ORDER = {RiskLevel.LOW: 0, RiskLevel.MODERATE: 1, RiskLevel.HIGH: 2}


def map_urgency_to_risk(text: str) -> RiskLevel:
    lowered = text.lower()
    if any(marker in lowered for marker in HIGH_MARKERS):
        return RiskLevel.HIGH
    if any(marker in lowered for marker in MODERATE_MARKERS):
        return RiskLevel.MODERATE
    return RiskLevel.LOW


def merge_risk(a: RiskLevel, b: RiskLevel) -> RiskLevel:
    return a if _RISK_ORDER[a] >= _RISK_ORDER[b] else b
