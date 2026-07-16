import pytest

from femsympqa.risk_mapping import map_urgency_to_risk, merge_risk
from femsympqa.schemas import RiskLevel


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Call 999 or go to A&E now", RiskLevel.HIGH),
        ("Ask for an urgent GP appointment", RiskLevel.HIGH),
        ("Go to your nearest emergency department immediately", RiskLevel.HIGH),
        ("See a GP if these symptoms persist", RiskLevel.MODERATE),
        ("Speak to a GP about your symptoms", RiskLevel.MODERATE),
        ("Talk to your doctor if you are worried", RiskLevel.MODERATE),
        ("This is common and usually nothing to worry about", RiskLevel.LOW),
        ("You can usually treat this yourself at home", RiskLevel.LOW),
    ],
)
def test_map_urgency_to_risk(text, expected):
    assert map_urgency_to_risk(text) is expected


def test_high_beats_moderate_in_same_text():
    text = "See a GP, but call 999 if you have severe pain"
    assert map_urgency_to_risk(text) is RiskLevel.HIGH


def test_merge_risk_returns_higher_tier():
    assert merge_risk(RiskLevel.LOW, RiskLevel.MODERATE) is RiskLevel.MODERATE
    assert merge_risk(RiskLevel.HIGH, RiskLevel.MODERATE) is RiskLevel.HIGH
    assert merge_risk(RiskLevel.LOW, RiskLevel.LOW) is RiskLevel.LOW
