from femsympqa.extraction import RawEntry, build_extract_prompt, merge_entries
from femsympqa.html_clean import Section
from femsympqa.schemas import RiskLevel


def raw(symptom="Irregular periods", url="https://nhs.uk/x",
        quote="See a GP if this persists", action="See a GP"):
    return RawEntry(condition="PCOS", source_url=url, symptom=symptom,
                    canonical_query="My periods are all over the place.",
                    recommended_action=action, urgency_quote=quote)


def test_build_extract_prompt_substitutes_fields():
    template = "Condition: {condition}\nURL: {url}\n{sections}"
    prompt = build_extract_prompt(
        "PCOS", "https://nhs.uk/x",
        [Section(heading="Symptoms", text="Tiredness and acne.")], template)
    assert "Condition: PCOS" in prompt
    assert "## Symptoms" in prompt
    assert "Tiredness and acne." in prompt


def test_merge_same_symptom_across_sources_merges_evidence():
    entries = merge_entries(
        [raw(url="https://nhs.uk/x"), raw(symptom="irregular  PERIODS", url="https://cdc.gov/y")],
        "pcos")
    assert len(entries) == 1
    assert {e.source_url for e in entries[0].evidence} == {"https://nhs.uk/x", "https://cdc.gov/y"}
    assert entries[0].base_id == "pcos-s01"
    assert entries[0].conflict_note is None


def test_merge_conflict_takes_higher_tier_and_records_note():
    entries = merge_entries(
        [raw(quote="This is common and usually nothing to worry about", action="Self-care"),
         raw(url="https://cdc.gov/y", quote="Ask for an urgent GP appointment",
             action="Urgent GP appointment")],
        "pcos")
    assert entries[0].risk_level is RiskLevel.HIGH
    assert "disagree" in entries[0].conflict_note
    # recommended_action comes from the highest-risk source
    assert entries[0].recommended_action == "Urgent GP appointment"


def test_distinct_symptoms_stay_separate_with_sequential_ids():
    entries = merge_entries([raw(), raw(symptom="Acne")], "pcos")
    assert len(entries) == 2
    assert sorted(e.base_id for e in entries) == ["pcos-s01", "pcos-s02"]
