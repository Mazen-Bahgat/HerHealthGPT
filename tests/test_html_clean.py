from pathlib import Path

from femsympqa.html_clean import extract_sections

FIXTURE = Path("tests/fixtures/sample_condition_page.html")


def _sections():
    return extract_sections(FIXTURE.read_text(encoding="utf-8"))


def test_headings_extracted_in_order():
    headings = [s.heading for s in _sections()]
    assert headings == ["Example condition", "Symptoms", "When to see a GP", "Urgent advice"]


def test_list_items_joined_into_section_text():
    symptoms = next(s for s in _sections() if s.heading == "Symptoms")
    assert "heavy bleeding during periods" in symptoms.text
    assert "pain in the lower tummy" in symptoms.text


def test_boilerplate_excluded():
    all_text = " ".join(s.text for s in _sections())
    assert "banner" not in all_text
    assert "Footer legal" not in all_text
    assert "console.log" not in all_text


def test_short_sections_dropped():
    assert "Tiny" not in [s.heading for s in _sections()]
