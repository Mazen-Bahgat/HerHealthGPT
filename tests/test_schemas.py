import pytest
from pydantic import ValidationError

from femsympqa.schemas import BaseEntry, Evidence, FinalRecord, Provenance, RiskLevel


def make_entry(**overrides):
    data = dict(
        base_id="pcos-s01",
        condition="PCOS",
        symptom="irregular periods",
        canonical_query="My periods have been really irregular for months.",
        risk_level="moderate",
        recommended_action="See a GP for assessment.",
        evidence=[Evidence(source_url="https://www.nhs.uk/x", quote="see a GP")],
    )
    data.update(overrides)
    return BaseEntry(**data)


def test_base_entry_valid():
    entry = make_entry()
    assert entry.risk_level is RiskLevel.MODERATE
    assert entry.conflict_note is None


def test_base_entry_rejects_bad_risk():
    with pytest.raises(ValidationError):
        make_entry(risk_level="critical")


def test_base_entry_requires_evidence():
    with pytest.raises(ValidationError):
        make_entry(evidence=[])


def test_final_record_roundtrip():
    rec = FinalRecord(
        id="fsq-pcos-s01-p03-fr",
        base_id="pcos-s01",
        parent_id="fsq-pcos-s01-p03-en",
        lang="fr",
        variant_type="translation",
        query="Mes règles sont très irrégulières depuis des mois.",
        condition="PCOS",
        risk_level="moderate",
        recommended_action="See a GP for assessment.",
        evidence=[Evidence(source_url="https://www.nhs.uk/x", quote="see a GP")],
        provenance=Provenance(
            generated_by="facebook/nllb-200-distilled-600M",
            generated_at="2026-07-05",
            prompt_id="nllb-beam4-v1",
        ),
    )
    again = FinalRecord.model_validate(rec.model_dump())
    assert again == rec


def test_final_record_rejects_bad_lang():
    with pytest.raises(ValidationError):
        FinalRecord(
            id="x", base_id="b", parent_id=None, lang="de",
            variant_type="canonical", query="q", condition="PCOS",
            risk_level="low", recommended_action="a",
            evidence=[Evidence(source_url="u", quote="q")],
            provenance=Provenance(generated_by="g", generated_at="d", prompt_id="p"),
        )
