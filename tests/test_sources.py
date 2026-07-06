from pathlib import Path

from femsympqa.sources import CONDITIONS, SOURCES, snapshot_path


def test_five_sources_defined():
    assert len(SOURCES) == 5
    assert len({s.slug for s in SOURCES}) == 5  # slugs unique


def test_all_conditions_covered():
    assert {s.condition for s in SOURCES} == set(CONDITIONS.values())
    assert set(CONDITIONS) == {"pcos", "heavy-periods", "infertility", "endometriosis"}


def test_snapshot_path():
    assert snapshot_path("nhs-pcos") == Path("data/raw/nhs-pcos.html")
