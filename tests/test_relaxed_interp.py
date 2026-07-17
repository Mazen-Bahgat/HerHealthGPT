"""Tests for the relaxed clinically-acceptable interpretation metric."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import relaxed_interp as ri  # noqa: E402


def _markers(seed, text):
    return ri.seed_markers([{"seed_id": seed, "input_text": text}])


def test_conception_menstrual_seed_accepts_fertility():
    m = _markers("menst-002", "Do irregular periods influence the ability to get pregnant?")
    acc = ri.acceptable_set("menst-002", "menstrual", m)
    assert "menstrual" in acc and "fertility" in acc
    # no ovarian-cyst content -> pcos NOT acceptable
    assert "pcos" not in acc


def test_polycystic_menstrual_seed_accepts_pcos():
    m = _markers("menst-030", "irregular periods, ultrasound showed multiple cysts on the ovaries")
    acc = ri.acceptable_set("menst-030", "menstrual", m)
    assert "pcos" in acc


def test_plain_menstrual_seed_accepts_only_menstrual():
    m = _markers("menst-050", "my periods are heavier than usual this month")
    acc = ri.acceptable_set("menst-050", "menstrual", m)
    assert acc == {"menstrual"}


def test_relaxed_credits_justified_adjacent_not_unjustified():
    # gold menstrual, conception content -> fertility justified, pcos not
    recs = [
        {"parse_ok": True, "seed_id": "menst-002", "gold_category": "menstrual",
         "predicted_category": "fertility", "category_correct": False,
         "input_text": "irregular periods, trying to get pregnant"},
        {"parse_ok": True, "seed_id": "menst-002", "gold_category": "menstrual",
         "predicted_category": "pcos", "category_correct": False,
         "input_text": "irregular periods, trying to get pregnant"},
    ]
    markers = ri.seed_markers(recs)
    s = ri.score_relaxed(recs, markers)
    # one of two credited (fertility yes, pcos no)
    assert abs(s["relaxed"] - 0.5) < 1e-9
    assert s["strict"] == 0.0


def test_strict_correct_always_relaxed_correct():
    recs = [{"parse_ok": True, "seed_id": "fert-001", "gold_category": "fertility",
             "predicted_category": "fertility", "category_correct": True,
             "input_text": "trying to conceive for a year"}]
    markers = ri.seed_markers(recs)
    s = ri.score_relaxed(recs, markers)
    assert s["relaxed"] == 1.0 and s["strict"] == 1.0


def test_loose_credits_any_clinical_category_but_not_other():
    # plain menstrual seed (no fertility/pcos content): gated relaxed rejects the
    # pcos read, but loose credits it; an "other" read is credited by neither.
    recs = [
        {"parse_ok": True, "seed_id": "menst-050", "gold_category": "menstrual",
         "predicted_category": "pcos", "category_correct": False,
         "input_text": "my periods are heavier than usual this month"},
        {"parse_ok": True, "seed_id": "menst-051", "gold_category": "menstrual",
         "predicted_category": "other", "category_correct": False,
         "input_text": "my periods are heavier than usual this month"},
    ]
    markers = ri.seed_markers(recs)
    s = ri.score_relaxed(recs, markers)
    assert s["strict"] == 0.0
    assert abs(s["relaxed"] - 0.0) < 1e-9   # neither gated-credited (no content)
    assert abs(s["loose"] - 0.5) < 1e-9     # pcos credited, other not


def test_loose_is_upper_bound():
    recs = [{"parse_ok": True, "seed_id": "menst-002", "gold_category": "menstrual",
             "predicted_category": "fertility", "category_correct": False,
             "input_text": "irregular periods, trying to get pregnant"}]
    markers = ri.seed_markers(recs)
    s = ri.score_relaxed(recs, markers)
    assert s["strict"] <= s["relaxed"] <= s["loose"]
