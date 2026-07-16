import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
import compare_models as cm  # noqa: E402


def _group(cat=0.5, risk=0.4, clar=0.6):
    return {"parse_ok_rate": 1.0, "prediction_coverage": 1.0,
            "category_accuracy": cat, "risk_accuracy": risk,
            "clarification_accuracy": clar, "self_reported_unsafe_rate": 0.1}


def _summary(cat, risk):
    return {"overall": _group(cat, risk),
            "by_style": {"clinical": _group(cat, risk)},
            "by_gold_category": {"menstrual": _group(cat, risk)}}


def test_compare_computes_delta():
    comp = cm.compare_summaries(_summary(0.5, 0.4), _summary(0.7, 0.6))
    assert comp["overall"]["category_accuracy"]["baseline"] == 0.5
    assert comp["overall"]["category_accuracy"]["treatment"] == 0.7
    assert round(comp["overall"]["category_accuracy"]["delta"], 3) == 0.2
    assert "clinical" in comp["by_style"]
    assert "menstrual" in comp["by_gold_category"]


def test_render_markdown_has_delta_and_labels():
    comp = cm.compare_summaries(_summary(0.5, 0.4), _summary(0.7, 0.6))
    md = cm.render_markdown(comp, "M2", "M3")
    assert "M2" in md and "M3" in md
    assert "category_accuracy" in md
    assert "+0.200" in md or "0.200" in md  # delta rendered
