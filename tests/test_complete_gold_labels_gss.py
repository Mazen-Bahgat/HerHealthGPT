import csv
import io
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
import complete_gold_labels_gss as g  # noqa: E402


def _rows():
    # One seed group (6 style-siblings, shared Answer) + a second seed group.
    shared_answer = "You should see a doctor or clinician about this promptly."
    styles = ["canonical", "clinical", "layperson", "indirect_cultural", "ambiguous", "emotionally_concerned"]
    rows = [{"Question": f"q1-{s}", "Answer": shared_answer, "Topic": "PCOS",
             "Keywords": "k", "Style": s} for s in styles]
    other_answer = "This is a normal part of the cycle and needs no treatment."
    rows += [{"Question": f"q2-{s}", "Answer": other_answer, "Topic": "Menopause",
              "Keywords": "k", "Style": s} for s in styles]
    return rows


def test_label_rows_shares_risk_across_style_siblings():
    labeled = g.label_rows(_rows())
    q1_risks = {r["gold_risk_level"] for r in labeled if r["Answer"].startswith("You should see")}
    assert q1_risks == {"see-doctor"}
    q2_risks = {r["gold_risk_level"] for r in labeled if r["Answer"].startswith("This is a normal")}
    assert q2_risks == {"routine"}


def test_label_rows_clarification_only_on_ambiguous_style():
    labeled = g.label_rows(_rows())
    for r in labeled:
        expected = "yes" if r["Style"] == "ambiguous" else "no"
        assert r["requires_clarification"] == expected, r


def test_label_rows_condition_copies_topic():
    labeled = g.label_rows(_rows())
    assert all(r["gold_condition"] == r["Topic"] for r in labeled)


def test_write_labeled_csv_round_trips(tmp_path):
    out_path = tmp_path / "out.csv"
    g.write_labeled_csv(g.label_rows(_rows()), out_path)
    with out_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12
    assert {"gold_risk_level", "requires_clarification", "gold_condition"} <= set(rows[0].keys())
