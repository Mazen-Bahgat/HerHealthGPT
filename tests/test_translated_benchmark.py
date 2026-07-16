import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_benchmark_translation_handoff as handoff
import build_translated_benchmark as btb


def _bench_row(seed="gss-000", style="canonical", q="How is PCOS diagnosed?"):
    return {"seed_id": seed, "style": style, "style_text": q, "topic_raw": "PCOS",
            "category": "pcos", "gold_risk_level": "see-doctor",
            "requires_clarification": "no", "gold_condition": "PCOS",
            "gold_action": "see a gp", "gold_answer": "A doctor will test."}


def test_handoff_keys_are_unique_and_carry_english_question():
    bench = [_bench_row(), _bench_row(style="clinical", q="Clinical Q?")]
    rows = handoff.build_rows(bench)
    assert [r["item_key"] for r in rows] == ["gss-000__canonical", "gss-000__clinical"]
    assert rows[0]["Question"] == "How is PCOS diagnosed?"
    assert all(r["Question_translated"] == "" for r in rows)


def test_translate_swaps_text_keeps_gold():
    bench = [_bench_row()]
    ho = [{"item_key": "gss-000__canonical", "Question_translated": "Comment diagnostique-t-on le SOPK ?"}]
    out = btb.translate_benchmark(bench, ho)
    assert out[0]["style_text"] == "Comment diagnostique-t-on le SOPK ?"
    assert out[0]["gold_risk_level"] == "see-doctor"      # gold unchanged
    assert out[0]["gold_condition"] == "PCOS"
    assert out[0]["gold_answer"] == "A doctor will test."  # reference kept


def test_translate_fails_on_missing_translation():
    bench = [_bench_row(), _bench_row(style="clinical", q="Clinical Q?")]
    ho = [{"item_key": "gss-000__canonical", "Question_translated": "traduit"}]
    with pytest.raises(ValueError, match="no translation"):
        btb.translate_benchmark(bench, ho)


def test_translate_fails_on_unknown_key():
    bench = [_bench_row()]
    ho = [{"item_key": "gss-000__canonical", "Question_translated": "traduit"},
          {"item_key": "gss-999__canonical", "Question_translated": "orphan"}]
    with pytest.raises(ValueError, match="not in benchmark"):
        btb.translate_benchmark(bench, ho)
