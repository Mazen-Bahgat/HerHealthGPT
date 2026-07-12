import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
import prepare_ft_data as p  # noqa: E402


def _row(cat="menstrual", n=0):
    return {
        "instruction": "You are a supportive assistant.",
        "input": f"question {n}",
        "output": f"answer {n}",
        "category": cat,
        "source_dataset": "MENST",
        "source_row_id": str(n),
    }


def test_to_chat_record_maps_three_roles():
    rec = p.to_chat_record(_row(n=1))
    roles = [m["role"] for m in rec["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert rec["messages"][0]["content"] == "You are a supportive assistant."
    assert rec["messages"][1]["content"] == "question 1"
    assert rec["messages"][2]["content"] == "answer 1"
    assert rec["category"] == "menstrual"


def test_split_is_balanced_and_deterministic():
    rows = (
        [_row("menstrual", i) for i in range(100)]
        + [_row("pcos", i) for i in range(100)]
        + [_row("fertility", i) for i in range(100)]
    )
    train, val = p.split_train_val(rows, val_frac=0.05, seed=42)
    assert len(val) == 15
    from collections import Counter

    assert Counter(r["category"] for r in val) == {
        "menstrual": 5,
        "pcos": 5,
        "fertility": 5,
    }
    assert len(train) == 285
    train2, val2 = p.split_train_val(rows, val_frac=0.05, seed=42)
    assert [r["input"] for r in val] == [r["input"] for r in val2]
    val_ids = {(r["category"], r["input"]) for r in val}
    assert not any((r["category"], r["input"]) in val_ids for r in train)
