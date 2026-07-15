import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import merge_ft_langs as mfl


def _write(path, texts):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"messages": [{"role": "user", "content": t}]}) + "\n")


def test_merge_is_deterministic_and_complete(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write(a, ["a1", "a2"])
    _write(b, ["b1"])
    merged1 = mfl.merge_shuffle([a, b], seed=3407)
    merged2 = mfl.merge_shuffle([a, b], seed=3407)
    assert merged1 == merged2
    contents = {r["messages"][0]["content"] for r in merged1}
    assert contents == {"a1", "a2", "b1"}
