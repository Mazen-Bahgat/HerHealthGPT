import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import merge_ft_langs as merge


def _write_corpus(data_dir: Path, lang: str, split: str, n: int) -> None:
    d = data_dir / f"{lang}_v2_json"
    d.mkdir(parents=True)
    with (d / f"{split}.jsonl").open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"lang": lang, "i": i}) + "\n")


def test_merge_concatenates_all_languages_and_is_deterministic(tmp_path):
    _write_corpus(tmp_path, "en", "train", 3)
    _write_corpus(tmp_path, "fr", "train", 2)
    _write_corpus(tmp_path, "ar", "train", 4)
    a = merge.merge_lines(["en", "fr", "ar"], "train", seed=3407, data_dir=tmp_path)
    b = merge.merge_lines(["en", "fr", "ar"], "train", seed=3407, data_dir=tmp_path)
    assert len(a) == 9
    assert a == b  # same seed, same order
    langs = [json.loads(x)["lang"] for x in a]
    assert sorted(langs) == ["ar"] * 4 + ["en"] * 3 + ["fr"] * 2


def test_merge_preserves_records_verbatim(tmp_path):
    _write_corpus(tmp_path, "en", "val", 2)
    src = (tmp_path / "en_v2_json" / "val.jsonl").read_text(encoding="utf-8")
    out = merge.merge_lines(["en"], "val", seed=1, data_dir=tmp_path)
    assert sorted(out) == sorted(src.strip().split("\n"))


def test_merge_fails_on_missing_language(tmp_path):
    _write_corpus(tmp_path, "en", "train", 1)
    with pytest.raises(SystemExit):
        merge.merge_lines(["en", "fr"], "train", seed=1, data_dir=tmp_path)
