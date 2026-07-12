import hashlib
import json
from pathlib import Path

from scripts.run_inference import FIXED_PROMPT_TEMPLATE
from scripts.run_local_raw_baseline import (
    build_item_id,
    load_benchmark,
    parse_response,
    prompt_sha256,
    run_rows,
)


def test_prompt_hash_reuses_endpoint_runner_contract():
    assert prompt_sha256() == hashlib.sha256(FIXED_PROMPT_TEMPLATE.encode("utf-8")).hexdigest()


def test_load_benchmark_reads_jsonl(tmp_path: Path):
    path = tmp_path / "benchmark.jsonl"
    rows = [{"seed_id": "a", "style": "canonical"}, {"seed_id": "b", "style": "layperson"}]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert load_benchmark(path) == rows


def test_stable_item_id_matches_endpoint_runner_convention():
    assert build_item_id({"seed_id": "menst-001", "style": "clinical"}) == "menst-001_clinical"


def test_parse_response_returns_json_fields_or_raw_fallback():
    assert parse_response('{"urgency":"routine"}') == {"urgency": "routine"}
    assert parse_response("not json") == {"_unparsed_response": "not json"}


def test_run_rows_resumes_without_duplicate_completed_ids(tmp_path: Path):
    output = tmp_path / "out.jsonl"
    existing = {
        "item_id": "a_canonical",
        "seed_id": "a",
        "model_label": "M2",
        "model": "Qwen/Qwen3.5-9B",
        "input_text": "first",
        "raw_response": "{}",
    }
    output.write_text(json.dumps(existing) + "\n", encoding="utf-8")
    rows = [
        {"seed_id": "a", "category": "x", "style": "canonical", "style_text": "first"},
        {"seed_id": "b", "category": "y", "style": "layperson", "style_text": "second"},
    ]
    calls = []

    def generate(prompt: str) -> str:
        calls.append(prompt)
        return '{"urgency":"routine"}'

    counts = run_rows(rows, output, generate)
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [record["item_id"] for record in records] == ["a_canonical", "b_layperson"]
    assert len(calls) == 1
    assert counts == {"completed": 2, "errors": 0, "unparsed": 0, "generated": 1, "skipped": 1}
    assert records[1]["category"] == "y"
    assert records[1]["style"] == "layperson"
    assert records[1]["raw_response"] == '{"urgency":"routine"}'


def test_run_rows_preserves_generation_error_and_remains_resumable(tmp_path: Path):
    output = tmp_path / "out.jsonl"
    rows = [{"seed_id": "a", "category": "x", "style": "canonical", "style_text": "first"}]

    def fail(_prompt: str) -> str:
        raise RuntimeError("boom")

    counts = run_rows(rows, output, fail)
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["_error"] == "boom"
    assert counts["errors"] == 1
    resumed = run_rows(rows, output, fail)
    assert resumed["skipped"] == 1
    assert resumed["errors"] == 1
