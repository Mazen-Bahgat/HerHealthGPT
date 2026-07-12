"""Run the untouched Qwen3.5-9B checkpoint on the frozen English benchmark."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from scripts.run_inference import FIXED_PROMPT_TEMPLATE

MODEL_ID = "Qwen/Qwen3.5-9B"
MODEL_LABEL = "M2"
DEFAULT_BENCHMARK = Path("HerHealthGPT-LU_seed/seeds_en_v1.jsonl")
DEFAULT_OUTPUT = Path("HerHealthGPT-LU_seed/inference/M2_qwen35_raw_en.jsonl")


def prompt_sha256() -> str:
    return hashlib.sha256(FIXED_PROMPT_TEMPLATE.encode("utf-8")).hexdigest()


def load_benchmark(path: Path) -> list[dict]:
    if path.suffix.lower() != ".jsonl":
        raise ValueError("raw baseline benchmark must be a JSONL file")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_item_id(row: dict) -> str:
    return f"{row['seed_id']}_{row['style']}"


def parse_response(content: str) -> dict:
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return {"_unparsed_response": content}
    return parsed if isinstance(parsed, dict) else {"_unparsed_response": content}


def _existing_records(output: Path) -> list[dict]:
    if not output.exists():
        return []
    records: list[dict] = []
    with output.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                record["item_id"]
                records.append(record)
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"invalid resume row {line_number} in {output}: {exc}") from exc
    return records


def run_rows(rows: Iterable[dict], output: Path, generate: Callable[[str], str]) -> dict[str, int]:
    rows = list(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = _existing_records(output)
    done_ids = {record["item_id"] for record in existing}
    counts = {
        "completed": len(done_ids),
        "errors": sum("_error" in record for record in existing),
        "unparsed": sum("_unparsed_response" in record for record in existing),
        "generated": 0,
        "skipped": 0,
    }
    with output.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(rows, 1):
            item_id = build_item_id(row)
            if item_id in done_ids:
                counts["skipped"] += 1
                continue
            text = row.get("style_text") or row.get("canonical_text") or row.get("text", "")
            raw_response = ""
            try:
                raw_response = generate(FIXED_PROMPT_TEMPLATE.format(text=text))
                parsed = parse_response(raw_response)
            except Exception as exc:  # preserve each failure so a run is resumable
                parsed = {"_error": str(exc)}
            record = {
                "item_id": item_id,
                "seed_id": row.get("seed_id"),
                "category": row.get("category"),
                "style": row.get("style"),
                "model_label": MODEL_LABEL,
                "model": MODEL_ID,
                "input_text": text,
                "raw_response": raw_response,
                **parsed,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            done_ids.add(item_id)
            counts["generated"] += 1
            counts["completed"] += 1
            counts["errors"] += int("_error" in parsed)
            counts["unparsed"] += int("_unparsed_response" in parsed)
            print(f"[{index}/{len(rows)}] {item_id}", flush=True)
    return counts


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


class LocalQwenGenerator:
    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.model.eval()
        self.revision = getattr(self.model.config, "_commit_hash", None)
        self.dtype = str(next(self.model.parameters()).dtype)

    def __call__(self, prompt: str) -> str:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        input_length = inputs["input_ids"].shape[-1]
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, do_sample=False, max_new_tokens=512)
        return self.processor.decode(generated[0][input_length:], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    metadata_output = args.metadata_output or args.output.with_suffix(".metadata.json")
    rows = load_benchmark(args.benchmark)
    if args.limit is not None:
        rows = rows[: args.limit]
    started = datetime.now(timezone.utc)
    generator = LocalQwenGenerator()
    counts = run_rows(rows, args.output, generator)
    ended = datetime.now(timezone.utc)
    metadata = {
        "model_id": MODEL_ID,
        "resolved_revision": generator.revision,
        "git_sha": _git_sha(),
        "package_versions": {name: _version(name) for name in ("torch", "transformers", "accelerate", "huggingface-hub")},
        "dtype": generator.dtype,
        "decoding": {"do_sample": False, "temperature": None, "max_new_tokens": 512, "enable_thinking": False},
        "prompt_sha256": prompt_sha256(),
        "benchmark_path": str(args.benchmark),
        "benchmark_count": len(rows),
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        **counts,
    }
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
