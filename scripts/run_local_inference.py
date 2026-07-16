"""Local (no-server) benchmark inference for HerHealthGPT-LU (M2/M3).

Loads a model locally via Unsloth (base, or base+adapter) and generates
structured predictions over the frozen English benchmark, writing JSONL in the
exact schema scripts/evaluate.py consumes. No server; reuses the prompt and
record schema from scripts/run_inference.py (DRY).

Heavy deps (torch/unsloth) are imported lazily inside LocalGenerator so the pure
helpers import cleanly on the Windows .venv for unit testing.

Run inside WSL (Ubuntu, ft-train-venv), offline:
  HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 /home/sw2/ft-train-venv/bin/python \
    scripts/run_local_inference.py --label M2 \
    --model /home/sw2/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a \
    --benchmark HerHealthGPT-LU_seed/seeds_en_v1.jsonl \
    --output HerHealthGPT-LU_seed/inference/M2_en.jsonl
  # M3 adds: --adapter models/qwen3.5-9b-herhealth-en-lora
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_inference as inf  # noqa: E402

MAX_SEQ = 2048
LANGUAGE = "en"


def iter_benchmark(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def done_item_ids(path: Path) -> set[str]:
    path = Path(path)
    if not path.exists():
        return set()
    ids = set()
    for line in path.open(encoding="utf-8"):
        if line.strip():
            ids.add(json.loads(line)["item_id"])
    return ids


def record_for_row(row: dict, raw_response: str, model: str, label: str, row_number: int) -> dict:
    parsed = inf.parse_model_content(raw_response)
    record = inf.build_output_record(row, parsed, raw_response, model, label, LANGUAGE, row_number)
    record["input_text"] = inf.select_input_text(row, None, LANGUAGE)
    return record


class LocalGenerator:
    def __init__(self, model_path: str, adapter: str | None, max_new_tokens: int,
                 max_time: float | None = None) -> None:
        from unsloth import FastLanguageModel
        import torch
        self.torch = torch
        load_from = adapter or model_path
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=load_from, max_seq_length=MAX_SEQ, load_in_4bit=True, dtype=None)
        FastLanguageModel.for_inference(self.model)
        self.max_new_tokens = max_new_tokens
        # Per-row wall-clock safety bound: greedy decoding on some FR/AR inputs
        # can spin in the (Python-level) generation loop without emitting EOS,
        # stalling a multi-hour run on a single row. transformers checks max_time
        # between decode steps, so a stuck row is cut off and marked instead of
        # hanging. Rows that finish under the bound are byte-identical to a run
        # without it, so EN/FR/AR stay comparable.
        self.max_time = max_time

    def __call__(self, prompt: str) -> str:
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False,
            return_tensors="pt", return_dict=True).to("cuda")
        gen_kwargs = dict(max_new_tokens=self.max_new_tokens,
                          do_sample=False, use_cache=True)
        if self.max_time:
            gen_kwargs["max_time"] = self.max_time
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="M2 / M3 (paper label)")
    ap.add_argument("--model", required=True, help="base model path/id")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (M3)")
    ap.add_argument("--benchmark", type=Path, default=Path("HerHealthGPT-LU_seed/seeds_en_v1.jsonl"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--language", default="en",
                    help="benchmark language; select_input_text picks {language}_text "
                         "(e.g. fr_text/ar_text on the multilingual benchmark)")
    ap.add_argument("--gen-max-time", type=float, default=None,
                    help="per-row wall-clock cap (s) for generation; guards against "
                         "non-terminating greedy decode stalling the run")
    args = ap.parse_args()

    global LANGUAGE
    LANGUAGE = args.language

    rows = iter_benchmark(args.benchmark)
    if args.limit:
        rows = rows[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    already = done_item_ids(args.output)
    print(f"{len(rows)} rows -> {args.label} ({args.adapter or args.model}); {len(already)} already done")

    gen = LocalGenerator(args.model, args.adapter, args.max_new_tokens, args.gen_max_time)
    with args.output.open("a", encoding="utf-8") as out:
        for i, row in enumerate(rows, start=1):
            item_id = inf.build_item_id(row, i, LANGUAGE)
            if item_id in already:
                continue
            text = inf.select_input_text(row, None, LANGUAGE)
            try:
                raw = gen(inf.FIXED_PROMPT_TEMPLATE.format(text=text))
            except Exception as exc:  # keep going; mark the row
                raw = ""
                record = inf.build_output_record(
                    row, {"_error": str(exc), "_parse_error": "generation_error"},
                    raw, args.model, args.label, LANGUAGE, i)
            else:
                record = record_for_row(row, raw, args.model, args.label, i)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            if i % 25 == 0:
                print(f"[{i}/{len(rows)}] ...")
    print(f"done -> {args.output}")


if __name__ == "__main__":
    main()
