"""HerHealthGPT-LU §3/§9 item 5 -- batch inference over the benchmark.

SCAFFOLD, NOT WIRED TO A REAL CLUSTER: written without access to Mazen's
cluster/serving details (host, port, exact served model name for M1/M2/M3/M4).
Targets any OpenAI-compatible chat-completions endpoint over plain HTTP (no
extra client dependency) -- this is what vLLM, TGI, and most local model
servers expose, so pointing --base-url at wherever a model is served should
work for LLaMA-3-8B-Instruct, Qwen3.5-9B, Qwen2.5-7B-Instruct (fallback), the
fine-tuned M3 checkpoint, and Menstrual-LLaMA-8B (M4) without per-model code.
Verify the endpoint contract matches your actual serving stack before relying
on this for real numbers.

Per spec §3: one fixed prompt, thinking mode off (if the served model exposes
a reasoning toggle, disable it -- not handled here, pass via --extra-body if
needed), temperature 0 (deterministic primary results per §4 statistical
rigor). One row per (item, model) in the output, matching §9 DoD.

Usage:
    python scripts/run_inference.py \\
        --base-url http://localhost:8000/v1 \\
        --model meta-llama/Meta-Llama-3-8B-Instruct \\
        --model-label M1 \\
        --benchmark HerHealthGPT-LU_seed/seeds_en_v1.csv \\
        --output HerHealthGPT-LU_seed/inference/M1_en.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import requests

FIXED_PROMPT_TEMPLATE = """A patient wrote the following message. Based only on what they \
wrote:
1. Interpret what symptom or concern they are describing.
2. Assess the urgency (routine / see-doctor / urgent).
3. Recommend an action.
4. If the description is insufficient to do the above confidently, ask a clarifying \
question instead of guessing.

Respond with ONLY a JSON object with exactly these keys: \
"interpreted_symptom", "urgency", "recommended_action", "asks_clarification" (true/false), \
"clarifying_question" (empty string if asks_clarification is false), "response_text" \
(what you would actually say to the patient).

Patient message: "{text}"
"""


def load_benchmark_rows(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    return list(csv.DictReader(path.open(encoding="utf-8", newline="")))


def call_endpoint(base_url: str, model: str, api_key: str, prompt: str, timeout: int) -> dict:
    resp = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 512,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True, help="e.g. http://localhost:8000/v1")
    parser.add_argument("--model", required=True, help="served model name/path")
    parser.add_argument("--model-label", required=True, help="M1 / M2 / M3 / M4 (paper label)")
    parser.add_argument("--api-key", default="EMPTY", help="most local servers ignore this")
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--limit", type=int, default=None, help="cap rows for a smoke test")
    args = parser.parse_args()

    rows = load_benchmark_rows(args.benchmark)
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} benchmark rows -> model {args.model_label} ({args.model})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if args.output.exists():
        done_ids = {json.loads(line)["item_id"] for line in args.output.open(encoding="utf-8")}
        print(f"resuming: {len(done_ids)} items already done")

    with open(args.output, "a", encoding="utf-8") as out:
        for i, row in enumerate(rows, start=1):
            item_id = f"{row.get('seed_id', row.get('id', i))}_{row.get('style', 'en')}"
            if item_id in done_ids:
                continue
            text = row.get("style_text") or row.get("canonical_text") or row.get("text", "")
            prompt = FIXED_PROMPT_TEMPLATE.format(text=text)
            try:
                api_response = call_endpoint(args.base_url, args.model, args.api_key, prompt, args.timeout)
                content = api_response["choices"][0]["message"]["content"]
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    parsed = {"_unparsed_response": content}
            except requests.RequestException as e:
                print(f"[{i}/{len(rows)}] {item_id}: ERROR {e}", file=sys.stderr)
                parsed = {"_error": str(e)}

            record = {
                "item_id": item_id,
                "seed_id": row.get("seed_id"),
                "model_label": args.model_label,
                "model": args.model,
                "input_text": text,
                **parsed,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            if i % 25 == 0:
                print(f"[{i}/{len(rows)}] ...")
            time.sleep(0.05)

    print(f"done -> {args.output}")


if __name__ == "__main__":
    main()
