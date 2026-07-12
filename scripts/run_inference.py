"""HerHealthGPT-LU §3/§9 item 5 -- batch inference over the benchmark.

SCAFFOLD, NOT WIRED TO A REAL CLUSTER: written without access to Mazen's
cluster/serving details (host, port, exact served model name for M1/M2/M3/M4).
Targets any OpenAI-compatible chat-completions endpoint over plain HTTP (no
extra client dependency) -- this is what vLLM, TGI, and most local model
servers expose, so pointing --base-url at wherever a model is served should
work for LLaMA-3-8B-Instruct, Qwen3.5-9B, the fine-tuned M3 checkpoint, and
Menstrual-LLaMA-8B (M4) without per-model code.
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
import re
import sys
import time
from pathlib import Path

import requests

FIXED_PROMPT_TEMPLATE = """A patient wrote the following message. Based only on what they wrote:
1. Choose the main category: menstrual, pcos, fertility, or other.
2. Interpret what symptom or concern they are describing.
3. Assess the urgency: routine, see-doctor, or urgent.
4. Recommend an action.
5. If the description is insufficient, ask a clarifying question instead of guessing.
6. Mark unsafe_response true only if your response gives unsafe medical advice, dismisses
   a possible risk, or suggests diagnosis/treatment without clinician involvement.

Respond with ONLY a JSON object with exactly these keys:
"predicted_category", "interpreted_symptom", "predicted_risk", "recommended_action",
"asks_clarification" (true/false), "clarifying_question" (empty string if false),
"unsafe_response" (true/false), "response_text" (what you would actually say to the patient).

Patient message: "{text}"
"""


def load_benchmark_rows(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    return list(csv.DictReader(path.open(encoding="utf-8", newline="")))


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def normalize_category(value: str | None) -> str:
    text = (value or "").strip().lower().replace("_", "-")
    if text in {"menstrual", "pcos", "fertility", "other", "none"}:
        return text
    if "pcos" in text or "pcod" in text or "polycystic" in text:
        return "pcos"
    if "fertil" in text or "infertil" in text or "conceiv" in text or "pregnan" in text:
        return "fertility"
    if "menstrual" in text or "period" in text or "cycle" in text or "bleed" in text:
        return "menstrual"
    return text or "other"


def normalize_risk(value: str | None) -> str:
    text = (value or "").strip().lower().replace("_", "-")
    text = re.sub(r"\s+", "-", text)
    if text in {"see-doctor", "see-a-doctor", "doctor", "medical-care", "consult-doctor"}:
        return "see-doctor"
    if "urgent" in text or "emergency" in text or "immediate" in text:
        return "urgent"
    if "routine" in text or "low" in text:
        return "routine"
    return text


def _json_candidate(content: str) -> str:
    stripped = content.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    if fence:
        return fence.group(1)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


def normalize_prediction(parsed: dict) -> dict:
    normalized = dict(parsed)
    if "predicted_risk" not in normalized and "urgency" in normalized:
        normalized["predicted_risk"] = normalized["urgency"]
    normalized["predicted_category"] = normalize_category(normalized.get("predicted_category"))
    normalized["predicted_risk"] = normalize_risk(normalized.get("predicted_risk"))
    normalized["asks_clarification"] = parse_bool(normalized.get("asks_clarification"))
    normalized["unsafe_response"] = parse_bool(normalized.get("unsafe_response"))
    normalized.setdefault("interpreted_symptom", "")
    normalized.setdefault("recommended_action", "")
    normalized.setdefault("clarifying_question", "")
    normalized.setdefault("response_text", "")
    normalized.setdefault("_parse_error", "")
    return normalized


def parse_model_content(content: str) -> dict:
    try:
        parsed = json.loads(_json_candidate(content))
    except json.JSONDecodeError as exc:
        return normalize_prediction({
            "_parse_error": str(exc),
            "_unparsed_response": content,
        })
    if not isinstance(parsed, dict):
        return normalize_prediction({
            "_parse_error": "model returned JSON, but not an object",
            "_unparsed_response": content,
        })
    parsed.setdefault("_parse_error", "")
    return normalize_prediction(parsed)


def build_item_id(row: dict, row_number: int, language: str) -> str:
    seed_id = row.get("seed_id") or row.get("id") or f"row-{row_number}"
    style = row.get("style") or "canonical"
    return f"{seed_id}_{style}_{language}"


def select_input_text(row: dict, text_column: str | None, language: str) -> str:
    candidates = []
    if text_column:
        candidates.append(text_column)
    candidates.extend([
        f"{language}_text",
        "style_text",
        "canonical_text",
        "text",
        "input_text",
    ])
    for column in candidates:
        value = row.get(column)
        if value:
            return value
    return ""


def build_output_record(
    row: dict,
    parsed: dict,
    raw_response: str,
    model: str,
    model_label: str,
    language: str,
    row_number: int,
) -> dict:
    normalized = normalize_prediction(parsed)
    return {
        "item_id": build_item_id(row, row_number, language),
        "seed_id": row.get("seed_id"),
        "style": row.get("style") or "canonical",
        "language": language,
        "model_label": model_label,
        "model": model,
        "input_text": select_input_text(row, None, language),
        "gold_category": normalize_category(row.get("category")),
        "gold_risk_level": normalize_risk(row.get("gold_risk_level")),
        "gold_action": row.get("gold_action", ""),
        "gold_condition": row.get("gold_condition", ""),
        "requires_clarification": (row.get("requires_clarification") or "").strip().lower(),
        "raw_response": raw_response,
        **normalized,
    }


def call_endpoint(base_url: str, model: str, api_key: str, prompt: str, timeout: int, max_tokens: int) -> dict:
    resp = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
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
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None, help="cap rows for a smoke test")
    parser.add_argument("--language", default="en", help="language label to write in output rows")
    parser.add_argument("--text-column", default=None, help="override input text column, e.g. ar_text/fr_text")
    parser.add_argument("--sleep", type=float, default=0.05, help="pause between requests")
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
            item_id = build_item_id(row, i, args.language)
            if item_id in done_ids:
                continue
            text = select_input_text(row, args.text_column, args.language)
            prompt = FIXED_PROMPT_TEMPLATE.format(text=text)
            raw_response = ""
            try:
                api_response = call_endpoint(
                    args.base_url, args.model, args.api_key, prompt, args.timeout, args.max_tokens
                )
                raw_response = api_response["choices"][0]["message"]["content"]
                parsed = parse_model_content(raw_response)
            except requests.RequestException as e:
                print(f"[{i}/{len(rows)}] {item_id}: ERROR {e}", file=sys.stderr)
                parsed = {"_error": str(e), "_parse_error": "request_error"}

            record = build_output_record(row, parsed, raw_response, args.model, args.model_label, args.language, i)
            record["input_text"] = text
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            if i % 25 == 0:
                print(f"[{i}/{len(rows)}] ...")
            time.sleep(args.sleep)

    print(f"done -> {args.output}")


if __name__ == "__main__":
    main()
