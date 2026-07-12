# Qwen3.5-9B English QLoRA Fine-Tune (M3) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce M3 — a QLoRA adapter for Qwen3.5-9B fine-tuned on the existing 2,700-pair English corpus — locally, plus a reproducible WSL2 training pipeline.

**Architecture:** Three modular scripts under `scripts/` run inside WSL2 Ubuntu on the local RTX 5000 Ada (32 GB): `prepare_ft_data.py` (CPU, deterministic, unit-tested) formats the corpus into Qwen chat messages and a train/val split; `train_qlora.py` (GPU, Unsloth) runs QLoRA with a 10-step smoke gate before the full run; `sanity_generate.py` (GPU) eyeballs outputs. Training dependencies live only in the WSL venv, never in `pyproject.toml`.

**Tech Stack:** WSL2 Ubuntu, `uv` (Python 3.11 venv), PyTorch (CUDA 12.4), Unsloth, transformers/trl/peft/bitsandbytes/datasets, pytest (data-prep test on the Windows `.venv`).

## Global Constraints

- **Source corpus (training):** `HerHealthGPT-LU_seed/ft_corpus_v1.jsonl` — 2,700 rows, keys `instruction`/`input`/`output`/`category`/`source_dataset`/`source_row_id`, balanced 900/category. This is the ONLY training input.
- **Never train on** `HerHealthGPT-LU_seed/seeds_en_v1.{csv,jsonl}` (the 540-row frozen benchmark) or any candidate CSV. The corpus is already dual-key leakage-cleaned against the seeds; do not add seed data.
- **Chat template:** Qwen, **thinking-mode OFF** (`enable_thinking=False`) at both format and inference time, so train/eval match.
- **QLoRA recipe (from parent spec §2C / MenstLLaMA):** 4-bit NF4, LoRA r=16 alpha=16 dropout=0, lr 2e-4, warmup ratio 0.03, max grad norm 0.3, max seq 2048, paged AdamW 8-bit, cosine schedule, bf16. Epochs start at 3 (ceiling 5).
- **Model:** primary `Qwen/Qwen3.5-9B`; fallback `Qwen/Qwen2.5-7B-Instruct` via one `--model` flag change if the primary won't load/train in Unsloth within a bounded window.
- **Seeds:** training seed `3407`; val-split seed `42`. Both recorded.
- **Python:** WSL system Python is 3.14 (unsupported by torch/Unsloth) — the training env MUST pin 3.11 via `uv`.
- **Repo path in WSL:** `/mnt/d/Grad-Project/HerHealthGPT` (Windows `d:\Grad-Project\HerHealthGPT`).
- **Parent spec (source of truth):** `docs/superpowers/specs/2026-07-06-herhealthgpt-lu-design.md`. **This plan's spec:** `docs/superpowers/specs/2026-07-12-qwen35-english-qlora-finetune-design.md`.

---

## File Structure

- `docs/wsl_finetune_setup.md` — WSL2 + uv + Unsloth setup, reproducible (Task 1).
- `scripts/prepare_ft_data.py` — corpus → chat messages + train/val jsonl (Task 2).
- `tests/test_prepare_ft_data.py` — unit tests for the pure functions (Task 2).
- `data/ft/en/{train,val}.jsonl` — generated, gitignored (Task 2).
- `scripts/train_qlora.py` — Unsloth QLoRA training + `run_config.json` (Task 3, run in Tasks 3–4).
- `models/qwen3.5-9b-herhealth-en-lora/` — adapter output, gitignored (Task 4).
- `scripts/sanity_generate.py` — load adapter, generate on prompts (Task 5).
- `.gitignore` — add `models/` and `data/ft/` (Tasks 2–3).
- `README.md` + `docs/finetune_run_notes.md` — pipeline row + paper/method note (Task 6).

---

## Task 1: WSL2 + Unsloth environment (setup doc)

**Files:**
- Create: `docs/wsl_finetune_setup.md`

**Interfaces:**
- Produces: an isolated WSL venv at `/mnt/d/Grad-Project/HerHealthGPT/.venv-ft` (Python 3.11) with torch+Unsloth importable and CUDA visible. Later tasks run their `python`/`pytest` GPU commands from this venv.

- [ ] **Step 1: Confirm GPU passthrough in WSL**

Run (from Windows, in the Bash tool):
```bash
wsl.exe -d Ubuntu -- bash -lc "nvidia-smi --query-gpu=name,memory.total --format=csv | tr -d '\0'"
```
Expected: a line containing `NVIDIA RTX 5000 Ada Generation, 32760 MiB`.

- [ ] **Step 2: Create the isolated 3.11 venv with uv**

Run:
```bash
wsl.exe -d Ubuntu -- bash -lc "cd /mnt/d/Grad-Project/HerHealthGPT && uv venv --python 3.11 .venv-ft"
```
Expected: `Creating virtual environment ... .venv-ft`. (WSL system Python is 3.14 and must not be used.)

- [ ] **Step 3: Install torch (CUDA 12.4) then Unsloth + training deps**

Run:
```bash
wsl.exe -d Ubuntu -- bash -lc "cd /mnt/d/Grad-Project/HerHealthGPT && \
  .venv-ft/bin/python -m pip install --upgrade pip && \
  .venv-ft/bin/pip install 'torch==2.5.*' --index-url https://download.pytorch.org/whl/cu124 && \
  .venv-ft/bin/pip install unsloth 'transformers>=4.57' trl peft bitsandbytes datasets accelerate"
```
Expected: installs complete without error. (RTX 5000 Ada is sm_89 — cu124 wheels support it. A very recent `transformers` is required for Qwen3.5's GatedDeltaNet; bump the floor if the model card specifies higher.)

- [ ] **Step 4: Verify torch sees the GPU and Unsloth imports**

Run:
```bash
wsl.exe -d Ubuntu -- bash -lc "cd /mnt/d/Grad-Project/HerHealthGPT && \
  .venv-ft/bin/python -c 'import torch, unsloth; print(\"cuda\", torch.cuda.is_available(), torch.cuda.get_device_name(0))'"
```
Expected: `cuda True NVIDIA RTX 5000 Ada Generation`.

- [ ] **Step 5: Write the setup doc**

Create `docs/wsl_finetune_setup.md` capturing Steps 1–4 verbatim as copy-paste blocks, with this header:
```markdown
# WSL2 fine-tuning environment (local RTX 5000 Ada, 32 GB)

Reproducible setup for the M3 QLoRA fine-tune. Runs inside WSL2 Ubuntu; the
Windows `.venv` is CPU-only and unchanged. Training deps are intentionally NOT
in pyproject.toml — install them here.

Env: `/mnt/d/Grad-Project/HerHealthGPT/.venv-ft` (Python 3.11 via uv).
Why 3.11: WSL system Python is 3.14, unsupported by torch/Unsloth.
```
Then the four command blocks above with their expected outputs, and a one-line troubleshooting note: "If `import unsloth` fails on a Qwen3.5 architecture error, that is the go/no-go signal for the Task 3 fallback."

- [ ] **Step 6: Commit**

```bash
git add docs/wsl_finetune_setup.md
git commit -m "docs: WSL2 + Unsloth environment setup for local fine-tune"
```

---

## Task 2: Data prep script + unit tests

**Files:**
- Create: `scripts/prepare_ft_data.py`
- Create: `tests/test_prepare_ft_data.py`
- Modify: `.gitignore` (add `data/ft/`)
- Generates: `data/ft/en/train.jsonl`, `data/ft/en/val.jsonl`

**Interfaces:**
- Produces (imported by the test; consumed as JSONL by Task 3):
  - `to_chat_record(row: dict) -> dict` → `{"messages": [{"role": "system", "content": <instruction>}, {"role": "user", "content": <input>}, {"role": "assistant", "content": <output>}], "category": <category>}`
  - `split_train_val(rows: list[dict], val_frac: float, seed: int) -> tuple[list[dict], list[dict]]` — balanced per category, deterministic.
- Output JSONL schema: one `{"messages": [...], "category": ...}` object per line.

- [ ] **Step 1: Write the failing test**

Create `tests/test_prepare_ft_data.py`:
```python
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
import prepare_ft_data as p  # noqa: E402


def _row(cat="menstrual", n=0):
    return {"instruction": "You are a supportive assistant.",
            "input": f"question {n}", "output": f"answer {n}",
            "category": cat, "source_dataset": "MENST", "source_row_id": str(n)}


def test_to_chat_record_maps_three_roles():
    rec = p.to_chat_record(_row(n=1))
    roles = [m["role"] for m in rec["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert rec["messages"][0]["content"] == "You are a supportive assistant."
    assert rec["messages"][1]["content"] == "question 1"
    assert rec["messages"][2]["content"] == "answer 1"
    assert rec["category"] == "menstrual"


def test_split_is_balanced_and_deterministic():
    rows = ([_row("menstrual", i) for i in range(100)]
            + [_row("pcos", i) for i in range(100)]
            + [_row("fertility", i) for i in range(100)])
    train, val = p.split_train_val(rows, val_frac=0.05, seed=42)
    assert len(val) == 15  # 5% of 300
    from collections import Counter
    assert Counter(r["category"] for r in val) == {"menstrual": 5, "pcos": 5, "fertility": 5}
    assert len(train) == 285
    # deterministic
    train2, val2 = p.split_train_val(rows, val_frac=0.05, seed=42)
    assert [r["input"] for r in val] == [r["input"] for r in val2]
    # disjoint
    val_ids = {(r["category"], r["input"]) for r in val}
    assert not any((r["category"], r["input"]) in val_ids for r in train)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_prepare_ft_data.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'prepare_ft_data'`.

- [ ] **Step 3: Write the script**

Create `scripts/prepare_ft_data.py`:
```python
"""HerHealthGPT-LU — format the silver FT corpus for Qwen QLoRA training.

Reads the leakage-cleaned corpus (HerHealthGPT-LU_seed/ft_corpus_v1.jsonl) and
emits Qwen chat-message records plus a balanced train/val split. Thinking-mode
is applied later at train time (enable_thinking=False); this step only shapes
the messages, so it is pure Python and unit-tested on the Windows venv.

Never reads seeds_en_v1.* (the frozen benchmark) — training/eval stay disjoint.

Run: python scripts/prepare_ft_data.py
Outputs (gitignored):
  data/ft/en/train.jsonl  {"messages": [...], "category": ...}
  data/ft/en/val.jsonl    same schema, balanced 5% held-out
"""
import argparse
import json
import pathlib
import random
from collections import defaultdict

CORPUS = pathlib.Path("HerHealthGPT-LU_seed/ft_corpus_v1.jsonl")
OUT_DIR = pathlib.Path("data/ft/en")


def to_chat_record(row: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": row["instruction"]},
            {"role": "user", "content": row["input"]},
            {"role": "assistant", "content": row["output"]},
        ],
        "category": row["category"],
    }


def split_train_val(rows: list[dict], val_frac: float, seed: int):
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    rng = random.Random(seed)
    train, val = [], []
    for cat in sorted(by_cat):
        items = by_cat[cat][:]
        rng.shuffle(items)
        n_val = round(len(items) * val_frac)
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    return train, val


def _write(path: pathlib.Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(to_chat_record(rec), ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rows = [json.loads(l) for l in CORPUS.open(encoding="utf-8")]
    train, val = split_train_val(rows, args.val_frac, args.seed)
    _write(OUT_DIR / "train.jsonl", train)
    _write(OUT_DIR / "val.jsonl", val)
    print(f"corpus={len(rows)} train={len(train)} val={len(val)} -> {OUT_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_prepare_ft_data.py -v
```
Expected: PASS (2 passed).

- [ ] **Step 5: Generate the real split and eyeball it**

Run:
```bash
.venv/Scripts/python.exe scripts/prepare_ft_data.py
```
Expected: `corpus=2700 train=2565 val=135 -> data\ft\en`.

- [ ] **Step 6: Add data outputs to .gitignore**

Add to `.gitignore`:
```
data/ft/
```

- [ ] **Step 7: Commit**

```bash
git add scripts/prepare_ft_data.py tests/test_prepare_ft_data.py .gitignore
git commit -m "feat: FT data prep (corpus -> Qwen chat messages + balanced split)"
```

---

## Task 3: Training script + 10-step smoke gate (go/no-go)

**Files:**
- Create: `scripts/train_qlora.py`
- Modify: `.gitignore` (add `models/`)
- Generates: `models/_smoke/` (throwaway) + `run_config.json`

**Interfaces:**
- Consumes: `data/ft/en/{train,val}.jsonl` from Task 2.
- Produces (used by Task 4/5): a CLI `python scripts/train_qlora.py [--model M] [--epochs N] [--max-steps K] [--output DIR] [--seed S]`. Saves a LoRA adapter + tokenizer to `--output`, and writes `<output>/run_config.json` = `{model, epochs, max_steps, lr, seed, git_sha, per_device_batch, grad_accum}`.

- [ ] **Step 1: Confirm the model repo id**

Verify the exact Hugging Face id before running. Run:
```bash
wsl.exe -d Ubuntu -- bash -lc "cd /mnt/d/Grad-Project/HerHealthGPT && \
  .venv-ft/bin/python -c 'from huggingface_hub import model_info; print(model_info(\"Qwen/Qwen3.5-9B\").id)'"
```
Expected: `Qwen/Qwen3.5-9B`. If this errors with "repo not found", set the correct id from huggingface.co/Qwen and pass it via `--model` in later steps (default in the script stays `Qwen/Qwen3.5-9B`).

- [ ] **Step 2: Write the training script**

Create `scripts/train_qlora.py`:
```python
"""HerHealthGPT-LU — QLoRA fine-tune of Qwen3.5-9B on the English FT corpus (M3).

Recipe (parent spec §2C, MenstLLaMA-derived): 4-bit NF4, LoRA r=16, lr 2e-4,
warmup 0.03, max grad norm 0.3, max seq 2048, paged AdamW 8-bit, cosine, bf16.
Thinking-mode OFF so train/eval templates match. Trains on responses only.

Run inside WSL (.venv-ft). Smoke gate first:
  python scripts/train_qlora.py --max-steps 10 --output models/_smoke
Full run:
  python scripts/train_qlora.py --epochs 3 --output models/qwen3.5-9b-herhealth-en-lora

Fallback: --model Qwen/Qwen2.5-7B-Instruct if the primary won't load/train.
"""
import argparse
import json
import pathlib
import subprocess

from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

MAX_SEQ = 2048


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--data-dir", default="data/ft/en")
    ap.add_argument("--output", default="models/qwen3.5-9b-herhealth-en-lora")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    args = ap.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=MAX_SEQ,
        load_in_4bit=True, dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=args.seed,
    )

    data_dir = pathlib.Path(args.data_dir)
    ds = load_dataset("json", data_files={
        "train": str(data_dir / "train.jsonl"),
        "val": str(data_dir / "val.jsonl")})

    def fmt(batch):
        texts = [tokenizer.apply_chat_template(m, tokenize=False,
                    add_generation_prompt=False, enable_thinking=False)
                 for m in batch["messages"]]
        return {"text": texts}

    ds = ds.map(fmt, batched=True, remove_columns=ds["train"].column_names)

    cfg = SFTConfig(
        output_dir=str(pathlib.Path(args.output) / "_hf"),
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.03, num_train_epochs=args.epochs, max_steps=args.max_steps,
        learning_rate=args.lr, bf16=True, optim="paged_adamw_8bit",
        max_grad_norm=0.3, lr_scheduler_type="cosine", seed=args.seed,
        logging_steps=5, eval_strategy="steps", eval_steps=50,
        save_strategy="no", dataset_num_proc=2, max_seq_length=MAX_SEQ,
        report_to="none",
    )
    trainer = SFTTrainer(model=model, tokenizer=tokenizer,
                         train_dataset=ds["train"], eval_dataset=ds["val"],
                         dataset_text_field="text", args=cfg)
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    trainer.train()

    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    (out / "run_config.json").write_text(json.dumps({
        "model": args.model, "epochs": args.epochs, "max_steps": args.max_steps,
        "lr": args.lr, "seed": args.seed, "per_device_batch": args.batch,
        "grad_accum": args.grad_accum, "git_sha": git_sha(),
    }, indent=2))
    print(f"saved adapter -> {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Add model outputs to .gitignore**

Add to `.gitignore`:
```
models/
```

- [ ] **Step 4: Run the 10-step smoke gate (GO/NO-GO)**

Run:
```bash
wsl.exe -d Ubuntu -- bash -lc "cd /mnt/d/Grad-Project/HerHealthGPT && \
  .venv-ft/bin/python scripts/train_qlora.py --max-steps 10 --output models/_smoke"
```
Expected: model downloads, training logs ~10 steps with a decreasing loss, prints `saved adapter -> models/_smoke`.

**GO/NO-GO decision:**
- **GO** — it loads and steps → continue to Task 4 with `Qwen/Qwen3.5-9B`.
- **NO-GO** — if it fails on the model architecture (GatedDeltaNet unsupported) and is not resolved within ~2–3 h, rerun this exact command with `--model Qwen/Qwen2.5-7B-Instruct`. Expected then: same success on the fallback. Record which base was used.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_qlora.py .gitignore
git commit -m "feat: Unsloth QLoRA training script + 10-step smoke gate"
```

---

## Task 4: Full training run (seed 1)

**Files:**
- Generates: `models/qwen3.5-9b-herhealth-en-lora/` (adapter + `run_config.json`)

**Interfaces:**
- Consumes: `scripts/train_qlora.py` (Task 3), `data/ft/en/*` (Task 2).
- Produces (used by Task 5): the M3 adapter directory loadable by Unsloth.

- [ ] **Step 1: Launch the full 3-epoch run**

Use the base that passed Task 3's gate (default primary shown; append `--model Qwen/Qwen2.5-7B-Instruct` if the fallback triggered). Run:
```bash
wsl.exe -d Ubuntu -- bash -lc "cd /mnt/d/Grad-Project/HerHealthGPT && \
  .venv-ft/bin/python scripts/train_qlora.py --epochs 3 \
  --output models/qwen3.5-9b-herhealth-en-lora"
```
Expected: training runs to completion over 3 epochs (~480 optimizer steps at effective batch 16 over 2,565 examples), periodic eval loss logged, ends with `saved adapter -> models/qwen3.5-9b-herhealth-en-lora`.

- [ ] **Step 2: Confirm the eval loss trended down (not up)**

Read the trainer's logged `eval_loss` values from the run output. Expected: eval loss decreases then flattens. If it clearly rises across the last epoch (overfitting on silver data), note it — Task 6 records the decision; a rerun with `--epochs 2` is the remedy, not a plan change.

- [ ] **Step 3: Verify the adapter + run_config were written**

Run:
```bash
wsl.exe -d Ubuntu -- bash -lc "cd /mnt/d/Grad-Project/HerHealthGPT && \
  ls models/qwen3.5-9b-herhealth-en-lora && cat models/qwen3.5-9b-herhealth-en-lora/run_config.json"
```
Expected: `adapter_config.json`, `adapter_model.safetensors`, tokenizer files, and a `run_config.json` showing the model id, seed 3407, and the git SHA.

- [ ] **Step 4: Commit run_config (adapter itself is gitignored)**

```bash
git add -f models/qwen3.5-9b-herhealth-en-lora/run_config.json
git commit -m "chore: record M3 run_config (seed 1) for reproducibility"
```

---

## Task 5: Sanity-generation check

**Files:**
- Create: `scripts/sanity_generate.py`

**Interfaces:**
- Consumes: the adapter dir from Task 4.
- Produces: prints prompt→answer pairs for manual review (no assert; this is a human verification gate).

- [ ] **Step 1: Write the sanity-generation script**

Create `scripts/sanity_generate.py`:
```python
"""HerHealthGPT-LU — sanity generations from the M3 adapter.

Loads the fine-tuned adapter and generates on a few in-domain prompts across
the three categories, thinking-mode OFF (matches training/eval). Manual review:
answers should be coherent, on-topic, and recommend seeing a clinician where
appropriate rather than giving unsafe directive advice.

Run inside WSL (.venv-ft):
  python scripts/sanity_generate.py --adapter models/qwen3.5-9b-herhealth-en-lora
"""
import argparse

from unsloth import FastLanguageModel

SYSTEM = ("You are a supportive women's-health information assistant. Answer the "
          "patient's question clearly and safely, and recommend seeing a clinician "
          "for diagnosis or treatment decisions.")
PROMPTS = [
    "I haven't had my period in 4 months and I'm not pregnant. Should I worry?",   # menstrual
    "I have irregular cycles, acne, and extra hair growth. What could this be?",    # pcos
    "We've been trying to conceive for over a year with no luck. What now?",        # fertility
    "Is it normal for periods to hurt so much I can't go to work?",                 # menstrual
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="models/qwen3.5-9b-herhealth-en-lora")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter, max_seq_length=2048, load_in_4bit=True, dtype=None)
    FastLanguageModel.for_inference(model)

    for q in PROMPTS:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": q}]
        inputs = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_tensors="pt").to("cuda")
        out = model.generate(input_ids=inputs, max_new_tokens=args.max_new_tokens,
                             use_cache=True, temperature=0.7, do_sample=True)
        text = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        print("\n=== Q:", q, "\nA:", text.strip())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and review the answers**

Run:
```bash
wsl.exe -d Ubuntu -- bash -lc "cd /mnt/d/Grad-Project/HerHealthGPT && \
  .venv-ft/bin/python scripts/sanity_generate.py --adapter models/qwen3.5-9b-herhealth-en-lora"
```
Expected: four coherent, on-topic answers; each addresses the category correctly and points to a clinician for diagnosis/treatment. If answers are incoherent, off-topic, or contain unsafe directive advice, M3 is NOT done — revisit Task 4 (epochs/lr) before proceeding.

- [ ] **Step 3: Commit**

```bash
git add scripts/sanity_generate.py
git commit -m "feat: M3 sanity-generation check across the three categories"
```

---

## Task 6: Documentation — pipeline row + paper/method note

**Files:**
- Create: `docs/finetune_run_notes.md`
- Modify: `README.md` (pipeline table)

**Interfaces:** none (docs only).

- [ ] **Step 1: Write the run-notes / paper file**

Create `docs/finetune_run_notes.md`:
```markdown
# M3 fine-tune — run notes (for the paper's method section)

- **Trainer:** Unsloth (local single-GPU), **not** Llama-Factory as the parent
  spec stated. QLoRA recipe unchanged: 4-bit NF4, LoRA r=16 alpha=16, lr 2e-4,
  warmup 0.03, max grad norm 0.3, max seq 2048, paged AdamW 8-bit, cosine, bf16.
- **Hardware:** local RTX 5000 Ada (32 GB), WSL2 Ubuntu, Python 3.11 (uv).
- **Data:** HerHealthGPT-LU_seed/ft_corpus_v1.jsonl (2,700 EN pairs, balanced),
  95/5 train/val split (seed 42). Benchmark seeds excluded (dual-key leakage filter).
- **Base model used:** <Qwen/Qwen3.5-9B | Qwen/Qwen2.5-7B-Instruct — fill in which>.
- **Training seed:** 3407 (this run = seed 1; seeds 2–3 are reruns with --seed).
- **Epochs:** <3 | adjusted — fill in>. Eval-loss trend: <fill in>.
- **Thinking mode:** OFF at train and inference (enable_thinking=False).
```
Fill the `<...>` fields from the actual Task 3–5 outcomes before committing.

- [ ] **Step 2: Add a pipeline row to README**

In `README.md`, add a row after step 6 in the "HerHealthGPT-LU pipeline" table:
```markdown
| 6b | `scripts/prepare_ft_data.py` + `scripts/train_qlora.py` + `scripts/sanity_generate.py` | `models/qwen3.5-9b-herhealth-en-lora/` | Local M3 QLoRA fine-tune (Unsloth/WSL2). See `docs/wsl_finetune_setup.md` and `docs/finetune_run_notes.md`. |
```

- [ ] **Step 3: Commit**

```bash
git add docs/finetune_run_notes.md README.md
git commit -m "docs: M3 run notes + README pipeline row for the local fine-tune"
```

---

## Self-Review

**Spec coverage:** §1 scope → Tasks 1–6; §4 env → Task 1; §5 data prep → Task 2; §6 training + go/no-go gate → Tasks 3–4; §7 outputs + sanity → Tasks 4–5; §8 testing (unit/smoke/verify) → Task 2 Step 4 / Task 3 Step 4 / Task 5 Step 2; §9 paper note → Task 6; §10 risks (Python 3.11 pin, fallback flag, overfitting, gitignored artifacts, thinking-off) all mapped to concrete steps. No gaps.

**Placeholders:** The only intentional fill-in is `docs/finetune_run_notes.md`'s `<...>` fields, which capture *runtime outcomes* (which base model ran, actual epochs/eval loss) — these cannot be known at plan time and Step 1 instructs filling them before commit. The `--model` repo-id confirmation (Task 3 Step 1) is a runtime check with a concrete default, not a placeholder.

**Type consistency:** `to_chat_record` / `split_train_val` signatures match between the test (Task 2 Step 1) and the script (Task 2 Step 3). The `{"messages": [...], "category": ...}` JSONL schema produced in Task 2 is exactly what `train_qlora.py`'s `fmt()` consumes via `batch["messages"]` in Task 3. `--output` in Task 3 is the `--adapter` input in Task 5. `run_config.json` keys are consistent between Task 3 (write) and Task 4 Step 3 (read).
