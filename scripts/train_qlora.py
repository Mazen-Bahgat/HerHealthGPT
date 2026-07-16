"""HerHealthGPT-LU — QLoRA fine-tune of Qwen3.5-9B on the English FT corpus (M3).

Recipe (parent spec §2C, MenstLLaMA-derived): 4-bit NF4, LoRA r=16, lr 2e-4,
warmup 0.03, max grad norm 0.3, max seq 2048, paged AdamW 8-bit, cosine, bf16.
Thinking-mode OFF so train/eval templates match. Trains on responses only.

Runs inside WSL (HerHealthUbuntu, root) with the pinned .venv-ft stack.
Smoke gate first:
  .venv-ft/bin/python scripts/train_qlora.py --max-steps 10 --output models/_smoke
Full run:
  .venv-ft/bin/python scripts/train_qlora.py --epochs 3 --output models/qwen3.5-9b-herhealth-en-lora
"""
from __future__ import annotations

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
        dataset_text_field="text", report_to="none",
    )
    trainer = SFTTrainer(model=model, tokenizer=tokenizer,
                         train_dataset=ds["train"], eval_dataset=ds["val"],
                         args=cfg)
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
