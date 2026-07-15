# Design: New-benchmark metrics + EN / EN+FR+AR fine-tunes

Date: 2026-07-15. Deadline context: paper submission moved to 2026-07-17; all
stages below target end-of-day 2026-07-15 (Stage 5 gated on team translations).
Approved by Mazen with a checkpoint after every stage: report results, get
approval, then start the next stage.

## Goal

Produce paper-ready metrics on the team-revised benchmark
(`Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/gold_seeds_styled_labeled.csv`)
and fine-tune Qwen3.5-9B on the new canonical QA splits — first English-only,
then jointly on EN+FR+AR once the team's translations arrive.

## Facts established from the data (verified 2026-07-15)

- The new labeled benchmark has the **same 540 questions** (90 seeds x 6
  styles) as `Train_Val_Dataset/gold_seeds_styled_labeled.csv`. Only the gold
  fields changed: revised risk distribution (348 routine / 174 see-doctor / 18
  urgent vs old 342/180/18), revised `gold_condition`, a newly filled
  `gold_action` column, and a different column order
  (`Question,Answer,Topic,Keywords,Style,gold_condition,gold_risk_level,gold_action,requires_clarification`).
- Because zero-shot predictions are independent of gold labels, the existing
  M2 raw responses (`Train_Val_Dataset/M2_gss_en.jsonl`) can be re-joined
  against the new gold and re-scored **without any GPU run**.
- FT splits: `200_Seed_Dataset/train_canonical.csv` (480 rows) and
  `validation_canonical.csv` (120 rows), plain Question/Answer/Topic/Keywords.
- Leakage found: 1 train + 2 val questions appear verbatim (case-insensitive
  exact match) in the benchmark; 4 questions are duplicated between train and
  val. These must be dropped before training.
- Previous full FT run: 2,565 chat pairs x 3 epochs, lr 2e-4 → ~2.5 h. That
  calibrates the runtime estimates below.

## Decisions (made with Mazen, 2026-07-15)

1. **Translations:** the team translates FR and AR themselves. We produce
   machine-ingestible handoff templates and an ingest path.
2. **FT target format:** plain Q→A chat pairs (system prompt + question →
   answer). No JSON-task wrapping. Eval JSON compliance relies on the
   zero-shot prompt (M2 already parses at 100%); a post-FT parse_ok check is
   part of Stage 4/5 acceptance.
3. **Runs:** two fine-tunes — EN-only first, then joint EN+FR+AR.

## Stages

Each stage ends with a report to Mazen and an explicit approval gate before
the next stage starts.

### Stage 1 — Re-score M2 zero-shot on the new gold (no GPU)

- Generalize `scripts/convert_gold_seeds_styled.py` to read the new file
  (column order differs; carry `gold_action` through) via a `--src`/`--out`
  parameterization, keeping the old paths as defaults.
- Regenerate the benchmark JSONL into `200_Seed_Dataset/gold_seeds_styled.jsonl`.
- Patch the existing M2 raw responses onto the new gold by `item_id` (the
  established patch-in-place approach; item identity holds because the 540
  questions are unchanged).
- Run `scripts/evaluate.py` and `scripts/safety_metrics.py`.
- Outputs in `200_Seed_Dataset/`: `M2_gss_en.jsonl`, `M2_gss_summary.json`,
  `safety_M2_gss.json`, `safety_M2_gss.md`.
- Acceptance: parse_ok 1.000 (responses unchanged), all 540 items joined, new
  majority baselines reported from the new gold distribution.

### Stage 2 — FT data prep (plain Q→A chat pairs)

- New script `scripts/prepare_ft_data_v2.py`:
  - Reads `train_canonical.csv` / `validation_canonical.csv` (or `--train`/
    `--val` overrides for FR/AR files with identical schema, plus `--lang`).
  - Drops rows whose Question matches a benchmark question
    (case-insensitive exact match against all 540 styled questions), and
    dedups questions appearing in both train and val (val wins, so validation
    stays untouched; duplicates leave train).
  - Writes a leakage/dedup log (`data/ft/en_v2/leakage_log.csv`) naming every
    dropped row and why.
  - Emits Qwen chat-message JSONL — same record shape as
    `scripts/prepare_ft_data.py` (system/user/assistant messages, thinking
    off) — to `data/ft/en_v2/{train,val}.jsonl`.
- Expected sizes after cleaning: ~476 train / 118 val (exact counts from the
  log).
- Acceptance: unit tests (Windows `.venv`, pytest) for leakage drop, train/val
  dedup direction, schema of emitted records; zero benchmark questions in the
  output (re-checked by a test, not by eye).

### Stage 3 — Translation handoff for the team

- Export the leakage-cleaned train+val rows to
  `Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/translation_handoff_v2/{fr,ar}.csv`
  with columns: `row_id`, `split`, `Question`, `Answer`, `Topic`, `Keywords`,
  `Question_translated` (empty), `Answer_translated` (empty).
- `row_id` is stable (split + source row index) so returned files join back
  deterministically; Topic/Keywords stay in English (metadata, not shown to
  the model as text to translate).
- A short `README.md` in the handoff folder states the rules (translate
  Question/Answer only, keep row_id untouched, CSV stays UTF-8).
- Ingest path: `prepare_ft_data_v2.py --lang fr --train <returned fr.csv>`
  maps `*_translated` columns onto Question/Answer and emits
  `data/ft/fr_v2/{train,val}.jsonl` (same for `ar`).
- Acceptance: round-trip test — a synthetic filled template ingests to valid
  chat JSONL with the translated text in place.

### Stage 4 — Fine-tune run 1: EN-only (GPU, ~25–30 min)

- Env: WSL `Ubuntu` / sw2 / `/home/sw2/ft-train-venv` (per
  `docs/ENVIRONMENTS.md`).
- `scripts/train_qlora.py` unchanged: **3 epochs, lr 2e-4, batch 2** on
  `data/ft/en_v2/` (~476 rows ≈ 19% of the previous corpus → ~28 min; well
  under the 2.5 h ceiling).
- Smoke gate first: `--max_steps 5` must complete cleanly before the full run.
- Adapter → `models/qwen3.5-9b-herhealth-en-v2-lora`.
- Eval: `scripts/run_local_inference.py --adapter ...` on the new benchmark
  JSONL → `evaluate.py` + `safety_metrics.py` → comparison vs Stage-1 M2
  numbers (label `M3en-v2` vs `M2-gss`).
- Acceptance: adapter saved with run manifest; post-FT parse_ok reported (if
  it drops materially below 1.0, that is a finding to report at the gate, not
  silently absorb); side-by-side metric table delivered.

### Stage 5 — Fine-tune run 2: joint EN+FR+AR (GPU, ~1 h; gated on translations)

- Concatenate `data/ft/{en,fr,ar}_v2/train.jsonl` (~1,430 rows) and the three
  val files; shuffle with a fixed seed.
- **2 epochs, lr 2e-4, batch 2** — larger corpus, and each English answer
  recurs once per language, so fewer epochs to limit memorization and keep
  runtime ~1 h.
- Adapter → `models/qwen3.5-9b-herhealth-enfrar-lora`; same eval loop
  (label `M3ml-v2`), evaluated on the **English** benchmark.
- If translations are not back today: Stages 1–4 stand alone; Stage 5 is a
  single command once the templates return.

## Error handling

- Converter fails loudly on unexpected columns or a row count ≠ 540.
- Patch step fails loudly on any unmatched `item_id` (all 540 must join).
- Ingest fails loudly on missing/empty `*_translated` cells or unknown
  `row_id`s, listing offending rows.
- GPU runs use the established smoke-gate pattern before full runs.

## Testing

- Windows `.venv` pytest for all pure-Python logic (converter column
  handling, leakage/dedup, handoff round-trip). Existing suites must stay
  green.
- GPU stages verified by smoke gate + run manifest + post-run eval metrics.

## Addendum (2026-07-15, after team delivered styled splits)

Decisions made with Mazen when the styled EN dataset landed:

- **Canonical file layout changed:** benchmark = `200_Seed_Dataset/Test/gold_seeds_styled_labeled.csv`
  (verified byte-identical conversion to the Stage-1 benchmark JSONL, so
  Stage-1 results stand); FT train = `Train/train_canonical_styled.csv`
  (2,880 rows = 480 seeds x 6 styles); FT val =
  `validate/validation_canonical_styled.csv` (720 rows = 120 x 6).
- **FT input is styled-only** — the plain canonical CSVs are superseded.
- **EN run: 2 epochs** (not 3): 2,880 rows ≈ 1.9 h, and the 6 style variants
  per seed already act as augmentation.
- **Stage 3 (handoff templates) dropped:** the team translates the three
  styled files themselves in the same schema; `prepare_ft_data_v2.py --lang
  fr|ar --train/--val <translated csv>` consumes them directly (the
  `*_translated`-column path remains only as a fallback if handoff-style
  files ever appear).
- Answer-level leakage verified zero: none of the 90 benchmark seeds'
  answers appear in the styled train/val files.

## Out of scope (noted as paper limitations, not built today)

- FR/AR translation of the 540-row benchmark itself — both fine-tuned models
  are evaluated on the English benchmark only.
- Re-running M3/M3-v2 (old adapters) on the new benchmark.
- Any JSON-task or mixed-format FT corpus (M3-v2 recipe).
