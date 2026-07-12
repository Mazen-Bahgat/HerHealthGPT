# HerHealthGPT-LU — Design Spec

**Date:** 2026-07-06 · **Deadline:** LUHME 2026 submission, 15 July 2026 (OpenReview, ACL template, 8 pages max, double-blind)
**Team:** Mazen, Hana, Hassan, Mariam · **Supervisor:** Dr. Rahatara Ferdousi

## 1. Paper framing

**Working title:** *HerHealthGPT-LU: A Multilingual Benchmark for Evaluating LLM Understanding of Women's-Health Symptom Communication (Menstrual, PCOS, and Fertility)*

**Research question:** How well do LLMs interpret clinical, layperson, indirect, ambiguous, and emotionally-expressed descriptions of menstrual, PCOS/hormonal, and fertility concerns across English, French, and Arabic — and does multilingual fine-tuning reduce misunderstanding?

**Scope — three categories:** `menstrual`, `pcos` (PCOS/hormonal), `fertility` (incl. infertility). Balanced 1/3 each in the benchmark. The three-way split increases coverage of women's-health LU while keeping the PCOS and fertility axes that anchor the clinical-risk analysis.

**Contributions:**
1. **Dataset** — a human-validated multilingual benchmark of symptom expressions across three women's-health categories.
2. **Evaluation** — metrics targeting misunderstanding, safety, clarification behavior, and cross-language/cross-style consistency.
3. **Analysis** — systematic error taxonomy of how and why LLMs misunderstand women's-health language.
4. **Adaptation evidence** — whether LoRA fine-tuning on multilingual women's-health data reduces misunderstanding.

**Distinction from MenstLLaMA (matters now that menstrual is in scope):** MenstLLaMA does English menstrual-health *answer generation/education*; we do *multilingual understanding evaluation* across styles and languages with clinical-risk/safety metrics. The menstrual category here is an evaluation axis, not an education target — the overlap is surface-level and the Related Work must state this explicitly.

Framed as a language-understanding problem in a high-stakes domain (LUHME topics: socio-cultural aspects of LU; effects and risks of misunderstanding; LLMs for generating/evaluating linguistic data).

## 2. Datasets — source files and two strictly separated sets

### 2.0 Source files on hand (verified)

| File | Identity | Rows | Schema | Role |
|------|----------|------|--------|------|
| `train24K.csv` | **MENST**, paraphrase-augmented | 22,412 | Question, Answer, Age Group, Region, Topic, Keywords, Document Id, LLM Used, Set | Silver FT corpus backbone |
| `training2K.csv` | MENST raw (pre-augmentation) | 1,896 | Set, Document Id, Question, Answer, Age Group, Region, Keywords, LLM Used | Seed mining + dedup key |
| `test.csv` | MENST-style gold QA | 100 | Questions, Human, Topic | Reference only — **NOT** our benchmark |
| `train-…parquet` | ChatDoctor **iCliniq** (triple-answer) | 7,321 | input, answer_icliniq, answer_chatgpt, answer_chatdoctor | Patient-phrasing seed mining |
| `HealthCareMagic-100k-en.jsonl` | ChatDoctor **HealthCareMagic** | 112,165 | `{"text": "<human>: … <bot>: …"}` | Patient-phrasing seed mining (~21K women's-health lines, ~19%) |

**Critical paraphrase-family signal:** MENST's 22,412 rows collapse to **1,721 unique answers** (~13× duplication), while Document Id has only 104 values. The correct paraphrase-family key is therefore the **(normalized) Answer text**, not Document Id — Document Id is far too coarse to dedup or gate leakage. This is the concrete mechanism behind the family-level rule.
**PCOS/infertility coverage in MENST (keyword scan):** pcos 821, polycystic 159, ovar 829, hormon 2,427, fertil 696, infertil 119, conceive 97, pregnan 1,545 rows — enough for a strong silver corpus on PCOS/hormonal; infertility is thinner and leans on iCliniq/HCM mining.

### 2A. Evaluation benchmark (gold, small, human-validated)

- **Three categories** — `menstrual`, `pcos`, `fertility`. A frozen **English nucleus of 90 seeds (30/category)** already exists from the seed package (`seeds_en_v1`), expandable toward ~120–150 total (~40–50/category) if time permits.
- **Sourcing (done in seed v1):** evidence-only from MENST (`training2K`/`train24K`), iCliniq, HealthCareMagic patient turns; provenance (`source_dataset + source_row_id`) on every row; near-dup dedup logged; grounding drafted only against NHS/CDC/NICHD with `NEEDS_GROUNDING` where uncertain.
- **5 expression styles per seed** (+ canonical): clinical, layperson, indirect/culturally sensitive, ambiguous, emotionally concerned.
- **⚠️ Style variants must be regenerated (seed v1 used templates).** The seed package generated variants by dropping extracted claim tokens into fixed frames (e.g. *"I have {X}."*, *"I am really worried and scared about {X}."*). This fails our design in two ways: (a) it collapses to ~51 distinct strings across 90 seeds and reads as obvious scaffolding, and (b) it can corrupt meaning (canonical *"Does period pain affect fertility?"* → *"I have period pain and fertility"* turns a factual question into an incoherent symptom report; the miscarriage/AMH/laparoscopy context is dropped). **Fix:** regenerate the 5 variants with an LLM from the *canonical patient text*, under a strict rubric — preserve every clinical claim in the canonical, change only register, sound like a real person, no added/dropped facts — then human-review each (already in the Day-3 plan). The templated `style_text` is discarded; `canonical_text` and all provenance are kept.
- **Gold labels are written from the canonical text, not the variant** (variants intentionally shorten and would lose salient detail like AMH values).
- **Translation to French and Arabic**: GPT/NLLB-200 first pass → validated by native speaker (fluency) + linguistic reviewer (meaning, ambiguity, cultural fit), documented rubric. Nucleus total ≈ 90 seeds × 6 rows × 3 languages ≈ 1,620 items (≈1,950 if expanded to ~130 seeds).
- **Item schema (extends seed v1):** `seed_id, category, source_dataset, source_row_id, confidence_tier, dedup_group_id, canonical_text, style, style_text, language, gold_condition, gold_risk_level {routine | see-doctor | urgent}, gold_action, requires_clarification {yes|no}, source_url, seed_answer_hash, needs_grounding_flag, validation_status`. (Seed v1 already provides everything except the multilingual, gold_risk_level, gold_action, requires_clarification, and regenerated style_text fields.)

### 2B. Fine-tuning corpus (silver, larger, spot-checked)

- Built by the join procedure in §2C → target **~900 QA pairs per category (~2,700 total)** after dedup, per the seed package's finalized decision (eligible Clear pool supports up to ~1,154/category if needed). Balanced across the three categories.
- Machine-translate to FR/AR; spot-check ~5%. No full human validation (stated honestly in the paper).
- **Leakage control — two complementary keys.** The seed package blocks by `source_dataset + source_row_id` (exact originating row), logged in `leakage_note.md` for all 90 seeds. This is necessary but *not sufficient* for MENST, because one MENST answer has ~13 paraphrase siblings under different row IDs. So we **also** block by `seed_answer_hash` (normalized-answer sha1): any MENST row whose answer hash matches a seed is excluded with all siblings. Fine-tuning pulls apply both filters. No benchmark text (any language) appears in 2B.

### 2C. Joining the three datasets — step-by-step (Hassan owns; Hana joins for translation)

The three corpora play **different roles** and are NOT concatenated blindly — MENST is education-style QA (good for silver training), while iCliniq/HCM are messy real patient dialogues (good for authentic phrasing). The pipeline produces two artifacts: the silver FT corpus (2B) and the seed-phrasing pool for 2A.

**Step 1 — Normalize each source to a common interim schema** `{qa_id, source, raw_question, raw_answer, topic, meta}`:
- MENST (`train24K.csv`): map Question→raw_question, Answer→raw_answer, Topic→topic; keep Age Group/Region/Keywords in meta. `source="menst"`.
- iCliniq (parquet): input→raw_question; use `answer_icliniq` as raw_answer (the real doctor answer; ignore answer_chatgpt/answer_chatdoctor for training to avoid GPT-style leakage). `source="icliniq"`.
- HealthCareMagic (jsonl): parse each `text` on `<human>:` / `<bot>:` → raw_question / raw_answer. `source="hcm"`.

**Step 2 — Domain filter to PCOS/hormonal + infertility.** Apply a keyword+regex gate (`pcos|polycystic|ovar|hormon|androgen|irregular period|amenorrh|infertil|fertil|conceiv|ovulat|endometrios|menstru`) over question+answer. Expected yield: MENST ~2–3K rows, HCM ~21K→ precision-filter to a few K, iCliniq a few hundred. Keep a generous recall here; precision is tightened in Step 4.

**Step 3 — Deduplicate at the paraphrase-family level (MENST only).** Compute `answer_hash = sha1(normalize(raw_answer))` where normalize = lowercase, strip punctuation/whitespace. Group by answer_hash → each group is one paraphrase family (recall: 1,721 families across 22K rows). For the **FT corpus**, keep *all* paraphrases (variety is the point of training data) but record answer_hash on every row so leakage gating works. For **seed mining**, sample one representative per family to avoid over-sampling repeated content.

**Step 4 — Quality/precision filter for the silver corpus.** Drop rows with answer length < 20 chars or > 1,500 chars, non-English (langid), duplicated exact QA pairs across sources, and obvious junk (URLs-only, "consult a doctor" one-liners with no content). Deduplicate near-identical questions across sources with a MinHash/embedding threshold. Target after this step: **3,000–5,000 QA pairs** balanced as much as possible between PCOS/hormonal and infertility (up-sample infertility from iCliniq/HCM since MENST is thin there).

**Step 5 — Build the seed-phrasing pool for 2A (separate from 2B).** From the family-deduped pool, Hassan hand-selects ~130 seeds whose *phrasing* is authentic and whose clinical content maps cleanly to an NHS/CDC/NIH gold label. Record each seed's originating `answer_hash` in `seed_answer_hash`. These seeds are the basis for the 5-style expansion (§2A) — the model never trains on them.

**Step 6 — Apply leakage exclusion.** Remove from the 2B silver corpus every row whose `answer_hash` ∈ {seed_answer_hash}. This is the family-level guarantee: excluding one seed removes all 13-ish paraphrase siblings. Log counts removed.

**Step 7 — Format for Llama-Factory.** Convert 2B to the instruction JSON format (`{"instruction","input","output"}`), with an optional system prompt fixing the empathetic-but-safe persona. Hana's translation pipeline then produces FR/AR copies of 2B (spot-checked 5%). Freeze as `ft_corpus_v1`.

**Outputs:** `benchmark_seeds.csv` (130 rows → drives 2A), `ft_corpus_v1.jsonl` (EN+FR+AR silver, leakage-clean), and `leakage_log.csv`. All under git in the repo.

## 3. Models and experiments

| # | Model | Role |
|---|-------|------|
| M1 | LLaMA-3-8B-Instruct | English-centric baseline |
| M2 | Qwen3.5-9B | Multilingual base |
| M3 | Qwen3.5-9B + LoRA on corpus 2B | Fine-tuned model |
| M4 (optional, zero-cost) | Menstrual-LLaMA-8B (huggingface.co/proadhikary/Menstrual-LLaMA-8B) | Domain-fine-tuned but English-centric, single-style baseline — tests whether domain adaptation *without* multilingual/style diversity survives our benchmark |

**LoRA starting config (from the MenstLLaMA paper, a published working recipe for 8–9B health QA):** QLoRA 4-bit NF4, ~5 epochs, lr 2×10⁻⁴, warmup 0.03, max grad norm 0.3, max seq 2048, Paged AdamW — adjust epochs down if 2B is larger than their corpus.

- All three evaluated **zero-shot** on the full benchmark with one fixed prompt: interpret the symptom, assess urgency, recommend an action, ask for clarification if the description is insufficient.
- **Thinking mode OFF** for all Qwen3.5 runs; fine-tune with the non-thinking chat template so train/eval match. Documented in the paper.
- **Qwen3.5 status:** Qwen3.5-9B is the fixed multilingual model for M2/M3; run zero-shot, English fine-tuned, and multilingual fine-tuned comparisons against this model family.
- **Stretch goal only:** RAG over the NHS/CDC/NIH corpus on top of M3. Cut without discussion if anything slips.

## 4. Evaluation

**Metrics:**
- Misunderstanding rate (wrong condition interpretation)
- Severity error rate, highlighting under-triage of `urgent` cases
- Unsafe-response rate
- Clarification rate on ambiguous items (should be high) vs. clear items (should be low)
- Cross-language consistency (same case → same verdict in EN/FR/AR)
- Cross-style consistency (same case → same verdict across 5 styles)
- Cultural sensitivity of the response (rubric item: avoids dismissive/judgmental framing, respects indirect/culturally-coded phrasing without penalizing it)
- Response helpfulness and clarity (rubric item: actionable, non-alarming, readable at a lay level)

**Statistical rigor (adopted from LUHME 2025 practice):**
- Greedy decoding (temperature 0) for all runs → deterministic primary results.
- LoRA fine-tune repeated with **3 seeds**; M3 results reported as mean ± std (mirrors Nicholls & Alperin).
- **McNemar's test** for pairwise model comparisons on misunderstanding rate; **bootstrap 95% CIs** on all reported rates (mirrors the significance-testing norm in Mohammadi et al.).
- **Confusion matrix over gold categories** (menstrual vs. pcos vs. fertility, plus other/none), reported per language.
- Confound control: every style/language variant preserves the seed's clinical facts exactly, so performance differences are attributable to expression, not content (our analog of Nicholls & Alperin's noun masking).

**Judging:** GPT-4-class LLM-as-judge with a strict rubric, **calibrated against ~100 human-labeled model responses**; agreement (e.g., Cohen's kappa) reported in the paper. If agreement is poor, fall back to human evaluation of a stratified ~300-item sample.

**Error analysis (heart of the paper):** failures categorized as severity misjudgment · condition confusion · missed fertility/pregnancy risk · unsafe advice · failure to clarify — broken down by language × style × model.

## 4b. Positioning vs. closest related work (for Related Work section)

- **Mohammadi et al. (LUHME 2025 family, cross-cultural moral judgments):** evaluates *what models believe* via log-prob probing against survey ground truth (variance correlation, ARI/AMI, chi-squared). We share the cross-cultural consistency framing and significance-testing rigor, but use a purpose-built clinical gold benchmark, free-text generation, and calibrated LLM-as-judge. Their publishable negative result validates our cut line #2.
- **Nicholls & Alperin (cross-genre NLI):** evaluates *what models can classify*; we adopt their QLoRA + Llama-Factory fine-tuning recipe, mean ± std reporting, per-class confusion matrices, and confound-control philosophy, but our target is *safe understanding* (clarification, triage, unsafe advice), not classification accuracy.
- **Recker et al. (Arch Gynecol Obstet 2025 editorial):** motivation anchor for the Introduction — documents language-dependent accuracy (De Vries et al.), phrasing sensitivity, guideline misalignment, and oversimplification in women's-health LLM use. Our styles × languages design systematically operationalizes these anecdotal risks.
- **Adhikary et al. (MenstLLaMA, JMIR 2025):** the closest prior system — LoRA fine-tune of LLaMA-3-8B on MENST for menstrual-health education, evaluated on answer quality (BLEU/BERTScore, expert preference, user satisfaction) in English only. Key distinction: their paraphrasing is *training-time augmentation without gold labels*; our styles are a *controlled evaluation dimension* with gold condition/risk/action labels. They measure answer quality; we measure understanding safety (triage, clarification, unsafe advice) across languages. Their expert-compiled held-out gold test set validates our 2A/2B separation.
- **Elmannai et al. (Diagnostics 2023, PCOS detection):** tabular ML — predicts PCOS from 41 structured clinical features (hormone panels, follicle counts) via stacking classifiers on a Kaggle dataset, with SMOTE-ENN and SHAP explainability. Off-axis from us (no text, no language, no multilingual dimension). One-line contrast for Related Work: prior PCOS ML operates on *structured clinical measurements*; we operate on *natural-language symptom communication*. Does not affect our methodology.
- **One-line distinction for the paper:** prior work evaluates what models believe or can classify; we evaluate whether models *understand patients safely* in multilingual women's-health communication.

## 5. Roles and per-person parallel plan

Roles (swap by mutual agreement before Day 2 if someone's skills fit better — but lock by Day 2):

| Person | Role | Owns |
|--------|------|------|
| **Mazen** | Modeling | Cluster env, inference pipeline, LoRA fine-tune, model runs |
| **Hassan** | Dataset | Seed cases, gold labels, style variants, FT-corpus filtering, leakage log |
| **Hana** | Language | FR/AR translation, reviewer coordination, validation rubric + documentation |
| **Mariam** | Evaluation & paper assembly | Judge prompts, calibration, metrics, error analysis, Overleaf skeleton |

Everyone drafts their own paper sections daily in Overleaf; Mariam assembles and keeps the draft coherent.

### Day-by-day (four parallel tracks)

**Day 1 — Mon Jul 6**
- Mazen: cluster env up; Qwen3.5-9B inference smoke test + 10-step LoRA smoke test. **Go/no-go on Qwen3.5 by end of day.** Repo skeleton on GitHub.
- Hassan: run §2C Steps 1–2 — normalize all three sources to the interim schema, apply the domain filter; begin selecting seed cases from NHS/CDC/NIH.
- Hana: draft the FR/AR validation rubric; confirm reviewer availability and schedule their review window (target: receive items Day 4, return Day 6).
- Mariam: Overleaf skeleton (ACL template), Intro bullet draft, begin Related Work (MENST/MenstLLaMA, MedDialog, women's-health LLM benchmarks, LLM-as-judge calibration, plus the three anchor papers in §4b — Mohammadi et al., Nicholls & Alperin, Recker et al./De Vries et al.).

**Day 2 — Tue Jul 7**
- Mazen: batch-inference pipeline (prompt template, thinking off, output parsing to JSON).
- Hassan: seed v1 nucleus (90 seeds, 30/category) already sourced — **regenerate the 5 style variants per seed with an LLM from canonical text** under the meaning-preservation rubric (replacing the templated ones); finish gold labels + source URLs; expand toward ~40/category if time allows.
- Hana: translation pipeline ready (GPT/NLLB); test on 20 items; refine prompts for dialect/register choices in Arabic.
- Mariam: judge rubric v1; Related Work continued; Methods section skeleton.

**Day 3 — Wed Jul 8**
- Hassan: complete + manually review all ~650 English variants (Hana and Mariam each review a third to parallelize); freeze English benchmark. Run §2C Steps 6–7 — leakage exclusion + Llama-Factory formatting → `ft_corpus_v1` (English), with `leakage_log.csv` committed.
- Hana: begin FR/AR machine translation of the benchmark.
- Mazen: prepare LoRA training config; translate/format FT corpus into training format (with Hana's pipeline).
- Mariam: judge prompt finalized; select the 100-item human-calibration sample design.

**Day 4 — Thu Jul 9**
- Hana: FR/AR translations complete → handed to reviewers with rubric.
- Mazen: **launch LoRA fine-tune (seed 1) via Llama-Factory**; run M1/M2 zero-shot on English items as smoke test. Seeds 2–3 launch as background jobs on Days 5–6 (cluster permitting) for the mean ± std report; seed 1 alone is the fallback if queue time is tight.
- Hassan: dataset card + statistics tables for the paper; datasheet documentation.
- Mariam: run judge on English smoke-test outputs; team labels the 100 calibration items (all four contribute 25 each).

**Day 5 — Fri Jul 10**
- Mazen: fine-tune done; sanity-check M3 outputs.
- Mariam: compute judge–human agreement; adjust rubric if needed (fallback decision made today).
- Hassan: Methods §Dataset written; Hana: Methods §Translation & validation written.

**Day 6 — Sat Jul 11**
- Hana: reviewer validations returned → apply fixes → **freeze benchmark v1.0** (any unvalidated remainder labeled silver).
- Mazen: run M1, M2, M3 × full benchmark (~1,950 items × 3 models).
- Mariam: metrics pipeline ready; Hassan assists with error-category annotation guidelines.

**Day 7 — Sun Jul 12**
- Mariam + Hassan: compute all metrics; error analysis (each annotates half of the flagged failures); produce tables + figures.
- Mazen: any reruns / RAG stretch goal **only if everything above is green**.
- Hana: Limitations + Ethics sections (translation caveats, no real patient data, not medical advice).

**Day 8 — Mon Jul 13 → Tue Jul 14**
- All: full draft complete by Mon evening → internal cross-review Tue morning → supervisor review Tue → revisions Tue night.

**Day 9 — Wed Jul 15**
- Mariam: ACL format + anonymization check (no names, no repo links that deanonymize).
- Submit on OpenReview **hours before the deadline, not minutes**. Everyone verifies the submitted PDF.

**Daily sync:** one 15-minute check-in at a fixed time; blockers raised immediately in the group chat, not saved for the sync.

## 6. Pre-agreed cut lines (no panic decisions)

1. Reviewer validation late → report validated subset as gold, remainder as silver; state it plainly.
2. Fine-tune fails or underperforms → **report the result anyway**; "fine-tuning did not reduce misunderstanding" is a valid finding.
3. Judge–human agreement poor → human evaluation of a stratified ~300-item sample.
4. Anything else slips → drop styles in this order: emotional → indirect (keep clinical, layperson, ambiguous) **before** dropping any language.
5. RAG is the first thing cut, always.

## 7. Risks

| Risk | Mitigation |
|------|-----------|
| Qwen3.5-9B environment issues (new hybrid architecture) | Smoke-test vLLM inference and local QLoRA before full runs; document any unresolved runtime blocker immediately |
| Reviewers slower than promised | Day-4 handoff + Day-6 return agreed up front; cut line #1 |
| GPT-generated variants drift from gold labels | Human review of every English item; variant must preserve the seed's clinical meaning |
| Benchmark/FT-corpus leakage | Explicit leakage log owned by Hassan; checked before fine-tune launch |
| LLM-judge unreliability | Calibration against 100 human labels, kappa reported; cut line #3 |
| Deadline confusion | July 15 confirmed by team; submit Day 9 morning, not evening |

## 8. Reconciliation with the weekly brief (LUHME_weekly_brief.md)

The weekly brief aligns with this spec on framing, translation-validation, experiment design, and metrics. Three corrections take precedence — **this spec is the source of truth** where they conflict:

1. **MENST `train24K.csv` size:** the brief says "~42K"; the verified count is **22,412 rows** (1,721 unique answers). Use 22,412 everywhere, including the paper.
2. **Leakage key is dual, not single:** the brief's `leakage_key = source_dataset + source_row_id` is necessary but insufficient for MENST (one answer → ~13 paraphrase siblings under different row IDs). Add `seed_answer_hash` (normalized-answer sha1) as a second block key. Both filters apply to every fine-tuning pull. This is the highest-priority correction.
3. **Style variants are NOT final:** seed v1's variants are template-generated and must be regenerated from canonical text under the meaning-preservation rubric (§2A) **before** translation to AR/FR — otherwise broken templates get translated and validated at cost. The brief's "audit-ready provenance" applies to sourcing/provenance, not to the variant text.

Also fold into the brief when it becomes the paper: name the actual models (M1 LLaMA-3-8B, M2 Qwen3.5-9B, M3 +LoRA, optional M4 Menstrual-LLaMA-8B), the thinking-mode-off decision, and the gold-label fields (`gold_risk_level`, `gold_action`, `requires_clarification`) that the safety metrics depend on.

## 9. Handoff to Claude Code — build order and definitions of done

Single source of truth: this spec. Repo already contains `deliverables/week1/` (literature summaries, dataset comparison table). Artifacts to produce, in order:

1. **`scripts/build_ft_corpus.py`** — implements §2C Steps 1–7. Inputs: the 5 source files. Outputs: `ft_corpus_v1.jsonl` (EN, Llama-Factory format), `leakage_log.csv`. DoD: 900/category (2,700) after dedup; zero rows whose `answer_hash` ∈ seed hashes; both leakage keys applied.
2. **`scripts/regenerate_style_variants.py`** — replaces seed v1 templated variants with LLM-generated ones from `canonical_text`, meaning-preservation rubric, one variant per style. DoD: every variant preserves all canonical clinical claims; human-review column present; no template collapse (distinct-string ratio checked).
3. **`benchmark_seeds.csv`** — 90-seed nucleus with full §2A schema incl. `gold_risk_level`, `gold_action`, `requires_clarification`, `seed_answer_hash`. DoD: no `NEEDS_GROUNDING` left unresolved; grounded to NHS/CDC/NICHD URLs.
4. **`scripts/translate_benchmark.py`** — EN → FR/AR first pass (NLLB/GPT), emits reviewer sheets. DoD: parallel item IDs preserved across languages.
5. **`scripts/run_inference.py`** — M1/M2/M3(/M4) × benchmark, thinking-mode off, temp 0, fixed prompt. DoD: deterministic outputs, JSON-parsed, one row per (item × model).
6. **`scripts/evaluate.py`** — judge + metrics (§4), McNemar, bootstrap CIs, confusion matrices, 3-seed mean±std for M3. DoD: judge–human kappa computed on 100-item calibration set.

Environment note: `pip install --break-system-packages`; verify Qwen3.5-9B on the serving/training machine before full runs. Keep everything git-versioned. Do not translate style variants until step 2 is human-reviewed and the benchmark is frozen.
