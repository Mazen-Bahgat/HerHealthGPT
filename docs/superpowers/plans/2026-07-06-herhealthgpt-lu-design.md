> ⚠️ **DEPRECATED — superseded by `docs/superpowers/specs/2026-07-06-herhealthgpt-lu-design.md`.**
> This file is an earlier draft (2-category scope: PCOS + infertility, ~130 aspirational seeds).
> The specs/ version is the finalized, measured, source-of-truth design (3 categories:
> menstrual + PCOS + fertility, 90 seeds verified from real files, dual leakage key, §2C join
> procedure, §9 build handoff). Do not edit or cite this file — kept only for history.

# HerHealthGPT-LU — Design Spec (superseded draft)

**Date:** 2026-07-06 · **Deadline:** LUHME 2026 submission, 15 July 2026 (OpenReview, ACL template, 8 pages max, double-blind)
**Team:** Mazen, Hana, Hassan, Mariam · **Supervisor:** Dr. Rahatara Ferdousi

## 1. Paper framing

**Working title:** *HerHealthGPT-LU: A Multilingual Benchmark for Evaluating LLM Understanding of PCOS and Infertility Symptom Communication*

**Research question:** How well do LLMs interpret clinical, layperson, indirect, ambiguous, and emotionally-expressed descriptions of PCOS/hormonal and infertility symptoms across English, French, and Arabic — and does multilingual fine-tuning reduce misunderstanding?

**Contributions:**
1. **Dataset** — a human-validated multilingual benchmark of symptom expressions for two conditions.
2. **Evaluation** — metrics targeting misunderstanding, safety, clarification behavior, and cross-language/cross-style consistency.
3. **Analysis** — systematic error taxonomy of how and why LLMs misunderstand women's-health language.
4. **Adaptation evidence** — whether LoRA fine-tuning on multilingual women's-health data reduces misunderstanding.

Framed as a language-understanding problem in a high-stakes domain (LUHME topics: socio-cultural aspects of LU; effects and risks of misunderstanding; LLMs for generating/evaluating linguistic data).

## 2. Datasets — two strictly separated sets

### 2A. Evaluation benchmark (gold, small, human-validated)

- **~130 seed cases**: ~70 PCOS/hormonal, ~60 infertility.
  - Gold labels (condition, risk level, action) grounded in: NHS PCOS symptoms page, NHS heavy-periods page, NHS infertility page, CDC PCOS page, NIH endometriosis/infertility resources.
  - Authentic patient phrasing mined from MENST (huggingface.co/datasets/proadhikary/MENST) and iCliniq / HealthCareMagic (ChatDoctor corpora on Hugging Face).
- **5 expression styles per seed**: clinical, layperson, indirect/culturally sensitive, ambiguous, emotionally concerned. GPT-assisted generation; every English item manually reviewed by the team. ≈ 650 English items.
- **Translation to French and Arabic**: GPT/NLLB-200 first pass → validated by a native speaker (fluency, naturalness) and a linguistic reviewer (meaning preservation, ambiguity, cultural appropriateness), with a documented rubric. Total ≈ 1,950 items.
- **Item schema:** `case_id, condition, style, language, text, gold_condition, gold_risk_level {routine | see-doctor | urgent}, gold_action, requires_clarification {yes|no}, source_url, validation_status`.

### 2B. Fine-tuning corpus (silver, larger, spot-checked)

- Keyword-filter MENST + HealthCareMagic + iCliniq for PCOS / hormonal / menstrual-irregularity / fertility content → **~3,000–5,000 QA pairs**.
- Machine-translate to FR/AR; spot-check ~5%. No full human validation (stated honestly in the paper).
- **Leakage control:** every dataset item that inspired a benchmark seed is logged and excluded from 2B. No benchmark text (any language) appears in 2B.

## 3. Models and experiments

| # | Model | Role |
|---|-------|------|
| M1 | LLaMA-3-8B-Instruct | English-centric baseline |
| M2 | Qwen3.5-9B | Multilingual base |
| M3 | Qwen3.5-9B + LoRA on corpus 2B | Fine-tuned model |

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

**Judging:** GPT-4-class LLM-as-judge with a strict rubric, **calibrated against ~100 human-labeled model responses**; agreement (e.g., Cohen's kappa) reported in the paper. If agreement is poor, fall back to human evaluation of a stratified ~300-item sample.

**Error analysis (heart of the paper):** failures categorized as severity misjudgment · condition confusion · missed fertility/pregnancy risk · unsafe advice · failure to clarify — broken down by language × style × model.

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
- Hassan: download MENST + ChatDoctor corpora; keyword-filter for PCOS/infertility; start selecting seed cases from NHS/CDC/NIH.
- Hana: draft the FR/AR validation rubric; confirm reviewer availability and schedule their review window (target: receive items Day 4, return Day 6).
- Mariam: Overleaf skeleton (ACL template), Intro bullet draft, begin Related Work (MENST/MenstLLaMA, MedDialog, women's-health LLM benchmarks, LLM-as-judge calibration).

**Day 2 — Tue Jul 7**
- Mazen: batch-inference pipeline (prompt template, thinking off, output parsing to JSON).
- Hassan: finish 130 seed cases with gold labels + source URLs; start GPT-assisted style-variant generation.
- Hana: translation pipeline ready (GPT/NLLB); test on 20 items; refine prompts for dialect/register choices in Arabic.
- Mariam: judge rubric v1; Related Work continued; Methods section skeleton.

**Day 3 — Wed Jul 8**
- Hassan: complete + manually review all ~650 English variants (Hana and Mariam each review a third to parallelize); freeze English benchmark. Finish FT corpus (2B) with leakage log.
- Hana: begin FR/AR machine translation of the benchmark.
- Mazen: prepare LoRA training config; translate/format FT corpus into training format (with Hana's pipeline).
- Mariam: judge prompt finalized; select the 100-item human-calibration sample design.

**Day 4 — Thu Jul 9**
- Hana: FR/AR translations complete → handed to reviewers with rubric.
- Mazen: **launch LoRA fine-tune**; run M1/M2 zero-shot on English items as smoke test.
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
