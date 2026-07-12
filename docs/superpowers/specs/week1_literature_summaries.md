# Week-1 Literature Summaries — HerHealthGPT-LU (LUHME 2026)

Six papers reviewed. Work was split across two pairs, with all four contributing to writing:

- **Hassan + Mariam — Dataset & Evaluation teams**
- **Mazen + Hana — Modeling & Language teams**
- **All four — Writing** (Introduction, Related Work, synthesis)

Each pair owns its block below and expands it into the Overleaf Related Work. All summaries are in our own words — no quoting.

---

## Hassan + Mariam — Dataset & Evaluation

**Datasets inspected (task 1 verdict).** We searched and inspected the resources named by the supervisor. **MENST** (huggingface.co/datasets/proadhikary/MENST; `train24K.csv` 22,412 rows → 1,721 unique answers, `training2K.csv` 1,896 raw) is the most practical primary source: it is women's-health-specific, topic-tagged (covers menstrual, abnormal-menstruation/PCOS, pregnancy), and directly usable for a silver fine-tuning pool. **HealthCareMagic** (112,165 lines, ~19% women's-health) and **iCliniq** (7,321 rows) are real patient–doctor dialogues we keep as supplementary sources for authentic layperson phrasing. **MedDialog-EN** (~0.26M dialogues) was considered but not selected — its size offers no advantage over HCM once filtered, and precision is lower. The **French women's intimate-health MedDialog corpus** could not be reliably sourced for the deadline, so our French data comes from validated translation rather than a standalone French corpus. **Practicality verdict: MENST primary + iCliniq/HCM supplementary.**

**MenstLLaMA (Adhikary et al., 2025, JMIR) — dataset lineage.** MENST's 23,820 pairs derive from ~1,985 raw pairs via 4× GPT paraphrasing, so answers duplicate ~13×. This drives our answer-hash dedup and family-level leakage rule. Their held-out expert-compiled gold test set (kept separate from training) validates our benchmark/fine-tuning separation.

**Evaluation literature.** Mohammadi et al. (2025) — a LUHME-family paper — models the statistical rigor we adopt: significance testing (chi-squared), confusion matrices, and per-topic breakdowns, with a publishable *negative* finding (models compress cross-cultural variation). Nicholls & Alperin (cross-genre NLI) contribute reporting norms we mirror: mean ± std over three runs, per-class confusion matrices, and a confound control (noun masking) analogous to our meaning-preservation constraint. Elmannai et al. (2023, Diagnostics) is a scope contrast for Related Work — tabular PCOS prediction from 41 clinical features, i.e. structured measurements, not natural-language communication.

---

## Mazen + Hana — Modeling & Language

**Model selection (task: base + multilingual).** Following the supervisor's "one base model and one multilingual model, simple and reproducible" instruction, we select **LLaMA-3-8B-Instruct** as the English-centric base and **Qwen3.5-9B as our multilingual model**. Qwen3.5-9B is Apache-2.0, covers 201 languages/dialects (strong Arabic and French), and its base variant supports efficient LoRA fine-tuning with the official chat template — ideal for our mandatory fine-tune (M3 = Qwen3.5-9B + LoRA). Optional zero-cost fourth baseline: Menstrual-LLaMA-8B.

**Fine-tuning recipe (MenstLLaMA + Nicholls & Alperin).** MenstLLaMA gives us a published, working QLoRA recipe for an 8–9B health-QA model on a single A100 (4-bit NF4, ~5 epochs, lr 2×10⁻⁴, max seq 2048, Paged AdamW). Nicholls & Alperin confirm QLoRA via Llama-Factory as a reproducible path for small open LLMs, with cross-genre transfer showing fine-tuned features generalize.

**Multilingual / language understanding (task: translation & validation).** Mohammadi et al. show LLMs compress cross-cultural variation — motivating our cross-language consistency metric. Recker et al. (2025, Arch Gynecol Obstet), via De Vries et al., document that LLM accuracy on women's-health topics *varies by language* (non-English less reliable), that *slight phrasing changes* alter answers, and that advice can *misalign with national guidelines*. This grounds our AR/FR translation-validation protocol: parallel items, dual human review (native speaker + linguistic reviewer), adjudication on disagreement — because translation quality and annotation consistency are the main error drivers in multilingual health QA.

---

## All four — Writing (synthesis for Related Work close)

The six papers cluster into three strands. **Systems** (MenstLLaMA) show domain fine-tuning improves English women's-health answer generation. **Evaluation-of-understanding** (Mohammadi et al.; Nicholls & Alperin) show how to rigorously test what models understand or classify, with significance testing, confusion matrices, and confound controls. **Clinical motivation** (Recker/De Vries; Elmannai as structured-data contrast) establish that multilingual, phrasing-sensitive misunderstanding in women's health carries real safety stakes.

**Gap statement:** prior work evaluates what models *believe* (log-prob probing vs. surveys) or can *classify* (NLI accuracy, tabular PCOS prediction), or improves *answer quality* in one language. None evaluates whether models *understand patients safely* when the same clinical content is expressed in different registers and languages. HerHealthGPT-LU fills this with a human-validated EN/AR/FR benchmark across three categories (menstrual, PCOS, fertility), safety-oriented metrics (triage, clarification, unsafe-response), and a misunderstanding taxonomy — evaluating LLaMA-3-8B, Qwen3.5-9B, and Qwen3.5-9B+LoRA under matched items.

---

*Sources: Adhikary et al. 2025 (JMIR, MENST/MenstLLaMA); Mohammadi et al. 2025; Nicholls & Alperin (cross-genre NLI); Recker et al. 2025 (Arch Gynecol Obstet); De Vries et al. 2025 (via Recker); Elmannai et al. 2023 (Diagnostics).*
