# CHANGELOG — camera_ready → camera_ready_updated_results

Integration of the **risk-corrected re-adaptation (M3-ML-RC)** result into the
supervisor-reviewed manuscript. The original `E:\camera_ready` project is left
**untouched** (it is the backup); all edits are in this folder only.

**Guiding principle:** no existing number was replaced. Every M2 / M3 / M3+O /
M3-ML value in the paper was re-verified against its source file and reproduced
exactly (see `RESULTS_AUDIT.md`). The update **adds** the corrected model as a new
column and extends the narrative; it does not overwrite prior results.

---

## Files modified

| File | Change |
|---|---|
| `overleaf_abstract.tex` | Added one sentence: the FR/AR under-triage collapse (0.994) is traced to a language-asymmetric labeling defect and **corrected** by re-adapting from English-derived risk labels (0.994→0.572/0.558), while clarification stays suppressed. (Also fixed a missing space, `0.994.Collectively`.) |
| `overleaf_intro_related.tex` | Extended the results-preview paragraph (§Introduction) to state the correction (M3-ML-RC) and that clarification loss persists. No other intro/related-work text changed. |
| `overleaf_methods.tex` | (1) §Annotation: the fix previously described as *future work* is now presented as executed — introduces **M3-ML-RC** and forward-references the result. (2) **Fixed a broken figure path** (pre-existing): `Overleaf_Template_20260716_1412/figures/data_processing_pipeline.png` → `figures/data_processing_pipeline.png` (the file's actual location; matches the other figure's path convention). This was required for the project to compile. |
| `overleaf_results_discussion.tex` | (1) Results intro + §Cross-Lingual intro: add M3-ML-RC to the model roster and multilingual comparison. (2) **`tab:multilingual`**: added an **RC (= M3-ML-RC) column per language** (parse, risk acc, under-triage, clarif recall, category, risk consistency), preserving design/`\resizebox`/failure-pattern bolding; extended the caption. (3) Added a new `\paragraph{Correcting the risk labels recovers safety.}` with the recovery numbers and the see-doctor-subset McNemar. (4) §Discussion RQ3: extended to state the regression is a correctable labeling defect and clarification is the residual failure. |

## Files NOT modified (unaffected by the new result)
`overleaf_titleauthor.tex`, `acl_latex.tex`, `acl_lualatex.tex`, `acl.sty`,
`acl_natbib.bst`, `custom.bib`, all of §Literature Review, `tab:main` (English
pilot), most of §Methods, §Limitations, §Ethical Considerations. Template,
packages, and formatting configuration are unchanged.

## Old results replaced
**None.** All existing numbers preserved and re-verified (parity confirmed).

## New results added
- **M3-ML-RC** column in `tab:multilingual` (EN/FR/AR), all on the **gss**
  evaluation benchmark (the paper's benchmark), same prompt + greedy decoding as
  the paper's runs (only the adapter differs).
- Under-triage recovery: FR/AR **0.994 → 0.572 / 0.558** (below the M2 baseline);
  EN 0.711 → 0.590.
- See-doctor-subset McNemar: FR p=4.0×10⁻²¹ (v2 escalates 74 vs 1); AR p=5.3×10⁻²³
  (75 vs 0).
- Cross-style risk consistency for RC: EN 0.367 / FR 0.467 / AR 0.422 (genuine,
  non-degenerate — unlike M3-ML's collapsed 0.978/0.967).

## Claims revised (strengthened, none weakened)
- **RQ3 answer** now: adaptation regressions are traceable to a labeling defect and
  **correctable** (M3-ML-RC); clarification collapse is the residual, independent
  failure. Previously RQ3 ended at "introduces severe safety regressions."
- **Abstract / Introduction** now report the correction alongside the regression.
- The paper's core thesis ("aggregate/consistency metrics mask safety") is
  **reinforced**: the recovery is invisible to aggregate risk accuracy and visible
  only in under-triage on see-doctor items.
- No claim of the proposed method "beating" baselines beyond what the data show:
  RC recovers/*reduces* under-triage (a safety metric) but does **not** restore
  clarification, and this is stated explicitly.

## Points requiring human review (see also inline `% TODO` and DOCTOR_REVIEW_SUMMARY.md)
1. **Model name `M3-ML-RC`** — please confirm the naming (risk-corrected) is
   acceptable, or supply a preferred label.
2. **Pre-existing inconsistency (not introduced here):** the `tab:positioning`
   caption in §Literature Review states *"the scored evaluation set is English"*,
   which conflicts with the French/Arabic results in `tab:multilingual`. Likely a
   stale caption from an English-only draft. Flagged with an inline
   `% TODO: Clarification required` in `overleaf_intro_related.tex`; not edited.
3. **Benchmark scope:** this manuscript is entirely on the **gss** evaluation
   benchmark. A second, larger analysis set (`benchmark_multilingual_v1`, a
   3-category uniformly high-risk set) and additional analyses (relaxed
   interpretation metric, menstrual category-boundary error analysis) exist on a
   *different* benchmark and are **out of scope** for this version — see
   `EXPERIMENT_LINEAGE_AUDIT.md` in the code repository. They are candidates for a
   future extension, not mixed into these tables.
4. **Silver labels** remain (unchanged limitation); M3-ML-RC corrects label
   *derivation*, not the silver→clinician gap.
