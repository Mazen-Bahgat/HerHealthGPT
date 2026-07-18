# Supervisor Review Summary

A concise brief on what changed in this version of the paper and why.

## What is new
The manuscript now includes a **risk-corrected re-adaptation of the multilingual
model, M3-ML-RC**, evaluated on the **same (gss) benchmark** as all existing
results. The paper's Methods already diagnosed the FR/AR under-triage collapse as a
labeling bug and stated the fix as future work ("derive risk once from the English
source and attach by shared `row_id`"). **We have now executed that fix and
evaluated it**, so the paper can report the outcome instead of only prescribing it.

Concretely added:
- A third model column (**M3-ML-RC**) in the multilingual results table.
- A short "Correcting the risk labels recovers safety" paragraph with statistics.
- One sentence each in the Abstract, Introduction, and RQ3 discussion.
- Nothing was deleted or renumbered; **no existing result was changed** (every M2 /
  M3 / M3+O / M3-ML number was re-verified and reproduced exactly).

## How it affects the scientific message
The message is **strengthened and completed**, not overturned:

- **Before:** multilingual adaptation causes a severe FR/AR under-triage collapse
  (→0.994) and a clarification collapse, hidden by consistency metrics.
- **Now:** *that collapse is a fixable supervision defect.* Deriving risk labels
  from the English source (M3-ML-RC) **reverses the under-triage collapse
  (0.994 → 0.572 / 0.558 in FR/AR — below even the zero-shot baseline)**, while the
  **clarification collapse persists** across every adapted model. So the paper now
  cleanly separates two failures: a *correctable* label-quality regression and an
  *unresolved* clarification-seeking regression.
- The core thesis — *aggregate/consistency metrics mask safety* — is **reinforced**:
  the recovery is statistically strong on the safety-critical see-doctor cases but
  **invisible to aggregate risk accuracy**, exactly as the paper argues.

## Strongest findings (new)
1. **Under-triage recovery is decisive and one-directional.** On see-doctor cases,
   M3-ML-RC correctly escalates **74** French and **75** Arabic cases that M3-ML had
   routed to `routine`, while reversing only **1** and **0** the other way (McNemar
   **p = 4.0×10⁻²¹** FR, **p = 5.3×10⁻²³** AR).
2. **Consistency "recovers" downward, which is the point.** M3-ML-RC's cross-style
   risk consistency drops to a genuine 0.47/0.42 (vs M3-ML's collapsed 0.98/0.97),
   confirming the earlier high consistency was an artifact of routine-collapse.
3. **Root cause is supervision, not language comprehension** — a single label-
   derivation change fixes the safety collapse without touching the base model.

## Points to discuss / decide
1. **Model name** "M3-ML-RC" (risk-corrected) — OK, or do you prefer another label?
2. **Pre-existing caption conflict** (flagged inline, not edited): the
   positioning-table caption in the Literature Review says *"the scored evaluation
   set is English,"* which no longer matches the French/Arabic results. Please
   reconcile (update caption vs. clarify scope).
3. **Benchmark scope.** Everything here is on the **gss** benchmark. Separate work
   on a second benchmark (a 3-category, uniformly high-risk set) plus a
   relaxed/clinically-acceptable interpretation metric and a menstrual
   category-boundary error analysis exists but is on a *different* benchmark and is
   **not** mixed into these tables. It is a candidate extension for a follow-up.
4. **Clarification remains the open problem** — M3-ML-RC does not restore it. Worth a
   sentence on future direction (guardrailed / uncertainty-aware dialogue), which
   the Limitations already gestures at.

## Verification
- Re-scoring the paper's own M2/M3-ML files reproduces the published numbers exactly
  (e.g., M2 EN under-triage 0.718; M3-ML FR/AR 0.994; FR consistency 0.978) — so the
  new column is directly comparable.
- The updated project compiles cleanly (12 pages, 0 unresolved references, 0
  unresolved citations, no errors). See `COMPILATION_REPORT.md`.
- Full experiment-lineage and benchmark-identity audit: `EXPERIMENT_LINEAGE_AUDIT.md`
  in the code repository.
