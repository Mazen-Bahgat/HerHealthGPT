# Experiment Lineage Audit — HerHealthEval

**Status:** read-only audit. No `.tex` edited, no result files overwritten, no
branches merged, no experiments rerun, nothing deleted.
**Date:** 2026-07-17. **Auditor:** automated forensic pass over git history + result contents.
**Question:** are the camera-ready paper's numbers and the newer `hassan-pc`
results one trustworthy, comparable evidence base — and if not, why do they differ?

> **Bottom line up front.** The difference is **a genuinely different benchmark
> per branch**, not a bug, a bad copy, or a misaligned comparison. The paper
> (`origin/main`) is built entirely on the **gss** benchmark and is *internally
> valid and complete*. The newer `hassan-pc` results are on the **menst**
> benchmark (`benchmark_multilingual_v1`) and are *also internally valid*, and they
> add the corrected model **M3ml-v2**. The two number sets are **not
> interchangeable** because the benchmarks have different items, gold
> distributions, and denominators. `0.994` in the paper is **under-triage**
> (verified), not a mis-copied consistency value.

---

## 1. Repository and branch summary

Repo: `HerHealthGPT` (`origin`: github.com/Mazen-Bahgat/HerHealthGPT).

| Ref | HEAD | Date | Role |
|---|---|---|---|
| `origin/main` | `db0c7c7` | 2026-07-17 16:28 | **PC1** work + **the paper** (matches `E:\camera_ready`) |
| `hassan-pc` = `origin/hassan-pc` | `54ede25` | 2026-07-17 17:30 | **PC2** work (M2 re-run on menst, M3ml-v2, result/) |
| `main` (local, stale) | `2d4c998` | 2026-07-16 13:52 | outdated pointer, **behind** `origin/main`; ignore |
| `origin/feat/qwen35-en-finetune` | `7f4f9dd` | 2026-07-13 | old feature branch, fully behind both; not relevant |

Machine attribution (from the base-model path stored in each result record):
- `origin/main` result files → `/home/sw2/…` → **PC1 (user `sw2`)**.
- `hassan-pc` result files → `/home/hassan/…` (or offline id `Qwen/Qwen3.5-9B`) → **PC2 (user `hassan`)**.

The paper (`E:\camera_ready`) equals `origin/main`'s Overleaf project: `overleaf_methods.tex`
identical; `abstract`/`intro`/`results_discussion` differ by only 2–10 lines
(supervisor's final local tweaks).

## 2. Common ancestor

`git merge-base origin/main hassan-pc` = **`2d85f07`** — *"Finished the gold seed
styled AR and FR translation"* (2026-07-16 15:33). This is the fork point between
the two working PCs. (Note: local `main` at `2d4c998` is an even older pointer and
is not the fork point; use `origin/main`.)

- `origin/main` ahead of ancestor: **23 commits**.
- `hassan-pc` ahead of ancestor: **30 commits**.
- Neither is an ancestor of the other → **genuine parallel divergence**. Do not merge.

## 3. Experiment timeline

**Pre-fork (≤ `2d85f07`, shared by both PCs):**
- English pilot on the **gss** benchmark (`gold_seeds_styled.jsonl`): zero-shot
  `M2_gss_en`; adapted `M3en_v2json_en` (JSON = "M3"), `M3en_os4_en` (4× oversample
  = "M3+O"); the joint EN+FR+AR adapter ("M3ml").
- The **menst** benchmark `benchmark_multilingual_v1.jsonl` created (3 categories ×
  6 registers, 100% see-doctor gold).
- **Last confirmed shared / "zero-shot" state:** `M2_gss_en` baseline; ancestor
  commit `2d85f07` (gss AR/FR translations finished).

**Post-fork — PC1 (`origin/main`, user `sw2`, 23 commits) → the paper:**
- Built gss FR/AR benchmarks (`gold_seeds_styled_fr.jsonl`, `…_ar.jsonl`).
- Ran `M2_gss_fr/ar` and `M3ml_fr/ar` **on gss**; produced
  `safety_M2_vs_M3ml_{fr,ar}.{json,md}`.
- Wrote the manuscript (results/discussion refactor, figures, bib) → `db0c7c7`.

**Post-fork — PC2 (`hassan-pc`, user `hassan`, 30 commits) → newer results:**
- Ran `M2ml_{en,fr,ar}` and `M3ml_{en,fr,ar}` **on menst** (`benchmark_multilingual_v1`).
- Fixed FR/AR risk labels by `row_id` (`8e7831b`); retrained **M3ml-v2**; ran
  `M3ml_v2_{en,fr,ar}` on menst.
- Added relaxed/loose interpretation metrics, menstrual error analysis, and the
  consolidated `result/` folder.

**Duplicated runs:** `M3ml_{en,fr,ar}` exists on *both* branches but as **different
eval files** (different machine + different benchmark), not redundant copies.
**Incomplete runs:** none detected (all result files are 540 rows).
**Unconfirmed branch:** none — every result file's machine is identifiable by model path.

## 4. Branch-by-branch differences (result artifacts)

| Artifact | `origin/main` (paper) | `hassan-pc` (new) |
|---|---|---|
| M2 baseline FR/AR | `M2_gss_fr/ar` (**gss**) | `M2ml_fr/ar` (**menst**) |
| M2 baseline EN | `M2_gss_en` (gss) | `M2_gss_en` (gss, inherited) **and** `M2ml_en` (menst) |
| M3ml FR/AR | `M3ml_fr/ar` (**gss**) | `M3ml_fr/ar` (**menst**) |
| M3ml-v2 (corrected) | **absent** | `M3ml_v2_{en,fr,ar}` (menst) |
| gss FR/AR benchmark | `gold_seeds_styled_{fr,ar}.jsonl` present | **absent** |
| menst benchmark | present (blob `2c03f3d`) | present (blob `5baec18`) — **content-identical** |
| safety summaries | `safety_M2_vs_M3ml_{fr,ar}` | McNemar/relaxed via `multilingual_report.py` |
| paper | full manuscript (db0c7c7) | `Overleaf_Template/` variant + `result/` |

## 5. Dataset identity checks — the crux

Two **different** benchmarks are in play. Both have n=540 but are otherwise distinct:

| Property | **gss** (`gold_seeds_styled*`) | **menst** (`benchmark_multilingual_v1`) |
|---|---|---|
| Seed namespace | `gss-*` (540) | `menst/pcos/fert-*` (180 each) |
| Gold risk | 174 see-doctor / 348 routine / 18 urgent | **540 see-doctor** (100%) |
| Risk majority baseline | 0.644 | 1.000 (→ under-triage is the only honest risk metric) |
| Clarification-yes | 90 | 24 |
| Under-triage denominator | 174 see-doctor items | 540 see-doctor items |
| Used by | **the paper** (all tables) | `hassan-pc` `result/` (all numbers) |

- gss and menst overlap by ~2/540 items → **genuinely different question sets.**
- The menst benchmark is **content-identical across branches** (gold 540 see-doctor,
  24/516 clarification on both); the blob hash differs only because `hassan-pc`
  commit `4e7d8b8` re-serialized it (line endings/ordering), not a content change.
- The **gss FR/AR** benchmark exists **only on `origin/main`** → `hassan-pc` cannot
  by itself reproduce the paper's FR/AR numbers.

**Verdict on the "mismatch":** it is a *genuinely different benchmark per branch*,
chosen deliberately on PC2 (a 3-category, uniformly high-risk set) vs the earlier
gss pilot set on PC1. It is **not** a label-loading bug, parse bug, partial run, or
accidental dataset swap.

## 6. Result-file provenance

| File | Branch | Machine | Benchmark | n | Gold risk (see-doctor n) | Under-triage |
|---|---|---|---|---|---|---|
| `M2_gss_en` | main+hassan-pc | sw2 | gss | 540 | 174 | 0.718 |
| `M2_gss_fr` | origin/main | sw2 | gss | 540 | 174 | 0.632 |
| `M2_gss_ar` | origin/main | sw2 | gss | 540 | 174 | 0.638 |
| `M3ml_fr` (paper) | origin/main | sw2 | gss | 540 | 174 | **0.994** |
| `M3ml_ar` (paper) | origin/main | sw2 | gss | 540 | 174 | **0.994** |
| `M2ml_fr` | hassan-pc | hassan | menst | 540 | 540 | 0.439 |
| `M2ml_ar` | hassan-pc | hassan | menst | 540 | 540 | 0.420 |
| `M3ml_fr` (new) | hassan-pc | hassan | menst | 540 | 540 | 0.998 |
| `M3ml_ar` (new) | hassan-pc | hassan | menst | 540 | 540 | 0.996 |
| `M3ml_v2_fr` | hassan-pc | hassan | menst | 540 | 540 | 0.448 |
| `M3ml_v2_ar` | hassan-pc | hassan | menst | 540 | 540 | 0.441 |

## 7. Paper-number provenance (traced)

| Paper location | Value | Source file (branch/benchmark) | Verified |
|---|---|---|---|
| `tab:main` M2 under-triage | 0.718 | `M2_gss_en` (main/gss) | ✓ |
| `tab:main` M2 clarif recall | 0.856 | `M2_gss_en` | ✓ (consistent) |
| `tab:main` M3 / M3+O | — | `M3en_v2json_en` / `M3en_os4_en` (gss) | present both branches |
| `tab:multilingual` M2 FR / AR u-triage | 0.632 / 0.638 | `M2_gss_fr/ar` (main/gss) | ✓ |
| `tab:multilingual` M3-ML FR / AR u-triage | **0.994 / 0.994** | `M3ml_fr/ar` (main/gss) | ✓ |
| Abstract "under-triage … to 0.994" | 0.994 | same | ✓ |

**The `0.994` question (explicit ask): it is UNDER-TRIAGE.** `safety_M2_vs_M3ml_fr.md`
on `origin/main` states `under_triage_rate (gold=see-doctor → routine): 0.632 →
0.994`, and the FR risk confusion for M3ml is `{routine: 173, see-doctor: 1}` →
173/174 = **0.9943**. It is **not** cross-language consistency and **not** a copy
error. (This overturns an earlier hypothesis raised before this audit that `0.994`
might be the consistency figure — that hypothesis was wrong.)

## 8. Valid / conditionally valid / invalid comparisons

**Valid (same benchmark, same denominator):**
- Paper `tab:main`: M2 / M3 / M3+O / M3-ML — all gss English. ✓
- Paper `tab:multilingual`: M2 vs M3-ML across EN/FR/AR — all gss, all 174-see-doctor
  denominator. ✓ **The paper's central comparison is internally valid.**
- `hassan-pc`: M2ml vs M3ml vs M3ml-v2 across EN/FR/AR — all menst, all
  540-see-doctor denominator. ✓ **The v2-fix comparison is internally valid.**

**Conditionally valid (qualitative, disclose benchmark):**
- The *finding* "multilingual adaptation collapses non-English under-triage toward
  routine, and label-corrected re-adaptation recovers it" holds **independently on
  both benchmarks** (gss: 0.632→0.994; menst: 0.44→0.998, then v2 0.448). The
  *narrative* is reproducible even though the *numbers* are not interchangeable.

**Invalid (cross-benchmark — must not be mixed):**
- Placing `hassan-pc` menst numbers (e.g. M2 0.44, M3ml-v2 0.448) into the paper's
  gss tables, or vice-versa. Different items, gold, and denominators.
- **Specifically forbidden:** claiming "v2 fixes the 0.994" by writing v2's menst
  under-triage (0.448) next to the paper's gss 0.994. To make that claim in the
  paper, v2 must be evaluated **on gss**.

## 9. Missing metadata

Result records store only `{seed_id, model_label, model (base-model path)}`.
**Not recorded** (must be inferred from code/commits, treat as uncertain):
adapter path, random seed, decoding params (greedy vs sampling, `max_time`),
prompt-template hash, benchmark file path/checksum, run timestamp, exact commit.
The base-model path identifies the *machine* (sw2 vs hassan) but not the seed or prompt.
Inference config is inferred from `scripts/run_local_inference.py` (greedy decode;
per-row time cap added in `f0fe5d2`); training seed 3407 from `scripts/train_qlora.py`.

## 10. Runs that must be repeated (to unify the evidence base)

To add the **M3ml-v2 correction** into the paper *validly* (paper is on gss):
1. **Evaluate M3ml-v2 on the gss benchmark**, all three languages →
   `M3ml_v2_gss_{en,fr,ar}`. Inputs all exist:
   - adapter: `models/qwen3.5-9b-herhealth-enfrar-lora-v2` (on `hassan-pc`);
   - gss benchmarks: `gold_seeds_styled{,_fr,_ar}.jsonl` (on `origin/main`);
   - `scripts/run_local_inference.py`.
2. (Optional, for the English pilot `tab:main`) evaluate M3ml-v2 on gss English too,
   to add a v2 column consistent with M2/M3/M3+O/M3-ML.

No other reruns are required — the paper's existing gss runs are complete and valid.

## 11. Recommended canonical result set

**The gss evidence base on `origin/main`** is the paper's canonical set: complete
(M2, M3, M3+O, M3-ML across EN/FR/AR) and internally valid. Keep it as the paper's
backbone. Treat the `hassan-pc` **menst** set (clean M2, M3ml, **M3ml-v2**, relaxed
metric, error analysis) as a **second, benchmark-distinct** evidence base whose
scientifically important contribution — the *label-corrected v2 recovery* — should
be **ported onto gss** (step 10) rather than swapped in.

## 12. Recommended reproducible base

- **Paper as-is:** `origin/main @ db0c7c7` (gss, complete, valid).
- **v2 correction / new metrics:** `hassan-pc @ 54ede25` (menst).
- **Unifying commit to branch from:** the ancestor `2d85f07`, or simply run step-10
  evals on `hassan-pc` (it has the v2 adapter) after cherry-picking / copying the
  gss FR/AR benchmark files from `origin/main`. Do **not** merge the branches to do
  this; copy the two benchmark files read-only.

## 13. Recommendation for updating the paper

**Do not** substitute menst numbers into the gss tables. The paper's numbers are
correct and valid *for gss*; the mismatch with `result/` is a benchmark difference,
not an error to "fix". The scientifically valuable new content (the v2 label-bug
correction) belongs in the paper, but only after it is measured on the paper's own
benchmark.

**→ Final recommendation: Option B — rerun selected experiments.**
Evaluate the existing **M3ml-v2 adapter on the gss benchmark (EN/FR/AR)** so the
correction is directly comparable to the paper's valid M2/M3-ML gss results. Then
the manuscript can tell the complete, single-benchmark story: *adaptation collapses
non-English under-triage (0.632 → 0.994), traced to language-dependent risk labels,
and label-corrected re-adaptation (v2) recovers baseline safety* — every number on
gss, every comparison valid. This is a small, targeted rerun (one adapter × one
benchmark × three languages), **not** a full restandardization.

- **Option A (use best runs from both PCs as-is): REJECTED.** The two PCs used
  *different benchmarks*, not merely different machines, so their numbers are not
  directly comparable.
- **Option C (standardize & rerun everything): not required now.** Only choose this
  if you decide to move the whole paper onto the menst benchmark (which would then
  need M3 and M3+O rerun on menst, and `tab:main` redone). More work, no added
  validity over Option B for the current narrative.

### Menst-only analyses (relaxed metric, menstrual error analysis)
These live on the menst benchmark and cannot be attached to gss tables. Include them
only if the paper adds an explicit menst-benchmark subsection, or defer to future
work. They do not affect the validity of the gss results.

---

*Prepared without modifying the manuscript, result files, or branches, per the
audit constraints. All values above are reproducible from the cited git blobs and
result files.*
