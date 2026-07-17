# Error Analysis — the "menstrual" interpretation gap

**Question raised by the baseline report:** the base model interprets menstrual
concerns at only ~0.39 while fertility is ~0.90 (§4.3). Is this a model blind spot,
or a labeling artifact? **Answer: predominantly a category-boundary artifact — the
"errors" are largely clinically-defensible reads of inherently overlapping cases.**

All figures from `M2ml_{en,fr,ar}.jsonl` scored by `scripts/evaluate.py`.

---

## 1. What menstrual-gold items get predicted as

Confusion for the 180 menstrual-gold items per language (base model M2, parse_ok):

| Predicted | EN | FR | AR |
|---|---|---|---|
| menstrual (correct) | 0.39 | 0.39 | 0.40 |
| **fertility** | 0.38 | 0.36 | 0.36 |
| PCOS | 0.14 | 0.20 | 0.20 |
| other | 0.08 | 0.04 | 0.04 |

The misreads are not random: **87.2%** of the 109 wrong EN menstrual items are
mapped to an **adjacent category (fertility or PCOS)**; only **12.8%** go to
"other". The model isn't failing to understand — it is disambiguating an
overlapping case toward a different, usually reasonable, category.

## 2. Why: the menstrual seeds are conception- and PCOS-themed

**14 of 30 menstrual seeds (47%) explicitly mention conception/pregnancy.** These
are labeled `menstrual` (presenting symptom = irregular periods) but their *chief
complaint / patient intent* is fertility. Actual inputs the model "got wrong":

| Seed | Gold | Model | Input (verbatim) |
|---|---|---|---|
| menst-002 | menstrual | fertility | "Do irregular periods influence the ability to get pregnant?" |
| menst-008 | menstrual | fertility | "Due to my irregular periods I am worried about getting pregnant … Is it possible for me to conceive even with irregular periods?" |
| menst-018 | menstrual | fertility | "married for 4yrs, but i havnt conceived, my menstrual cycle is sometimes painful … desperate need a child" |
| menst-019 | menstrual | fertility | "missed my periods by 3 days and the test on the strip came weakly positive … could I be pregnant?" |
| menst-021 | menstrual | fertility | "35 year old married woman … trying for child since 1 yr … i missed periods …" |
| menst-023 | menstrual | fertility | "want to be pregnant because of my irregular periods my dr suggest me to take siphene …" |
| **menst-030** | menstrual | **PCOS** | "irregular periods. After an ultrasound … I have multiple cysts over the periphery of ovaries. I want to conceive …" |

- The **fertility** confusions are conception-motivated period complaints — the
  model keys on the stated goal ("want to be pregnant", "trying to conceive")
  rather than the presenting symptom. Clinically this is a defensible, arguably
  more useful, read.
- The **PCOS** confusions (e.g. menst-030) are textbook PCOS presentations —
  *irregular periods + polycystic ovaries on ultrasound* — where the label
  `menstrual` is debatable and `PCOS` is arguably **more correct** than the gold.

12 menstrual seeds are wrong on **all 6 phrasings** in every language — i.e. it is
a stable property of the *case/label*, not of surface form or language.

## 3. Cross-model: fine-tuning shifts the confusion, doesn't fix it

| Model | Lang | menstrual interp | dominant wrong-prediction |
|---|---|---|---|
| M2 (base) | EN/FR/AR | 0.39 / 0.39 / 0.40 | **fertility** (69/65/65) |
| M3ml-v1 | FR/AR | 0.37 / 0.38 | **PCOS** (83/80) |
| M3ml-v2 | EN/FR/AR | 0.37 / 0.33 / 0.38 | **PCOS** (84/82/74) |

Fine-tuning **moves the menstrual confusion from fertility (base) toward PCOS**,
because the adaptation corpus reshapes the model's category priors — but the
underlying overlap is not resolved by any model. This is itself a finding: the
low menstrual score is a property of the **taxonomy/labels interacting with real
patient language**, robust across models and languages.

## 4. Implications for the benchmark and paper

1. **Raw category accuracy understates true interpretation quality.** ~87% of
   menstrual "errors" are clinically-adjacent, and a non-trivial share (the
   conception- and PCOS-themed seeds) are reasonable or arguably correct.
2. **Real patient messages do not respect clean category boundaries.** Half the
   menstrual seeds are conception-motivated; irregular-periods-with-ovarian-cysts
   is simultaneously "menstrual" and "PCOS". This is a genuine
   language-understanding phenomenon, not just noise.
3. **Recommended actions for the paper (pick one or combine):**
   - Report a **relaxed / clinically-acceptable interpretation** metric that
     credits adjacent-category reads (menstrual↔fertility when conception is the
     stated intent; menstrual↔PCOS when polycystic ovaries are described),
     alongside strict accuracy.
   - **Refine the gold labels** for the ~14 conception-motivated menstrual seeds
     (multi-label, or relabel by chief complaint), then re-score.
   - Keep strict accuracy but **explicitly frame the menstrual gap as
     category-boundary ambiguity** with this confusion analysis as evidence —
     turning a weak-looking number into a substantive finding.
4. **Do not** claim the base model "cannot understand menstrual concerns" — the
   evidence contradicts it. The correct claim is that menstrual presentations in
   real patient language frequently carry fertility or PCOS intent that the model
   (reasonably) surfaces.

## 5. Relaxed (clinically-acceptable) interpretation metric

We implemented the recommended relaxed metric (`scripts/relaxed_interp.py`, unit
tests in `tests/test_relaxed_interp.py`). A prediction is credited if it lands in a
per-item **acceptable set** = {gold} ∪ {clinically-adjacent categories *justified by
the case content*}. Adjacency is **content-gated**, not blanket: an adjacent
category counts only when the case text contains its clinical markers (conception/
pregnancy → fertility; cysts/polycystic/ovaries → PCOS; period/cycle/bleeding →
menstrual). Markers are detected on the **English source per seed** (case content is
language-independent) and applied uniformly across EN/FR/AR.

| Model | Lang | strict interp | **relaxed interp** | Δ | menstrual strict→relaxed |
|---|---|---|---|---|---|
| M2 (base) | EN | 0.617 | **0.878** | +0.261 | 0.394 → 0.811 |
| M2 (base) | FR | 0.614 | **0.881** | +0.267 | 0.394 → 0.800 |
| M2 (base) | AR | 0.617 | **0.883** | +0.267 | 0.400 → 0.806 |
| M3ml-v1 | FR | 0.605 | 0.803 | +0.199 | 0.374 → 0.592 |
| M3ml-v1 | AR | 0.618 | 0.805 | +0.187 | 0.378 → 0.600 |
| M3ml-v2 | EN | 0.644 | 0.806 | +0.161 | 0.367 → 0.583 |
| M3ml-v2 | FR | 0.617 | 0.810 | +0.193 | 0.333 → 0.589 |
| M3ml-v2 | AR | 0.646 | 0.829 | +0.182 | 0.380 → 0.631 |

**Two findings:**

1. **True interpretation quality is far higher than strict accuracy implies.** The
   base model's clinically-acceptable interpretation is **~0.88** (not 0.62), and
   menstrual rises from ~0.39 to ~0.81 — confirming most "menstrual errors" are
   justified adjacent reads.

2. **The strict-accuracy ranking flips under clinical gating.** By strict accuracy,
   M3ml-v2 leads (EN 0.644 > base 0.617). By the *content-gated relaxed* metric,
   the **base model leads** (EN 0.878 > v2 0.806). The reason is mechanistic: the
   base model's menstrual errors go to **fertility**, which the conception-themed
   seeds *justify* (credited); fine-tuning shifted those errors to **PCOS**, which
   those same seeds usually do *not* justify (not credited). So the fine-tune's
   strict-accuracy gain partly reflects a category-prior shift toward a
   *less-justified* adjacent label, not better clinical understanding. This is a
   caution against reading the strict-accuracy improvement as a genuine
   interpretation gain.

Reproduce:
```bash
python scripts/relaxed_interp.py \
  --dir Used_Datasets/Consolidated_Datasets/200_Seed_Dataset \
  --model M2ml --model M3ml --model M3ml_v2 --langs en,fr,ar
```

## 6. Reproduce (confusion + examples)
```bash
# confusion + examples were produced with scripts/evaluate.py scorers over
# Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/M2ml_{en,fr,ar}.jsonl
# (gold_category vs predicted_category on gold_category == "menstrual").
```
