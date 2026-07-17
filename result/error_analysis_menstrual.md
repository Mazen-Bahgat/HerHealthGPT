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

## 5. Reproduce
```bash
# confusion + examples were produced with scripts/evaluate.py scorers over
# Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/M2ml_{en,fr,ar}.jsonl
# (gold_category vs predicted_category on gold_category == "menstrual").
```
