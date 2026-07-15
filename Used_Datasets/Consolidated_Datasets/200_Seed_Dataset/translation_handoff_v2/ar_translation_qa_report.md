# Arabic (MSA) Translation QA Report — `ar.csv`

Reviewer stance: professional Arabic↔English medical translator, MSA register.
Scope: `translation_handoff_v2/ar.csv`, 3,580 rows (2,862 train / 718 val,
600 unique underlying answers × up to 6 style-variant questions each).

## Methodology

1. **Structural integrity (all 3,580 rows):** resolved every `row_id` back to
   its source position in `Train/train_canonical_styled.csv` /
   `validate/validation_canonical_styled.csv`, verified `Question` text and
   style/topic alignment. **0 mismatches.**
2. **Automated linguistic screens (all 3,580 rows):** Latin-script leakage,
   MSA question-mark punctuation (`؟` vs `?`), colloquial-Egyptian marker
   scan, immediate word-duplication ("stutter") detection, and two
   term-specific checks triggered by manual sampling (see Findings A/B).
3. **Manual linguistic review (stratified sample, 24 rows across all 6
   styles × mixed topics)** — read against source English for accuracy,
   fluency, register match, and medical-terminology correctness.

## Overall verdict

**Fluent, well-registered MSA; two systematic, high-confidence, easily
fixable mistranslations found; native-speaker human sign-off still
recommended before treating this as benchmark-quality (consistent with the
FR delivery's own disclosed caveat).**

## Strengths (confirmed, not assumed)

- Natural, idiomatic MSA prose throughout the sample — not literal/
  word-for-word machine output.
- Style/register is genuinely carried into the Arabic, not just the English
  wrapper text: `clinical` rows use a real chart-note framing verb
  (`تطرح المريضة التساؤل الآتي`), `indirect_cultural` uses a natural
  embarrassment framing (`أشعر ببعض الحرج من طرح هذا الأمر`),
  `emotionally_concerned` uses genuine anxious register
  (`أشعر بقلق شديد حيال هذا الأمر`) — these read as different registers in
  Arabic, not the same sentence with a prefix swapped.
- Correct gender agreement on second-person verbs/imperatives when
  addressing a female patient (`استمري`, `تجنبي`, `اعتني بنفسك`) — this
  requires real grammatical competence, not a template.
- Brand names, drug names, and medical acronyms (Diane-35, Ovitrop-75, BMI,
  PMS, HSG, DXA) are correctly *kept* in Latin script rather than invented
  Arabic transliterations — standard professional practice, not an error.
- Garbled/corrupted source artifacts (`HeatlhcareMagic`, `ChatDoctorDoctor`,
  a mangled URL fragment, an OCR'd platform name) are faithfully preserved
  rather than silently "fixed" — correct practice, matches the FR
  delivery's stated policy of disclosing rather than correcting source
  corruption.
- 0 answer-truncation outliers (Arabic answer implausibly short vs. English
  source, threshold ratio < 0.3): none found.

## Findings

### A. HIGH — "patent" (tubal patency) mistranslated as intellectual-property patent

**30 of 36 rows (83%)** containing the English word "patent" (in the
gynecological sense — *fallopian tubes are patent* = open/unobstructed) are
translated using `براءة اختراق` / `براءة الاختراق`, which means a **patent
as in intellectual property**, not medical patency. Correct MSA term is
`سالكية` (patency) / `سالك` (patent, adj., as in "tubes are open"). Zero of
the 36 rows use the correct term.

Example (`val-0492`, canonical): source *"this is good that your tubes are
patent"* → Arabic reads *"وهذا جيد أن أنابيبك براءة الاختراق"* — literally
"this is good that your tubes are an invention patent," which is
nonsensical in context and actively misleading rather than merely awkward.

This is a **false-friend/polysemy error** (the MT/translation step picked
the wrong English sense of "patent") and is systematic, not a one-off —
worth a targeted find-and-review pass rather than a full re-translation.

Affected row_ids (unique underlying answers, 6 style-siblings each):
`val-0492`, `train-2214`, `train-2346`, `train-2784`, `train-2850` group,
plus others sharing the same source answers.

### B. HIGH — "Copper T" (IUD) mistranslated as testosterone

**6 of 6 rows (100%)** containing "Copper T" (a common name for the copper
intrauterine device) are translated as `التستوستيرون النحاسي` ("copper
testosterone") instead of the correct `اللولب النحاسي` (copper IUD). This
confuses a contraceptive device with a hormone — a clinically significant
and confusing error, though narrow in scope (one source seed, all 6 style
variants).

Example (`train-1428` group): *"i had my copper t inserted"* →
*"تم تركيب التستوستيرون النحاسي"*.

### C. MEDIUM — Word-duplication ("stutter") artifacts

**113 of 3,580 rows (3.2%)** contain an immediately-repeated word in the
Arabic (e.g., `الحمل الحمل`, `أكثر أكثر`, `المتحركة المتحركة`) — a pattern
consistent with a generation/decoding repetition glitch rather than
intentional Arabic emphasis (MSA medical prose doesn't repeat words this
way). One flagged case (`val-0718`, `AAAAAA AAAAAA`) is a false positive —
it mirrors a redacted-name artifact already present in the English source
and should stay as-is. The other 112 are candidates for a cleanup pass;
listed in full in the row-list this report was generated from (see
"Reproducing this review" below).

### D. LOW — Isolated anatomical-term inconsistency

One source answer (the `train-2592`–`train-2597` style-sibling group, PID/
pelvic-inflammatory-disease topic) uses `التهاب الرحم` (inflammation of the
*uterus*) in one sentence and the correct `التهاب الحوض` (inflammation of
the *pelvis*, i.e. PID) in another, plus a duplicated `التهاب التهاب`
within the same passage. Checked against all 36 rows mentioning PID/pelvic
inflammatory disease in the source — this is isolated to that single seed,
not systematic (unlike A/B above).

### E. LOW — ASCII "?" retained instead of Arabic "؟" (25 rows)

Where the English source itself has multiple question marks from messy
real-world phrasing (e.g., *"???"*, *"????"*), the Arabic keeps the ASCII
marks verbatim instead of normalizing to a single `؟`. Cosmetic, but a
professional MSA deliverable would normalize this. Affects 3 source seeds
× ~5-6 style rows each (`train-1020`–`1025`, `train-1254`–`1259`,
`train-1446`–`1451`).

### F. LOW / judgment call — Literal translation of an obvious source OCR typo

`train-1258` group: English source has *"why i have means today"* where
"means" is an obvious OCR/scraping typo for "menses" (the entire message is
about a missed period). The Arabic translates it literally as `وسائل`
("means/tools"), producing a confusing sentence, rather than inferring the
intended "menses." This is defensible under a "don't silently correct the
source" policy (same principle applied correctly elsewhere in this
dataset — see Strengths), but a native reviewer should decide whether typo
inference is in-scope here, since unlike the platform-name cases, "means"
vs. "menses" has an unambiguous correct reading from context.

## Non-findings (ruled out during review, listed so they aren't re-flagged)

- **"Dialect marker" false alarms:** an initial automated scan for
  colloquial-Egyptian words (`فين`, `ليه`, etc.) hit on **every** occurrence
  — but every hit was a substring inside a legitimate MSA word
  (`الإيبوبروفين` "ibuprofen", `الكامفين` "clomiphene", `تنزفين` "you
  [fem.] bleed", `ليهدأ` "so that it calms," `عليهم` "upon them"). Zero
  confirmed dialect usage found. The translation is consistently MSA
  register, not Egyptian colloquial.
- **Latin-script "leakage":** 1,116 rows contain 2+ Latin letters; 335 of
  those are legitimate medical acronyms/brand names (BMI, PMS, HSG, Diane,
  DXA, etc. — correct to keep in Latin script). Of the rest, manual
  inspection of every row with a 16+ character Latin run (75 rows, ~13
  unique underlying answers) found each one to be a faithfully-preserved
  garbled source artifact (broken platform names, a truncated URL, a
  garbled signature) — not incomplete/failed translation.

## Recommendation

1. Fix findings A and B first — both are systematic, unambiguous, and
   scriptable as a targeted find-and-replace against the specific source
   phrases (`patent`/`patency` → `سالك`/`سالكية`; `copper t` → `اللولب
   النحاسي`), each followed by a native-speaker spot-check rather than a
   blind replace, since context varies.
2. Spot-check the 112 real word-duplication rows (finding C) and strip the
   repeat.
3. Findings D/E/F are low-severity/cosmetic or judgment calls — worth a
   pass but not blocking for FT use.
4. This report doesn't replace native-Arabic human review before any
   "benchmark-quality" claim, per the same standard already applied to the
   French delivery.

## Reproducing this review

Row-id lists and full-text dumps for each finding were generated via
ad-hoc analysis scripts against `ar.csv` joined with the styled train/val
source files by `row_id`; not checked into the repo as they were scratch
analysis, not a reusable tool. Re-run by joining `ar.csv`'s `row_id` back to
`Train/train_canonical_styled.csv` / `validate/validation_canonical_styled.csv`
(index = the numeric suffix of `row_id`, style = `index % 6` in
canonical/clinical/layperson/indirect_cultural/ambiguous/emotionally_concerned
order) and re-running the checks described above.
