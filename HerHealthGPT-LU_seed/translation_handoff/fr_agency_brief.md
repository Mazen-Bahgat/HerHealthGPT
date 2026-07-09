# Translation brief -- HerHealthGPT-LU (LUHME 2026) English -> French

## What this is

`fr_agency_handoff.csv` contains **synthetic, patient-style text** for a multilingual
language-understanding research benchmark. It is NOT real patient data -- each row is
either a de-identified/paraphrased seed sourced from public health-QA datasets, or a
style variant generated to test how language models interpret different registers of
the same underlying health concern (menstrual health, PCOS/hormonal symptoms, fertility).

## Critical requirement: preserve the register (`style` column), not just the meaning

Each row is deliberately written in ONE of five registers, and the register itself is
part of what we are measuring:

| `style` value | What it means | Translation guidance |
|---|---|---|
| `canonical` | Original patient wording, lightly cleaned | Translate naturally, preserve original tone |
| `clinical` | Clinical/chart-note phrasing | Keep formal, clinical French register |
| `layperson` | Everyday non-medical phrasing | Keep informal, everyday French -- do NOT upgrade to medical terminology |
| `indirect_cultural` | Indirect/euphemistic phrasing, avoids naming the condition | Preserve the indirectness -- do NOT make it more explicit or clinical |
| `ambiguous` | Deliberately vague, missing detail | Preserve the vagueness -- do NOT add specificity that isn't in the English |
| `emotionally_concerned` | Same content, worried/anxious tone | Preserve the emotional register |

**Please do not normalize register toward formal medical French across all rows.** A
`layperson` row translated into clinical French, or an `ambiguous` row translated with
added specificity, breaks the research design even if the translation is otherwise
accurate and natural.

## What we need back

Fill in `fr_text` for every row. Leave `fr_agency_translator_id` / `fr_agency_qa_id` with
your internal translator/QA identifiers if you track per-item ownership -- otherwise leave
blank. The `fr_house_*` columns are for our own team's spot-check pass after delivery and
should be left blank.

## Timeline

Please confirm turnaround time on receipt so we can schedule our in-house review window.

## Contact

Hana (Language lead) -- coordinates all translation handoffs for this project.
