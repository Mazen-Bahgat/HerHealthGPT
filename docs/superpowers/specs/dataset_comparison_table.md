# Dataset Comparison Table — HerHealthGPT-LU (LUHME 2026)

All row counts verified directly from the uploaded files. Selection decision at the bottom.

## Candidate datasets

| Dataset | File(s) | Size (verified) | Style / register | Languages | Women's-health yield | License | Role in our project |
|---------|---------|-----------------|------------------|-----------|----------------------|---------|---------------------|
| **MENST** | `train24K.csv`, `training2K.csv`, `test.csv` | 22,412 rows (aug.); 1,896 raw; 1,721 unique answers | Education-style QA, paraphrase-augmented | English (India-context) | High for menstrual/PCOS; thin for infertility | Open (HF: proadhikary/MENST) | **Primary** — silver FT backbone + seed mining |
| **iCliniq (ChatDoctor)** | `train-…parquet` | 7,321 rows | Real patient–doctor dialogue, triple-answer | English | Moderate (few hundred relevant) | Research use (ChatDoctor) | **Supplementary** — authentic patient phrasing |
| **HealthCareMagic (ChatDoctor)** | `HealthCareMagic-100k-en.jsonl` | 112,165 lines | Real patient–doctor dialogue | English | ~21K lines (~19%) match women's-health terms | Research use (ChatDoctor) | **Supplementary** — largest source of layperson phrasing |
| **PCOS Kaggle (Elmannai et al.)** | (not used) | 541 instances × 41 features | Structured clinical measurements | N/A (tabular) | PCOS only, no text | Open (Kaggle) | **Excluded** — no natural language |
| MedDialog-EN | (not sourced) | ~0.26M dialogues | Real dialogue | English | Unfiltered, low precision | Varies | Not selected — HCM/iCliniq sufficient |
| French women's intimate-health MedDialog | (not sourced) | — | Real dialogue | French | Unverified availability | Unclear | Deferred — our FR comes from translation + review |

## Why the tri-source choice (MENST + iCliniq + HealthCareMagic)

The three play complementary roles rather than being interchangeable. MENST gives structured, topic-tagged coverage of menstrual and PCOS concerns and a large answer pool for silver fine-tuning, but its education-style phrasing is too clean to test real-patient understanding. iCliniq and HealthCareMagic supply the messy, indirect, layperson phrasing our language-understanding benchmark depends on, but need heavy filtering (only ~19% of HCM is women's-health-relevant). Using all three lets us source *authentic phrasing* for the benchmark seeds while keeping a *large clean corpus* for training — with strict separation between the two (see leakage rule).

## Coverage by our three categories (seed nucleus, 30 each)

| Category | Seeds | Dominant source(s) |
|----------|------:|--------------------|
| menstrual | 30 | HealthCareMagic, MENST |
| pcos | 30 | HealthCareMagic, MENST, iCliniq |
| fertility | 30 | MENST, HealthCareMagic, iCliniq |

## Selected primary dataset

**MENST** as the primary/backbone, supplemented by **iCliniq** and **HealthCareMagic** for patient phrasing. The tabular PCOS Kaggle set (Elmannai et al.) is excluded as non-linguistic; MedDialog variants are not needed given HCM/iCliniq coverage, and French comes from validated translation rather than a separate French corpus.

*Counts verified from uploaded files on 2026-07-09.*
