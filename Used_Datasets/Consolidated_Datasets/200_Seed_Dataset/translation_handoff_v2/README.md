# Translation handoff v2 (FR / AR)

## Handoff contract

- Fill only `Question_translated` and `Answer_translated` for every row.
- Do not edit or reorder any other column; `row_id` must remain unchanged.
- Preserve meaning and register (for example, layperson wording stays layperson
  and ambiguous wording stays ambiguous). Translation is not fact-checking.
- Keep the file as UTF-8 CSV. `Topic` and `Keywords` are English metadata and
  are not translated.
- Return one completed file per language (`fr.csv`, `ar.csv`).

## French v2 delivery (2026-07-15)

`fr.csv` is complete: 3,580 silver fine-tuning rows (2,862 train and 718
validation). It is not the 540-row evaluation benchmark, and no French
benchmark-quality claim is made.

The French text was generated with OpenAI's Responses API using
`gpt-5.6-sol`, low reasoning effort, Structured Outputs, and `store: false`.
Style/register was recovered from each stable `row_id`; 3,573 unique
register-aware questions were translated. The 600 unique English answers were
translated once and reused wherever the source answer was identical.

Quality control included corpus-wide schema/identity/UTF-8/language checks,
triage of 404 unique source-vs-translation review jobs, and AI-assisted
stratified review of 180 rows (5.03%) spanning every style and topic. Seventy-one
fail-closed correction rules changed only translation cells. The final
validator result is 0 blocking errors across all 3,580 rows. Natural
cross-language number and punctuation changes remain non-blocking review flags.

Native-French human review is still pending. Known ambiguous source corruptions
are listed in `fr_translation_qa_report.json` instead of being silently
corrected. Full model, prompt, usage, hash, and review provenance is recorded in
`fr_translation_provenance.json`.

A follow-up professional-translator review found three encoding/typography
defects that survived the original QA pass, fixed by
`scripts/clean_translation_handoff_fr_v2.py`: a leaked JSON escape sequence
for a non-breaking space (rendered as literal backslash-u-00a0 text instead
of a space) in 24 validation-split answers; a stray literal backslash kept
in 3 rows' garbled height notation while sibling rows had already dropped it;
and inconsistent straight versus typographic apostrophes across 1,257 rows,
now normalized to the corpus's typographic apostrophe (the two deliberately
preserved, garbled "Va's Difference" occurrences were left untouched). Logged
in the `post_delivery_cleanup` section of `fr_translation_qa_report.json`.

`fr.csv` also had no UTF-8 BOM, so Windows tools that guess encoding from
content instead of a declared one (notably Excel's default "Open" behavior)
fell back to a Windows ANSI code page and displayed accented letters and the
French guillemets as mojibake on screen, even though the bytes on disk were
already correct UTF-8. `scripts/add_utf8_bom.py` prepended the standard 3-byte
UTF-8 BOM; this changes no character of content and is safely stripped by
every reader in this repo that already opens the file as `utf-8-sig`
(`validate_translation_handoff_v2.py`, `prepare_ft_data_v2.py`). Separately,
an editor's "ambiguous Unicode characters" warning on this file is expected
and not a defect: it flags legitimate French typography (curly apostrophes,
guillemets) that merely looks similar to ASCII look-alikes.

## French ingest

The returned handoff contains both splits, so pass the same file as `--train`
and `--val`:

```powershell
python scripts/prepare_ft_data_v2.py --lang fr `
  --train Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/translation_handoff_v2/fr.csv `
  --val Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/translation_handoff_v2/fr.csv
```

Verified ingest output is 2,858 train and 718 validation rows. The leakage
guard drops four training rows as `train_val_dup` because synonymous English
questions (for example, “menses” versus “period”) correctly collapse to the
same natural French wording across splits.

Arabic remains a separate handoff and was not changed by the French workflow.
