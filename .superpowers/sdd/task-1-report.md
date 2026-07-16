# Task 1 Report: Authoring clarification_variants_v2.json + style_rewrites_v2.json

(Note: this file path is shared with another plan's "task 1" (safety_metrics.py, commit 36c2362); that report was overwritten here by concurrent sessions. This version is the data-authoring task report, including the pcos provenance fix.)

## Status: DONE (after provenance fix)

Commits:
- `b5fa5f5` — "data: Claude-authored clarification variants + style rewrites for mix v2"
- `bce9635` — "fix: re-derive pcos clarification items from their claimed source rows"

## Steps executed

1. Ran the Step-1 extraction script exactly as specified. Output: `870 source rows written` (290/category: fertility, menstrual, pcos — contiguous blocks, first 90 of each used for clarification, remaining 200 of each used for style rewrites).
2. Authored `HerHealthGPT-LU_seed/clarification_variants_v2.json` — 270 objects, built category-by-category (fertility, menstrual, pcos) in separate Python-embedded chunks, then merged.
3. Authored `HerHealthGPT-LU_seed/style_rewrites_v2.json` — 600 objects, same category-by-category build process.
4. Ran the Step-4 validation script verbatim.
5. Deleted the scratch file `_authoring_source_v2.json` and committed the two data files with the exact message specified in the brief.
6. (Follow-up) Applied the reviewer-requested pcos provenance fix — see the "Provenance fix" section at the end.

## Counts

**clarification_variants_v2.json** — 270 total
- category: `{'fertility': 90, 'menstrual': 90, 'pcos': 90}`
- register: `{'json': 180, 'chat': 90}` (60 json / 30 chat per category)
- all 90 `chat` items have a non-null `chat_reply`; all 180 `json` items have `chat_reply: null`
- 270/270 unique `vague_question`, 270/270 unique `clarifying_question`

**style_rewrites_v2.json** — 600 total
- category: `{'fertility': 200, 'menstrual': 200, 'pcos': 200}`
- style: `{'layperson': 201, 'indirect': 198, 'emotional': 201}` (67/66/67 per category)
- 600/600 unique `rewritten_question`

## Validation output (Step 4, run verbatim; re-run after the fix)

```
OK: 270 clarification + 600 style, balanced, unique
```

## Sample items (spot-check)

**Clarification, register="json"** (chat_reply is `null`):
```json
{
 "source_row_id": "HealthCareMagic-100k:19372",
 "category": "fertility",
 "vague_question": "I've been trying to have a baby for a while and nothing's working, is something wrong with me?",
 "register": "json",
 "clarifying_question": "How long have you been trying to conceive, and have you been diagnosed with any conditions like PCOS or ovarian cysts that might affect fertility?",
 "chat_reply": null
}
```
```json
{
 "source_row_id": "HealthCareMagic-100k:17002",
 "category": "fertility",
 "vague_question": "My doctor put me on a hormone therapy to help me get pregnant, but I'm not sure if it's the right call for my cycle. Should I trust it?",
 "register": "json",
 "clarifying_question": "What specific hormone medication and dosage did your doctor prescribe, and how long have you been trying to conceive?",
 "chat_reply": null
}
```
```json
{
 "source_row_id": "HealthCareMagic-100k:34042",
 "category": "fertility",
 "vague_question": "I did a fertility monitoring cycle and took some medication afterward, but my pregnancy test came back negative. What does that mean?",
 "register": "json",
 "clarifying_question": "What medications were you given after the monitored cycle, and how many days after your last dose did you take the pregnancy test?",
 "chat_reply": null
}
```

**Clarification, register="chat"** (warm reply that asks the clarifying question and points to a clinician):
```json
{
 "source_row_id": "HealthCareMagic-100k:102776",
 "category": "fertility",
 "vague_question": "I had an infection treated a long time ago as a child and I'm worried it could still affect my fertility now. Should I be concerned?",
 "register": "chat",
 "clarifying_question": "How long ago was the infection fully treated, and have you had any recent tests to confirm there's no lingering effect?",
 "chat_reply": "It makes sense that an old diagnosis would still weigh on your mind. How long ago was it fully treated, and have you had any recent tests confirming it's cleared? A doctor can order the right follow-up tests to reassure you either way."
}
```
```json
{
 "source_row_id": "HealthCareMagic-100k:15780",
 "category": "fertility",
 "vague_question": "I have a growth in my cervix and I'm not sure if it needs to come out before my fertility treatment. Could it get in the way?",
 "register": "chat",
 "clarifying_question": "What size is the polyp and where is it located, and what have the two doctors you consulted each recommended?",
 "chat_reply": "That's a smart thing to sort out before starting treatment. Do you know the size and location of the polyp, and what each doctor you saw recommended? Since the two opinions differ, it's worth getting a fertility specialist's clear read on whether it could affect implantation."
}
```

**Style rewrite, style="layperson"**:
```json
{
 "source_row_id": "HealthCareMagic-100k:19789",
 "category": "fertility",
 "style": "layperson",
 "rewritten_question": "My doc's got me on some injections every other day for trying to have a baby and said no sex for 2 months cuz my uterus is swollen. Is it actually not okay to do it while I'm on the shots?"
}
```

**Style rewrite, style="indirect"**:
```json
{
 "source_row_id": "MENST_train24K:train24K:1127",
 "category": "fertility",
 "style": "indirect",
 "rewritten_question": "Asking for my sister, not me... she gets a lot of pain around the middle of her cycle. Any idea, in simple terms, why that happens?"
}
```

**Style rewrite, style="emotional"**:
```json
{
 "source_row_id": "HealthCareMagic-100k:30969",
 "category": "fertility",
 "style": "emotional",
 "rewritten_question": "I already lost one pregnancy to a chromosome problem and now this next one looks like it's failing too with barely-rising numbers and empty sacs on the scan! Is there ANY chance this could still turn into a healthy pregnancy, and what on earth caused this to happen twice?"
}
```

## Concerns (original report)

1. **[RESOLVED — see "Provenance fix" below]** PCOS clarification provenance: originally flagged as 2 padded items; the data-quality reviewer found the real scope was ~50 items (positions 37-89) whose `vague_question` did not derive from their claimed source row, caused by positional drift during manual transcription.
2. **Off-by-one counting errors during authoring (self-caught and fixed).** Three times my initial 60/30 json/chat split landed on 59/31 or similar due to a miscount while transcribing into the Python literal. Each was caught by an `assert` before merging. The larger positional-drift bug (concern #1) was the same class of error but was not caught by count asserts, since counts stayed correct while alignment slipped.
3. **Not independently fact-checked against the frozen benchmark for lexical overlap.** I never opened or read `seeds_en_v1.*` during authoring (per instructions), but no automated overlap/leakage check was run. A follow-up similar to the repo's existing `leakage_log.csv` process would be reasonable.
4. Content register/style balance is correct per-category but not perfectly even at the 600-item aggregate level by design (67/66/67 per category → 201/198/201 aggregate), matching the brief's "roughly balanced" requirement.

---

## Provenance fix (follow-up, commit `bce9635`)

The data-quality reviewer confirmed concern #1 was under-scoped: pcos clarification items at positions 37-89 had `vague_question`s that did not derive from their claimed `source_row_id` rows (positional drift beginning at index 37, not just the 2 padded items). Menstrual and fertility blocks passed review.

**Fix applied** (`fix: re-derive pcos clarification items from their claimed source rows`):
- Re-ran the deterministic Step-1 extraction (seed 42) to recover the claimed source rows; verified positions 0-36 of the pcos block genuinely derive from their sources (they do — alignment breaks exactly at position 37).
- **Rewrote 53 items** (pcos positions 37-89): new `vague_question` + `clarifying_question` for all 53, plus new `chat_reply` for the 30 chat-register items, each authored directly from that position's actual source row topic.
- Preserved: category=pcos ×90, per-item registers (json ×60, chat ×30), global uniqueness of all 270 vague_questions AND all 270 clarifying_questions, file count 270.
- Added a hard provenance assertion to the fix script: every one of the 90 pcos items' `source_row_id` equals the positional source row's ID, with content authored from that row's text.
- Step-4 validation re-run verbatim: `OK: 270 clarification + 600 style, balanced, unique`
- Scratch `_authoring_source_v2.json` deleted again before committing.

**Before/after samples (the reviewer's three cited indices):**

*pcos position 39 — claimed source `MENST_train24K:train24K:8594` ("Can my wife completely cure PCOS with medication?"), register=json*
- BEFORE (misaligned, kidney/back-pain topic): "I recently found out about an unusual kidney finding along with ongoing back pain, and I also have a hormonal condition. Could they be connected somehow?"
- AFTER: "My wife takes medication for her ovary condition and we keep wondering whether it can ever make the condition go away for good." / clarifying: "What medication is she currently taking, and which symptom is it mainly meant to control?"

*pcos position 54 — claimed source `HealthCareMagic-100k:111421` (husband's semen analysis, astheno-terato-zoospermia; wife has hypothyroid + PCOS), register=json*
- BEFORE (misaligned, generic mild-condition topic): "I have a mild version of a hormonal condition and I'm trying for a baby. Is the treatment I've just started likely to work for me?"
- AFTER: "My fertility test came back with a complicated diagnosis, and my wife has her own hormone conditions too. Is having a baby still possible for us?" / clarifying: "What did the semen analysis show for count and motility, and is your wife's condition currently well controlled?"

*pcos position 71 — claimed source `HealthCareMagic-100k:21332` (16-year-old worried about delayed puberty), register=chat*
- BEFORE (misaligned, unexplained-pain topic): "I was diagnosed with a hormonal condition a while ago and have had ongoing pain along with other symptoms doctors haven't fully explained. Any help would mean a lot."
- AFTER: "I feel like my body is way behind everyone else my age in development, and I'm worried something I'm doing is holding it back. Is that even possible?" / clarifying: "Which specific developmental changes haven't you noticed yet, and how old are you now?" / chat_reply asks the question warmly and directs to a doctor for a proper development and hormone check.

Provenance is now exact for all 270 clarification items.
