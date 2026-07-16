# Benchmark translation handoff (FR / AR)

These are the 540 English benchmark questions to translate for cross-lingual
evaluation.

- Fill ONLY `Question_translated`. Leave every other column untouched.
- `item_key` (seed_id + style) is the join key -- it MUST come back unchanged.
- Translate meaning, register, and naturalness. Keep the communication style:
  a `layperson` question stays lay, a `clinical` one stays clinical, an
  `ambiguous` one stays vague. Do NOT clarify or expand ambiguous questions.
- Do NOT translate gold labels -- there are none here, by design. The gold
  interpretation is language-invariant and stays in the English Test file.
- Keep the file CSV, UTF-8. Return one completed file per language.
- Native-speaker validation for meaning preservation and register is REQUIRED
  before these are used for evaluation.
