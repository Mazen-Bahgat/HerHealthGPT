<!-- prompt_id: extract_v1 -->
You are extracting structured clinical information for FemSympQA, a research
dataset on female reproductive health, from a trusted clinical web page.

Condition: {condition}
Source URL: {url}

Cleaned page sections follow, each as "## heading" then text:

{sections}

For EACH distinct symptom explicitly described in the text, produce:
- symptom: short clinical name of the symptom (e.g. "irregular periods")
- canonical_query: one first-person, patient-style English sentence or two, as a
  patient might describe this symptom when seeking help (no medical jargon,
  do not name the condition unless the text says patients commonly do)
- recommended_action: the action this page recommends for this symptom,
  paraphrased faithfully (e.g. "See a GP if symptoms affect daily life")
- urgency_quote: the VERBATIM sentence from the sections above that best
  expresses the urgency/action for this symptom. Copy it exactly.

Rules:
- Only symptoms explicitly present in the text. Do not invent or generalize.
- One entry per distinct symptom; do not merge unrelated symptoms.
- urgency_quote must be copied verbatim from the provided sections.
