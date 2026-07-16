# Related-Work Citation Integration Design

**Date:** 2026-07-16  
**Target manuscript:** `Overleaf_Template/overleaf_intro_related.tex`  
**Bibliography:** `Overleaf_Template/custom.bib`

## Objective

Strengthen the HerHealthGPT-LU literature review by integrating the relevant papers identified in the local Related Work collection. The revision must accurately position the benchmark against prior women's-health LLM systems, health-domain multilingual evaluations, and methodological work on clarification, cultural variation, terminology, and model-based evaluation.

## Scope

The revision will include two source groups:

1. Essential domain literature: WHBench; cross-language consistency of health-related LLM answers; Mai; direct PCOS evaluations; fertility and infertility evaluations; and the PCOS systematic review.
2. Methodological literature: common-ground and clarification research; cultural-awareness evaluation; terminology stewardship in translation; and LLM-as-a-judge reliability when model-based judging is part of the reported evaluation.

Papers in the local directory that address unrelated tasks, such as visual entailment, offensive-language generation, opinion generation, or generic prompt-search methods, will not be cited merely because they are locally available.

## Citation Strategy

Use narrative integration rather than citation dumping. Every added citation must support a nearby factual or comparative claim. A source will not be cited for a stronger conclusion than its study design supports.

The existing Related Work structure will be retained and revised as follows:

- **Women's Health, Medical QA, and LLMs:** add WHBench and direct evaluations involving PCOS, infertility, and fertility counselling. Distinguish expert-authored scenario evaluation from HerHealthGPT-LU's matched patient-expression evaluation.
- **Domain Adaptation for Women's-Health LLMs:** discuss Mai alongside MenstLLaMA. Replace the claim that MenstLLaMA alone is the closest prior system with a qualified comparison covering both systems.
- **Multilingual Clinical NLP and Cross-Cultural Evaluation:** prioritize the health-specific cross-language consistency study, then use cultural-awareness and terminology work to justify semantic and terminology controls in translation.
- **Safety, Ambiguity, and Clarification:** add common-ground research to motivate explicit clarification measurement. Add LLM-as-a-judge reliability work only where the manuscript describes a model-based evaluator; otherwise omit it from the prose and bibliography.
- **Positioning:** expand the table to include WHBench, Mai, and MenstLLaMA, and compare task, language coverage, controlled style variation, and safety/clarification supervision.

## Novelty Framing

The manuscript will not claim that prior work lacks women's-health benchmarks, multilingual health evaluation, or domain-specific chatbots. The defensible gap is narrower:

> Existing women's-health benchmarks and domain-specific systems evaluate clinical response quality, expert-authored scenarios, or educational generation, but do not jointly isolate style-controlled patient-language variation, triage calibration, and clarification behavior over matched symptom cases.

This positioning must remain consistent with the project's implemented evidence. Claims about human validation, multilingual benchmark results, or multilingual fine-tuning results must not be strengthened through the citation revision.

## Bibliography Design

- Add complete BibTeX records for non-ACL papers to `custom.bib`, using DOI or stable publication metadata where available.
- Resolve LUHME papers through the existing ACL Anthology bibliography shards when possible.
- Use stable, descriptive citation keys and avoid duplicate records already supplied by the anthology shards.
- Preserve ACL/Natbib-compatible capitalization with braces around acronyms and model names.

## Verification

After implementation:

1. Confirm every `\cite`, `\citet`, and `\citep` key resolves in `custom.bib` or the declared ACL Anthology shards.
2. Confirm each newly added bibliography entry is cited in the manuscript and no removed key is left dangling.
3. Check DOI and title metadata against the local paper and, when needed, the publisher or ACL Anthology record.
4. Compile the LaTeX project if a compiler is available; otherwise report that limitation and run static citation-key checks.
5. Review the positioning table for ACL column-width and readability constraints.

## Out of Scope

- Rewriting the Methods, Results, Discussion, or Limitations sections.
- Claiming evaluation results that are not present in the repository.
- Adding every paper from the Related Work directory irrespective of relevance.
- Changing benchmark data, training code, or evaluation code.

## Success Criteria

- Essential and methodological prior work is cited where it directly supports the narrative.
- The bibliography contains accurate records for all newly cited non-ACL sources.
- The Related Work section no longer presents MenstLLaMA as the sole closest system.
- The novelty claim is accurate relative to WHBench, Mai, and multilingual health-consistency research.
- Static citation checks pass, and compilation passes if the required LaTeX tooling is installed.
