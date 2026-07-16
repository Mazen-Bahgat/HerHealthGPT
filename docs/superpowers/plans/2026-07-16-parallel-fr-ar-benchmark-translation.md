# Parallel French and Arabic Benchmark Translation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Translate and fully machine-review all 540 benchmark questions into French and Modern Standard Arabic concurrently, then build gold-preserving per-language benchmark JSONL files.

**Architecture:** A dependency-free Node.js driver reuses the proven Responses API and resumable-cache pattern from `scripts/translate_handoff_fr.mjs`, but owns two independent language pipelines joined with `Promise.all`. A Python validator reconstructs immutable handoff rows from the canonical benchmark, while a separate structured semantic reviewer and fail-closed Python correction applicator keep generation, review, and mutation independently auditable.

**Tech Stack:** Node.js 20+ built-in `fetch`, Python 3.12+ standard library, `requests` only where already required by repository imports, pytest, OpenAI Responses API Structured Outputs.

## Global Constraints

- Follow `docs/superpowers/specs/2026-07-16-parallel-fr-ar-benchmark-translation-design.md` exactly.
- Use `gpt-5.6-sol`, low reasoning effort, Structured Outputs, and `store: false`; never fall back to another provider or model.
- Translate only `Question_translated`; preserve `item_key`, `seed_id`, `style`, `Topic`, `Question`, row order, and the 540-row count.
- Arabic is Modern Standard Arabic only; Egyptian and other dialects are out of scope.
- Preserve ambiguity, negation, urgency, uncertainty, person, tense, numerals, brands, URLs, measurements, and medical identity.
- Send no gold answers or labels to the translation or review API.
- Review all 540 pairs per language, not a sample.
- Keep native French and Arabic validation status `pending`; automated and AI-assisted review is not human sign-off.
- Preserve the user's existing `.gitignore` modification and untracked `final_version.zip`.

---

### Task 1: Deterministic Benchmark-Handoff Validator

**Files:**
- Create: `scripts/validate_benchmark_translation_handoff.py`
- Create: `tests/test_validate_benchmark_translation_handoff.py`
- Read: `scripts/build_benchmark_translation_handoff.py`

**Interfaces:**
- Consumes: canonical `gold_seeds_styled.jsonl`, one filled handoff CSV, and language code `fr` or `ar`.
- Produces: `ValidationResult(errors: list[str], warnings: list[str], row_count: int)` and CLI exit `0` only when `errors` is empty.
- Produces helper `validate_handoff(candidate: Path, benchmark: Path, language: str) -> ValidationResult` for later tests and final verification.

- [ ] **Step 1: Write failing validator tests**

```python
def test_valid_candidate_changes_only_translation(tmp_path):
    benchmark = _write_benchmark(tmp_path, [_bench("gss-000", "ambiguous", "Could this be something?")])
    candidate = _write_handoff(tmp_path, "fr", "Cela pourrait-il être quelque chose ?")
    result = validator.validate_handoff(candidate, benchmark, "fr")
    assert result.errors == []
    assert result.row_count == 1


def test_rejects_immutable_drift_and_blank_translation(tmp_path):
    benchmark = _write_benchmark(tmp_path, [_bench("gss-000", "canonical", "How is PCOS diagnosed?")])
    candidate = _write_handoff(tmp_path, "fr", "", question="How is it diagnosed?")
    result = validator.validate_handoff(candidate, benchmark, "fr")
    assert any("Question drift" in issue for issue in result.errors)
    assert any("blank Question_translated" in issue for issue in result.errors)


def test_arabic_dialect_and_latin_runs_are_review_flags_not_identity_errors(tmp_path):
    benchmark = _write_benchmark(tmp_path, [_bench("gss-000", "layperson", "Why is my period late?")])
    candidate = _write_handoff(tmp_path, "ar", "ليه الدورة متأخرة PCOS?")
    result = validator.validate_handoff(candidate, benchmark, "ar")
    assert result.errors == []
    assert any("possible dialect" in issue for issue in result.warnings)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
python -m pytest tests/test_validate_benchmark_translation_handoff.py -q
```

Expected: collection fails because `validate_benchmark_translation_handoff` does not exist.

- [ ] **Step 3: Implement the validator**

```python
@dataclass
class ValidationResult:
    errors: list[str]
    warnings: list[str]
    row_count: int


IMMUTABLE_FIELDS = ("item_key", "seed_id", "style", "Topic", "Question")
ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
FRENCH_SIGNAL_RE = re.compile(r"\b(?:le|la|les|un|une|des|est|sont|avec|pour|règles|grossesse)\b", re.I)
DIALECT_MARKERS = re.compile(r"(?:^|\s)(?:ليه|إزاي|ازاي|فين|إمتى|ازاى)(?:\s|$)")
URL_RE = re.compile(r"https?://\S+")


def expected_rows(benchmark: Path) -> list[dict[str, str]]:
    records = [json.loads(line) for line in benchmark.open(encoding="utf-8") if line.strip()]
    return build_benchmark_translation_handoff.build_rows(records)


def validate_handoff(candidate: Path, benchmark: Path, language: str) -> ValidationResult:
    expected = expected_rows(benchmark)
    rows = list(csv.DictReader(candidate.open(encoding="utf-8-sig", newline="")))
    errors: list[str] = []
    warnings: list[str] = []
    if len(rows) != len(expected):
        errors.append(f"row count: expected {len(expected)}, got {len(rows)}")
    for index, (want, got) in enumerate(zip(expected, rows, strict=False)):
        key = want["item_key"]
        for field in IMMUTABLE_FIELDS:
            if got.get(field) != want[field]:
                errors.append(f"{key}: {field} drift")
        translated = (got.get("Question_translated") or "").strip()
        if not translated:
            errors.append(f"{key}: blank Question_translated")
            continue
        if "\ufffd" in translated or any(ord(ch) < 32 and ch not in "\r\n\t" for ch in translated):
            errors.append(f"{key}: invalid replacement/control character")
        if URL_RE.findall(want["Question"]) != URL_RE.findall(translated):
            errors.append(f"{key}: URL mismatch")
        if language == "ar":
            if not ARABIC_RE.search(translated):
                warnings.append(f"{key}: weak Arabic-script signal")
            if DIALECT_MARKERS.search(translated):
                warnings.append(f"{key}: possible dialect marker")
        elif not FRENCH_SIGNAL_RE.search(translated):
            warnings.append(f"{key}: weak French-language signal")
    return ValidationResult(errors, warnings, len(rows))
```

The CLI prints a one-line PASS/FAIL summary plus each warning and returns nonzero on blocking errors.

- [ ] **Step 4: Run validator tests and existing handoff tests**

```powershell
python -m pytest tests/test_validate_benchmark_translation_handoff.py tests/test_translated_benchmark.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the validator**

```powershell
git add scripts/validate_benchmark_translation_handoff.py tests/test_validate_benchmark_translation_handoff.py
git commit -m "test: validate translated benchmark handoffs"
```

---

### Task 2: Resumable Parallel Translation Driver

**Files:**
- Create: `scripts/translate_benchmark_handoffs.mjs`
- Create: `tests/test_translate_benchmark_handoffs.mjs`
- Read: `scripts/translate_handoff_fr.mjs`

**Interfaces:**
- Consumes: blank `fr.csv` and `ar.csv`, `OPENAI_API_KEY`, per-language cache paths, `gpt-5.6-sol`.
- Produces: staged filled CSVs and generation provenance JSON for both languages.
- Exports pure helpers `parseCsv`, `stringifyCsv`, `makeJobs`, `validateCache`, and `fillRows` for Node tests.
- CLI: `node scripts/translate_benchmark_handoffs.mjs --cache-dir C:/tmp/herhealth-benchmark-translation --output-dir C:/tmp/herhealth-benchmark-filled`.

- [ ] **Step 1: Write failing Node tests for immutable fields, IDs, and cache drift**

```javascript
import test from "node:test";
import assert from "node:assert/strict";
import { fillRows, makeJobs, validateCache } from "../scripts/translate_benchmark_handoffs.mjs";

test("fillRows changes only Question_translated", () => {
  const rows = [{ item_key: "gss-000__ambiguous", seed_id: "gss-000", style: "ambiguous", Topic: "PCOS", Question: "Could this be something?", Question_translated: "" }];
  const jobs = makeJobs(rows, "ar");
  const cache = { translations: { [jobs[0].key]: { source_sha256: jobs[0].sourceSha256, translated: "هل يمكن أن يكون هذا شيئًا ما؟" } } };
  const filled = fillRows(rows, jobs, cache);
  assert.deepEqual({ ...filled[0], Question_translated: "" }, rows[0]);
  assert.equal(filled[0].Question_translated, "هل يمكن أن يكون هذا شيئًا ما؟");
});

test("validateCache rejects a stale source hash", () => {
  const jobs = [{ key: "fr:gss-000__canonical", sourceSha256: "new" }];
  assert.throws(() => validateCache({ translations: { [jobs[0].key]: { source_sha256: "old", translated: "Texte" } } }, jobs), /stale/);
});
```

- [ ] **Step 2: Run Node tests and verify RED**

```powershell
node --test tests/test_translate_benchmark_handoffs.mjs
```

Expected: module-not-found failure.

- [ ] **Step 3: Implement deterministic jobs and language prompts**

```javascript
const LANGUAGE_PROMPTS = Object.freeze({
  fr: `Translate each English women's-health benchmark question into natural, internationally understandable French. Preserve meaning, register, ambiguity, negation, urgency, uncertainty, person, tense, numerals, brands, URLs, measurements, and medical identity. Keep ambiguous questions vague. Return only the requested JSON items.`,
  ar: `Translate each English women's-health benchmark question into idiomatic Modern Standard Arabic only. Do not use Egyptian or another dialect. Preserve meaning, register, ambiguity, negation, urgency, uncertainty, person, tense, numerals, brands, URLs, measurements, and medical identity. Express informal or emotional styles through natural MSA wording. Keep ambiguous questions vague. Return only the requested JSON items.`,
});

export function makeJobs(rows, language) {
  return rows.map((row) => {
    const source = JSON.stringify({ item_key: row.item_key, style: row.style, Topic: row.Topic, Question: row.Question });
    return {
      id: row.item_key,
      key: `${language}:${row.item_key}`,
      language,
      source,
      sourceSha256: sha256(source),
    };
  });
}

export function validateCache(cache, jobs) {
  const byKey = new Map(jobs.map((job) => [job.key, job]));
  for (const [key, entry] of Object.entries(cache.translations ?? {})) {
    const job = byKey.get(key);
    if (!job || entry.source_sha256 !== job.sourceSha256 || !entry.translated?.trim()) {
      throw new Error(`stale or malformed cache entry: ${key}`);
    }
  }
}
```

- [ ] **Step 4: Implement Structured Outputs, retries, atomic cache writes, and parallel orchestration**

```javascript
async function requestBatch(batch, language, options) {
  const schema = {
    type: "object",
    additionalProperties: false,
    properties: {
      translations: {
        type: "array",
        minItems: batch.length,
        maxItems: batch.length,
        items: {
          type: "object",
          additionalProperties: false,
          properties: { id: { type: "string" }, translated: { type: "string", minLength: 1 } },
          required: ["id", "translated"],
        },
      },
    },
    required: ["translations"],
  };
  const response = await fetch("https://api.openai.com/v1/responses", {
    method: "POST",
    headers: { Authorization: `Bearer ${process.env.OPENAI_API_KEY}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      model: options.model,
      store: false,
      reasoning: { effort: "low" },
      input: [
        { role: "system", content: [{ type: "input_text", text: LANGUAGE_PROMPTS[language] }] },
        { role: "user", content: [{ type: "input_text", text: JSON.stringify(batch.map((job) => JSON.parse(job.source))) }] },
      ],
      text: { format: { type: "json_schema", name: `${language}_benchmark_translations`, strict: true, schema } },
      max_output_tokens: 16000,
    }),
  });
  if (!response.ok) throw new Error(`Responses API ${response.status}: ${await response.text()}`);
  return parseStructuredOutput(await response.json(), batch);
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const results = await Promise.all([
    runLanguage("fr", options),
    runLanguage("ar", options),
  ]);
  for (const result of results) process.stdout.write(`${result.language}: ${result.rows} translations complete\n`);
}
```

`runLanguage` uses batches of 45, at most two in-flight requests per language, six exponential-backoff attempts, an atomic `.tmp` cache replacement after every successful batch, and refuses to write a CSV until all 540 jobs exist. Provenance records prompt hash, source/output hashes, response IDs, token usage, model configuration, and `native_speaker_validation: "pending"`.

- [ ] **Step 5: Run Node tests and syntax validation**

```powershell
node --test tests/test_translate_benchmark_handoffs.mjs
node --check scripts/translate_benchmark_handoffs.mjs
```

Expected: all Node tests pass and syntax validation exits `0`.

- [ ] **Step 6: Commit the translation driver**

```powershell
git add scripts/translate_benchmark_handoffs.mjs tests/test_translate_benchmark_handoffs.mjs
git commit -m "feat: translate benchmark handoffs in parallel"
```

---

### Task 3: Full Semantic Reviewer and Fail-Closed Corrections

**Files:**
- Create: `scripts/review_benchmark_translations.mjs`
- Create: `scripts/apply_benchmark_translation_qa.py`
- Create: `tests/test_apply_benchmark_translation_qa.py`
- Modify: `tests/test_translate_benchmark_handoffs.mjs`

**Interfaces:**
- Consumes: filled CSVs, all 540 source/translation pairs, `gpt-5.6-sol`, and separate review caches.
- Produces: `<lang>_benchmark_translation_qa.json` with exactly 540 decisions and optional exact corrections.
- Produces `apply_corrections(rows: list[dict[str, str]], decisions: list[dict]) -> tuple[list[dict[str, str]], list[dict]]`.
- A correction is eligible only when `decision == "fix"`, `confidence == "high"`, `old_translation` exactly equals the current cell, and `new_translation` is nonblank.

- [ ] **Step 1: Write failing correction tests**

```python
def test_applies_only_exact_high_confidence_fix():
    rows = [_row("gss-000__canonical", "Ancien texte")]
    decisions = [{
        "item_key": "gss-000__canonical",
        "decision": "fix",
        "confidence": "high",
        "old_translation": "Ancien texte",
        "new_translation": "Nouveau texte",
        "reason": "negation fidelity",
    }]
    corrected, audit = qa.apply_corrections(rows, decisions)
    assert corrected[0]["Question_translated"] == "Nouveau texte"
    assert audit[0]["item_key"] == "gss-000__canonical"


def test_fails_closed_on_old_text_drift():
    rows = [_row("gss-000__canonical", "Current text")]
    decisions = [_fix("gss-000__canonical", old="Stale text", new="Fixed text")]
    with pytest.raises(qa.CorrectionError, match="old_translation drift"):
        qa.apply_corrections(rows, decisions)
```

- [ ] **Step 2: Run correction tests and verify RED**

```powershell
python -m pytest tests/test_apply_benchmark_translation_qa.py -q
```

Expected: import fails because the correction module does not exist.

- [ ] **Step 3: Implement the structured semantic review contract**

```javascript
const REVIEW_SCHEMA = {
  type: "object",
  additionalProperties: false,
  properties: {
    reviews: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        properties: {
          item_key: { type: "string" },
          decision: { type: "string", enum: ["pass", "fix", "human_review"] },
          severity: { type: "string", enum: ["none", "low", "medium", "high", "critical"] },
          confidence: { type: "string", enum: ["low", "medium", "high"] },
          checks: {
            type: "object",
            additionalProperties: false,
            properties: {
              meaning: { type: "boolean" }, negation: { type: "boolean" }, urgency: { type: "boolean" },
              uncertainty: { type: "boolean" }, numerals: { type: "boolean" }, terminology: { type: "boolean" },
              register: { type: "boolean" }, ambiguity_preserved: { type: "boolean" }, target_variety_compliant: { type: "boolean" },
            },
            required: ["meaning", "negation", "urgency", "uncertainty", "numerals", "terminology", "register", "ambiguity_preserved", "target_variety_compliant"],
          },
          old_translation: { type: "string" },
          new_translation: { type: "string" },
          reason: { type: "string" },
        },
        required: ["item_key", "decision", "severity", "confidence", "checks", "old_translation", "new_translation", "reason"],
      },
    },
  },
  required: ["reviews"],
};
```

The reviewer submits batches of 30 and instructs the model to compare, not retranslate by preference. `target_variety_compliant` means internationally understandable French for `fr` and MSA-only Arabic for `ar`, so one schema serves both languages. It validates one decision per requested `item_key`, persists a resumable cache, and blocks completion on missing decisions or unresolved `high`/`critical` `human_review` items.

- [ ] **Step 4: Implement fail-closed correction application**

```python
TRANSLATION_FIELD = "Question_translated"


def apply_corrections(rows: list[dict[str, str]], decisions: list[dict]) -> tuple[list[dict[str, str]], list[dict]]:
    corrected = [dict(row) for row in rows]
    by_key = {row["item_key"]: row for row in corrected}
    audit: list[dict] = []
    for decision in decisions:
        if decision["decision"] != "fix" or decision["confidence"] != "high":
            continue
        key = decision["item_key"]
        if key not in by_key:
            raise CorrectionError(f"unknown item_key: {key}")
        row = by_key[key]
        if row[TRANSLATION_FIELD] != decision["old_translation"]:
            raise CorrectionError(f"{key}: old_translation drift")
        new = decision["new_translation"].strip()
        if not new:
            raise CorrectionError(f"{key}: blank new_translation")
        row[TRANSLATION_FIELD] = new
        audit.append({
            "item_key": key,
            "old_translation": decision["old_translation"],
            "new_translation": new,
            "reason": decision["reason"],
        })
    return corrected, audit
```

The CLI preserves the five immutable columns, writes atomically, and appends correction counts plus source/output SHA-256 values to the QA report.

- [ ] **Step 5: Run reviewer/correction tests and syntax checks**

```powershell
python -m pytest tests/test_apply_benchmark_translation_qa.py -q
node --test tests/test_translate_benchmark_handoffs.mjs
node --check scripts/review_benchmark_translations.mjs
```

Expected: all tests pass.

- [ ] **Step 6: Commit semantic QA tooling**

```powershell
git add scripts/review_benchmark_translations.mjs scripts/apply_benchmark_translation_qa.py tests/test_apply_benchmark_translation_qa.py tests/test_translate_benchmark_handoffs.mjs
git commit -m "feat: review benchmark translations exhaustively"
```

---

### Task 4: Generate, Review, Correct, and Install Both Languages

**Files:**
- Modify: `Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/Test/translation_handoff_benchmark/fr.csv`
- Modify: `Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/Test/translation_handoff_benchmark/ar.csv`
- Create: `Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/Test/translation_handoff_benchmark/fr_benchmark_translation_provenance.json`
- Create: `Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/Test/translation_handoff_benchmark/ar_benchmark_translation_provenance.json`
- Create: `Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/Test/translation_handoff_benchmark/fr_benchmark_translation_qa.json`
- Create: `Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/Test/translation_handoff_benchmark/ar_benchmark_translation_qa.json`

**Interfaces:**
- Consumes the tools from Tasks 1–3 and the configured `OPENAI_API_KEY`.
- Produces two filled, reviewed, deterministically valid 540-row CSV files.

- [ ] **Step 1: Analyze both blank handoffs without calling the API**

```powershell
node scripts/translate_benchmark_handoffs.mjs --analyze-only
```

Expected: `fr: 540 jobs, 0 cached` and `ar: 540 jobs, 0 cached`, with 90 rows per style for each language.

- [ ] **Step 2: Launch both translation pipelines concurrently**

```powershell
node scripts/translate_benchmark_handoffs.mjs --cache-dir C:/tmp/herhealth-benchmark-translation --output-dir C:/tmp/herhealth-benchmark-filled
```

Expected: both language progress streams advance independently and finish with `540 translations complete`. If the process is interrupted, rerun the identical command; completed cache entries must be reused.

- [ ] **Step 3: Deterministically validate staged CSVs**

```powershell
python scripts/validate_benchmark_translation_handoff.py C:/tmp/herhealth-benchmark-filled/fr.csv --language fr
python scripts/validate_benchmark_translation_handoff.py C:/tmp/herhealth-benchmark-filled/ar.csv --language ar
```

Expected: 540 rows and zero blocking errors for both languages. Retain warning counts in provenance.

- [ ] **Step 4: Review all 1,080 pairs concurrently**

```powershell
node scripts/review_benchmark_translations.mjs --input-dir C:/tmp/herhealth-benchmark-filled --cache-dir C:/tmp/herhealth-benchmark-review --output-dir C:/tmp/herhealth-benchmark-review-reports
```

Expected: exactly 540 decisions for French and 540 for Arabic, with no missing or duplicate `item_key`.

- [ ] **Step 5: Apply exact high-confidence corrections**

```powershell
python scripts/apply_benchmark_translation_qa.py --language fr --input C:/tmp/herhealth-benchmark-filled/fr.csv --report C:/tmp/herhealth-benchmark-review-reports/fr_benchmark_translation_qa.json --output C:/tmp/herhealth-benchmark-filled/fr.reviewed.csv
python scripts/apply_benchmark_translation_qa.py --language ar --input C:/tmp/herhealth-benchmark-filled/ar.csv --report C:/tmp/herhealth-benchmark-review-reports/ar_benchmark_translation_qa.json --output C:/tmp/herhealth-benchmark-filled/ar.reviewed.csv
```

Expected: every applied correction has an exact old-text match and audit entry. Any high/critical `human_review` decision stops installation and is reported to the user rather than silently guessed.

- [ ] **Step 6: Revalidate and rerun semantic review on corrected outputs**

```powershell
python scripts/validate_benchmark_translation_handoff.py C:/tmp/herhealth-benchmark-filled/fr.reviewed.csv --language fr
python scripts/validate_benchmark_translation_handoff.py C:/tmp/herhealth-benchmark-filled/ar.reviewed.csv --language ar
node scripts/review_benchmark_translations.mjs --fr C:/tmp/herhealth-benchmark-filled/fr.reviewed.csv --ar C:/tmp/herhealth-benchmark-filled/ar.reviewed.csv --cache-dir C:/tmp/herhealth-benchmark-review-final --output-dir C:/tmp/herhealth-benchmark-review-final-reports
```

Expected: zero blocking deterministic errors and no unresolved high/critical semantic decisions.

- [ ] **Step 7: Atomically install reviewed CSVs and reports**

Use `Copy-Item` only after Step 6 passes:

```powershell
Copy-Item -LiteralPath C:/tmp/herhealth-benchmark-filled/fr.reviewed.csv -Destination Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/Test/translation_handoff_benchmark/fr.csv
Copy-Item -LiteralPath C:/tmp/herhealth-benchmark-filled/ar.reviewed.csv -Destination Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/Test/translation_handoff_benchmark/ar.csv
Copy-Item -LiteralPath C:/tmp/herhealth-benchmark-review-final-reports/fr_benchmark_translation_qa.json -Destination Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/Test/translation_handoff_benchmark/fr_benchmark_translation_qa.json
Copy-Item -LiteralPath C:/tmp/herhealth-benchmark-review-final-reports/ar_benchmark_translation_qa.json -Destination Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/Test/translation_handoff_benchmark/ar_benchmark_translation_qa.json
```

Expected: repository handoffs are replaced only after all gates pass.

- [ ] **Step 8: Commit the filled handoffs and audit artifacts**

```powershell
git add Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/Test/translation_handoff_benchmark/fr.csv Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/Test/translation_handoff_benchmark/ar.csv Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/Test/translation_handoff_benchmark/*benchmark_translation*.json
git commit -m "data: translate FR and MSA benchmark handoffs"
```

---

### Task 5: Build Gold-Preserving Benchmarks and Document the Delivery

**Files:**
- Modify: `tests/test_translated_benchmark.py`
- Modify: `Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/Test/translation_handoff_benchmark/README.md`
- Modify: `scripts/build_benchmark_translation_handoff.py`
- Create: `Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/gold_seeds_styled_fr.jsonl`
- Create: `Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/gold_seeds_styled_ar.jsonl`

**Interfaces:**
- Consumes the two final handoff CSVs and canonical English benchmark.
- Produces two 540-row JSONL files whose only canonical-field difference is `style_text`.

- [ ] **Step 1: Add failing integration tests for actual artifacts and gold preservation**

```python
@pytest.mark.parametrize("language", ["fr", "ar"])
def test_completed_benchmark_handoff_round_trip_preserves_gold(language):
    root = Path(__file__).resolve().parents[1]
    data = root / "Used_Datasets/Consolidated_Datasets/200_Seed_Dataset"
    bench = [json.loads(line) for line in (data / "gold_seeds_styled.jsonl").read_text(encoding="utf-8").splitlines() if line]
    output = data / f"gold_seeds_styled_{language}.jsonl"
    assert output.exists()
    translated = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line]
    assert len(translated) == 540
    for source, target in zip(bench, translated, strict=True):
        assert target["style_text"] != source["style_text"]
        assert {k: v for k, v in target.items() if k != "style_text"} == {
            k: v for k, v in source.items() if k != "style_text"
        }
```

- [ ] **Step 2: Run the integration test and verify RED before installing outputs**

```powershell
python -m pytest tests/test_translated_benchmark.py::test_completed_benchmark_handoff_round_trip_preserves_gold -q
```

Expected: failure because `gold_seeds_styled_fr.jsonl` and
`gold_seeds_styled_ar.jsonl` have not been built yet.

- [ ] **Step 3: Build both translated benchmark files**

```powershell
python scripts/build_translated_benchmark.py --lang fr --translated Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/Test/translation_handoff_benchmark/fr.csv
python scripts/build_translated_benchmark.py --lang ar --translated Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/Test/translation_handoff_benchmark/ar.csv
```

Expected: `540 translated benchmark items` for each language.

- [ ] **Step 4: Rerun the integration test and verify GREEN**

```powershell
python -m pytest tests/test_translated_benchmark.py::test_completed_benchmark_handoff_round_trip_preserves_gold -q
```

Expected: both language cases pass.

- [ ] **Step 5: Verify local-inference input compatibility without loading a model**

```powershell
python -m pytest tests/test_run_local_inference.py tests/test_translated_benchmark.py -q
```

Expected: all tests pass; `iter_benchmark` reads 540 records for both built files and `record_for_row` stamps `fr`/`ar` language IDs correctly.

- [ ] **Step 6: Document exact generation, QA, and pending human review**

Update both the checked-in handoff README and the `README` constant in
`scripts/build_benchmark_translation_handoff.py` with the same text. Record:

```markdown
## Completed model-assisted delivery (2026-07-16)

Both handoffs contain 540 translations. French uses natural international
French; Arabic uses Modern Standard Arabic only. Generation used OpenAI's
Responses API with `gpt-5.6-sol`, low reasoning, Structured Outputs, and
`store: false`. Deterministic validation and AI-assisted semantic review cover
all 540 items per language. Native-speaker validation remains pending, so these
files must not yet be described as human-validated or benchmark-ready.
```

Add final CSV and JSONL SHA-256 values, usage totals, correction counts, warning
counts, and QA report paths to each provenance JSON.

- [ ] **Step 7: Run the full focused verification suite**

```powershell
python -m pytest tests/test_validate_benchmark_translation_handoff.py tests/test_apply_benchmark_translation_qa.py tests/test_translated_benchmark.py tests/test_run_local_inference.py -q
node --test tests/test_translate_benchmark_handoffs.mjs
node --check scripts/translate_benchmark_handoffs.mjs
node --check scripts/review_benchmark_translations.mjs
git diff --check
```

Expected: all tests and syntax checks pass; `git diff --check` exits `0`.

- [ ] **Step 8: Commit the built benchmarks and documentation**

```powershell
git add Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/gold_seeds_styled_fr.jsonl Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/gold_seeds_styled_ar.jsonl Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/Test/translation_handoff_benchmark/README.md scripts/build_benchmark_translation_handoff.py tests/test_translated_benchmark.py
git commit -m "data: build French and MSA evaluation benchmarks"
```

## Final Verification Checklist

- [ ] Both CSVs have 540 rows, 540 unique ordered keys, and no blank translation.
- [ ] Both CSVs differ from the canonical handoff only in `Question_translated`.
- [ ] Arabic deterministic review reports no unresolved dialect violations.
- [ ] Semantic QA reports contain 540 decisions per language and no unresolved high/critical issue.
- [ ] French and Arabic JSONL outputs each contain 540 rows.
- [ ] Every gold field is exactly identical to English for every item.
- [ ] Provenance hashes match the installed artifacts.
- [ ] README and builder README constant are identical.
- [ ] Native-speaker status is explicitly `pending` in every relevant artifact.
