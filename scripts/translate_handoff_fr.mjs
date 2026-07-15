#!/usr/bin/env node

/**
 * Generate the French v2 translation handoff with OpenAI GPT-5.6 Sol.
 *
 * The script intentionally has no npm dependencies. It uses Node's built-in
 * fetch implementation and a strict RFC-4180-style CSV parser/writer so the
 * handoff's embedded commas, quotes, and newlines survive round trips.
 *
 * Examples (run from the repository root):
 *
 *   node scripts/translate_handoff_fr.mjs --analyze-only
 *   node scripts/translate_handoff_fr.mjs --smoke --cache C:/tmp/herhealth-fr-cache.json
 *   node scripts/translate_handoff_fr.mjs \
 *     --cache C:/tmp/herhealth-fr-cache.json \
 *     --output Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/translation_handoff_v2/fr.generated.csv \
 *     --provenance Used_Datasets/Consolidated_Datasets/200_Seed_Dataset/translation_handoff_v2/fr_translation_provenance.json
 *
 * OPENAI_API_KEY must be present in the environment for generation. The API
 * key is never written to disk or included in logs.
 */

import { createHash } from "node:crypto";
import { promises as fs } from "node:fs";
import path from "node:path";

const BASE = "Used_Datasets/Consolidated_Datasets/200_Seed_Dataset";
const DEFAULTS = Object.freeze({
  input: `${BASE}/translation_handoff_v2/fr.csv`,
  output: `${BASE}/translation_handoff_v2/fr.generated.csv`,
  provenance: `${BASE}/translation_handoff_v2/fr_translation_provenance.json`,
  trainSource: `${BASE}/Train/train_canonical_styled.csv`,
  valSource: `${BASE}/validate/validation_canonical_styled.csv`,
  model: "gpt-5.6-sol",
  reasoningEffort: "low",
  questionBatchSize: 72,
  answerBatchSize: 36,
  concurrency: 3,
  maxOutputTokens: 20_000,
  timeoutMs: 10 * 60 * 1000,
  maxAttempts: 6,
});

const EXPECTED_HEADER = Object.freeze([
  "row_id",
  "split",
  "Question",
  "Answer",
  "Topic",
  "Keywords",
  "Question_translated",
  "Answer_translated",
]);
const IMMUTABLE_FIELDS = Object.freeze(EXPECTED_HEADER.slice(0, 6));
const ALLOWED_STYLES = new Set([
  "canonical",
  "clinical",
  "layperson",
  "indirect_cultural",
  "ambiguous",
  "emotionally_concerned",
]);

const SYSTEM_PROMPT = `You are a professional English-to-French medical localization translator.

Translate synthetic patient-style women's-health questions and informational answers into natural, idiomatic, France-neutral French for a multilingual LLM fine-tuning corpus.

Hard requirements:
- Preserve the complete meaning. Do not add, omit, diagnose, soften, intensify, or correct medical content.
- Translation is not fact-checking: faithfully translate unsafe, obsolete, or incorrect source advice without adding a warning, correction, or disclaimer.
- Preserve negation, uncertainty, risk, urgency, cautions, and recommendations exactly.
- Preserve digit quantities as digits without conversion or rounding; keep unit symbols, URL substrings, brand names, and placeholders unchanged.
- Preserve every source line break and blank line, and add none.
- Use medically accurate French terminology. Map bare PCOS or PCOD to "SOPK". Map a spelled-out polycystic ovary syndrome/disease to "syndrome des ovaires polykystiques"; add "(SOPK)" only when the source also includes or defines an acronym. Do not leave PCOS/PCOD in French, substitute another acronym, or introduce SOPK when the source avoids naming the condition.
- Use project-standard equivalents: PMS -> SPM, IVF -> FIV, IUI -> IIU; semen analysis -> spermogramme; semen -> sperme or liquide séminal; sperm -> spermatozoïde(s); infertility -> infertilité (not stérilité); miscarriage -> fausse couche; abortion -> avortement; ectopic pregnancy -> grossesse extra-utérine; emergency room -> service des urgences; emergency contraception -> contraception d'urgence.
- Keep questions as questions and answers as answers. Return no commentary, labels, or alternative translations inside translated text.
- Treat every item independently. Never use another batch item or metadata to infer details that this item omits.
- Use "vous" by default, and do not infer a person's gender when the source does not state it.

Question register is experimental data and must be preserved:
- canonical: natural original patient wording.
- clinical: formal clinical/chart-note register and precise terminology.
- layperson: everyday non-medical language; do not upgrade it to clinical French.
- indirect_cultural: retain euphemism and indirectness; do not name what the source avoids naming.
- ambiguous: retain vagueness and missing detail; do not resolve ambiguity.
- emotionally_concerned: retain anxiety, emphasis, and emotional intensity without exaggerating it.

For answer items, use clear natural patient-facing informational French while preserving the source's degree of formality and every safety qualification.

Return one faithful French translation for every supplied id using the required JSON schema.`;

function parseArgs(argv) {
  const options = { ...DEFAULTS, cache: null, analyzeOnly: false, smoke: false };
  const valueOptions = new Map([
    ["--input", "input"],
    ["--output", "output"],
    ["--provenance", "provenance"],
    ["--train-source", "trainSource"],
    ["--val-source", "valSource"],
    ["--cache", "cache"],
    ["--model", "model"],
    ["--reasoning-effort", "reasoningEffort"],
    ["--question-batch-size", "questionBatchSize"],
    ["--answer-batch-size", "answerBatchSize"],
    ["--concurrency", "concurrency"],
    ["--max-output-tokens", "maxOutputTokens"],
    ["--timeout-ms", "timeoutMs"],
    ["--max-attempts", "maxAttempts"],
  ]);

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--analyze-only") {
      options.analyzeOnly = true;
      continue;
    }
    if (arg === "--smoke") {
      options.smoke = true;
      continue;
    }
    if (arg === "--help" || arg === "-h") {
      printHelp();
      process.exit(0);
    }
    const key = valueOptions.get(arg);
    if (!key) throw new Error(`Unknown option: ${arg}`);
    if (i + 1 >= argv.length) throw new Error(`Missing value for ${arg}`);
    options[key] = argv[++i];
  }

  for (const key of [
    "questionBatchSize",
    "answerBatchSize",
    "concurrency",
    "maxOutputTokens",
    "timeoutMs",
    "maxAttempts",
  ]) {
    options[key] = Number(options[key]);
    if (!Number.isInteger(options[key]) || options[key] < 1) {
      throw new Error(`${key} must be a positive integer`);
    }
  }
  if (!new Set(["none", "low", "medium", "high", "xhigh", "max"]).has(options.reasoningEffort)) {
    throw new Error(`Unsupported reasoning effort: ${options.reasoningEffort}`);
  }
  if (!options.cache && !options.analyzeOnly) {
    throw new Error("--cache PATH is required for resumable generation");
  }
  return options;
}

function printHelp() {
  process.stdout.write(`Usage: node scripts/translate_handoff_fr.mjs [options]\n\n`);
  process.stdout.write(`  --analyze-only             Parse and validate inputs without calling the API\n`);
  process.stdout.write(`  --smoke                    Translate 8 representative items, cache them, and stop\n`);
  process.stdout.write(`  --cache PATH               Required resumable translation cache\n`);
  process.stdout.write(`  --input PATH               Blank French handoff template\n`);
  process.stdout.write(`  --output PATH              Completed CSV destination (must differ from input)\n`);
  process.stdout.write(`  --provenance PATH          JSON provenance destination\n`);
  process.stdout.write(`  --model ID                 Default: gpt-5.6-sol\n`);
  process.stdout.write(`  --reasoning-effort LEVEL   Default: low\n`);
  process.stdout.write(`  --concurrency N            Default: 3\n`);
}

function sha256(value) {
  return createHash("sha256").update(value).digest("hex");
}

function parseCsv(text, sourceName = "CSV") {
  let input = text;
  if (input.charCodeAt(0) === 0xfeff) input = input.slice(1);
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  let fieldStarted = false;

  const endField = () => {
    row.push(field);
    field = "";
    fieldStarted = false;
  };
  const endRow = () => {
    endField();
    rows.push(row);
    row = [];
  };

  for (let i = 0; i < input.length; i += 1) {
    const ch = input[i];
    if (quoted) {
      if (ch === '"') {
        if (input[i + 1] === '"') {
          field += '"';
          i += 1;
        } else {
          quoted = false;
        }
      } else {
        field += ch;
      }
      continue;
    }

    if (ch === '"') {
      if (fieldStarted || field.length > 0) {
        throw new Error(`${sourceName}: unexpected quote outside a quoted field`);
      }
      quoted = true;
      fieldStarted = true;
    } else if (ch === ",") {
      endField();
    } else if (ch === "\r" || ch === "\n") {
      endRow();
      if (ch === "\r" && input[i + 1] === "\n") i += 1;
    } else {
      field += ch;
      fieldStarted = true;
    }
  }
  if (quoted) throw new Error(`${sourceName}: unterminated quoted field`);
  if (field.length > 0 || fieldStarted || row.length > 0) endRow();
  if (rows.length === 0) throw new Error(`${sourceName}: empty CSV`);
  return rows;
}

function csvEscape(value) {
  const text = String(value ?? "");
  if (!/[",\r\n]/.test(text)) return text;
  return `"${text.replaceAll('"', '""')}"`;
}

function stringifyCsv(rows) {
  return `${rows.map((row) => row.map(csvEscape).join(",")).join("\r\n")}\r\n`;
}

function toRecords(csvRows, sourceName) {
  const [header, ...dataRows] = csvRows;
  const duplicateHeaders = header.filter((name, index) => header.indexOf(name) !== index);
  if (duplicateHeaders.length) {
    throw new Error(`${sourceName}: duplicate headers: ${[...new Set(duplicateHeaders)].join(", ")}`);
  }
  const records = dataRows.map((values, rowIndex) => {
    if (values.length !== header.length) {
      throw new Error(
        `${sourceName}: row ${rowIndex + 2} has ${values.length} columns; expected ${header.length}`,
      );
    }
    return Object.fromEntries(header.map((name, index) => [name, values[index]]));
  });
  return { header, records };
}

async function readCsv(filePath) {
  const raw = await fs.readFile(filePath);
  const text = raw.toString("utf8");
  if (text.includes("\u0000")) throw new Error(`${filePath}: contains NUL bytes`);
  if (text.includes("\ufffd")) throw new Error(`${filePath}: contains Unicode replacement characters`);
  return { raw, text, ...toRecords(parseCsv(text, filePath), filePath) };
}

function assertArrayEqual(actual, expected, label) {
  if (actual.length !== expected.length || actual.some((value, index) => value !== expected[index])) {
    throw new Error(`${label}: expected [${expected.join(", ")}], got [${actual.join(", ")}]`);
  }
}

function validateAndRecoverStyles(handoff, train, val) {
  assertArrayEqual(handoff.header, EXPECTED_HEADER, "handoff header");
  for (const [name, source] of [
    ["train source", train],
    ["validation source", val],
  ]) {
    for (const required of ["Question", "Answer", "Topic", "Keywords", "Style"]) {
      if (!source.header.includes(required)) throw new Error(`${name}: missing ${required}`);
    }
  }

  const seenIds = new Set();
  const styleCounts = new Map();
  const rows = handoff.records.map((row, position) => {
    if (seenIds.has(row.row_id)) throw new Error(`duplicate row_id: ${row.row_id}`);
    seenIds.add(row.row_id);
    const match = /^(train|val)-(\d{4})$/.exec(row.row_id);
    if (!match) throw new Error(`invalid row_id at handoff row ${position + 2}: ${row.row_id}`);
    const split = match[1];
    const sourceIndex = Number(match[2]);
    if (row.split !== split) {
      throw new Error(`${row.row_id}: split column is ${row.split}, expected ${split}`);
    }
    const source = split === "train" ? train.records[sourceIndex] : val.records[sourceIndex];
    if (!source) throw new Error(`${row.row_id}: source index is out of range`);
    for (const field of ["Question", "Answer", "Topic", "Keywords"]) {
      if (row[field] !== source[field]) {
        throw new Error(`${row.row_id}: immutable ${field} does not match the styled source`);
      }
    }
    if (!ALLOWED_STYLES.has(source.Style)) {
      throw new Error(`${row.row_id}: unknown Style ${source.Style}`);
    }
    if (row.Question_translated.trim() || row.Answer_translated.trim()) {
      throw new Error(`${row.row_id}: input template must have blank translation fields`);
    }
    if (!row.Question.trim() || !row.Answer.trim()) {
      throw new Error(`${row.row_id}: source Question/Answer must be nonblank`);
    }
    styleCounts.set(source.Style, (styleCounts.get(source.Style) ?? 0) + 1);
    return { ...row, Style: source.Style };
  });
  return { rows, styleCounts };
}

function makeJob(kind, text, style, topic) {
  const identity = JSON.stringify({ kind, text, style });
  const digest = sha256(identity);
  return {
    key: `${kind}:${digest}`,
    id: `${kind === "question" ? "q" : "a"}-${digest.slice(0, 20)}`,
    kind,
    text,
    style,
    topic,
    source_sha256: sha256(text),
  };
}

function buildJobs(rows) {
  const jobs = new Map();
  const rowKeys = [];
  const answerTopics = new Map();
  for (const row of rows) {
    const question = makeJob("question", row.Question, row.Style, row.Topic);
    const answer = makeJob("answer", row.Answer, "answer", row.Topic);
    if (!jobs.has(question.key)) jobs.set(question.key, question);
    if (!jobs.has(answer.key)) jobs.set(answer.key, answer);
    if (answerTopics.has(row.Answer) && answerTopics.get(row.Answer) !== row.Topic) {
      throw new Error("An identical English answer maps to multiple topics; answer reuse is unsafe");
    }
    answerTopics.set(row.Answer, row.Topic);
    rowKeys.push({ questionKey: question.key, answerKey: answer.key });
  }
  const ids = [...jobs.values()].map((job) => job.id);
  if (new Set(ids).size !== ids.length) throw new Error("translation job id collision");
  return { jobs: [...jobs.values()], rowKeys };
}

function newCache(options) {
  return {
    schema_version: 1,
    provider: "OpenAI",
    model: options.model,
    reasoning_effort: options.reasoningEffort,
    translations: {},
    batches: [],
  };
}

async function loadCache(cachePath, options) {
  try {
    const cache = JSON.parse(await fs.readFile(cachePath, "utf8"));
    if (cache.schema_version !== 1) throw new Error("unsupported cache schema_version");
    if (cache.model !== options.model || cache.reasoning_effort !== options.reasoningEffort) {
      throw new Error(
        `cache model configuration is ${cache.model}/${cache.reasoning_effort}, ` +
          `not ${options.model}/${options.reasoningEffort}`,
      );
    }
    if (!cache.translations || !Array.isArray(cache.batches)) throw new Error("malformed cache");
    return cache;
  } catch (error) {
    if (error?.code === "ENOENT") return newCache(options);
    throw new Error(`${cachePath}: ${error.message}`);
  }
}

async function saveJsonAtomic(filePath, value) {
  const resolved = path.resolve(filePath);
  await fs.mkdir(path.dirname(resolved), { recursive: true });
  const temporary = `${resolved}.tmp-${process.pid}`;
  await fs.writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`, "utf8");
  try {
    await fs.rename(temporary, resolved);
  } catch (error) {
    if (error?.code !== "EEXIST" && error?.code !== "EPERM") throw error;
    await fs.copyFile(temporary, resolved);
    await fs.unlink(temporary);
  }
}

function validateCacheEntries(cache, jobs) {
  const jobsByKey = new Map(jobs.map((job) => [job.key, job]));
  for (const [key, entry] of Object.entries(cache.translations)) {
    const job = jobsByKey.get(key);
    if (!job) continue;
    if (entry.source_sha256 !== job.source_sha256 || entry.id !== job.id) {
      throw new Error(`stale or mismatched cache entry: ${key}`);
    }
    validateTranslatedText(entry.translated, job.id);
  }
}

function validateTranslatedText(text, id) {
  if (typeof text !== "string" || !text.trim()) throw new Error(`${id}: blank translation`);
  if (text.includes("\u0000") || text.includes("\ufffd")) {
    throw new Error(`${id}: invalid Unicode in translation`);
  }
}

function chunk(items, size) {
  const result = [];
  for (let i = 0; i < items.length; i += size) result.push(items.slice(i, i + size));
  return result;
}

function makeBatches(jobs, options) {
  const groups = new Map();
  for (const job of jobs) {
    const groupKey = `${job.kind}:${job.style}`;
    if (!groups.has(groupKey)) groups.set(groupKey, []);
    groups.get(groupKey).push(job);
  }
  const batches = [];
  for (const [groupKey, groupJobs] of groups) {
    const size = groupJobs[0].kind === "question" ? options.questionBatchSize : options.answerBatchSize;
    for (const groupChunk of chunk(groupJobs, size)) batches.push({ groupKey, jobs: groupChunk });
  }
  return batches;
}

function smokeJobs(allJobs) {
  const selected = [];
  for (const style of ALLOWED_STYLES) {
    const job = allJobs.find((candidate) => candidate.kind === "question" && candidate.style === style);
    if (!job) throw new Error(`No smoke-test question found for style ${style}`);
    selected.push(job);
  }
  selected.push(...allJobs.filter((job) => job.kind === "answer").slice(0, 2));
  return selected;
}

function responseText(responseJson) {
  for (const output of responseJson.output ?? []) {
    for (const content of output.content ?? []) {
      if (content.type === "refusal") throw new Error(`model refusal: ${content.refusal ?? "unknown"}`);
      if (content.type === "output_text" && typeof content.text === "string") return content.text;
    }
  }
  throw new Error("response did not contain output_text");
}

function translationSchema(batch) {
  return {
    type: "object",
    properties: {
      translations: {
        type: "array",
        minItems: batch.length,
        maxItems: batch.length,
        items: {
          type: "object",
          properties: {
            id: { type: "string", enum: batch.map((job) => job.id) },
            translated: { type: "string", minLength: 1 },
          },
          required: ["id", "translated"],
          additionalProperties: false,
        },
      },
    },
    required: ["translations"],
    additionalProperties: false,
  };
}

function apiPayload(batch, options) {
  const items = batch.map((job) => ({
    id: job.id,
    kind: job.kind,
    register: job.style,
    text: job.text,
  }));
  return {
    model: options.model,
    reasoning: { effort: options.reasoningEffort },
    store: false,
    max_output_tokens: options.maxOutputTokens,
    input: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: `Translate every item in this JSON array:\n${JSON.stringify(items)}` },
    ],
    text: {
      format: {
        type: "json_schema",
        name: "french_translation_batch",
        strict: true,
        schema: translationSchema(batch),
      },
    },
  };
}

function retryDelayMs(attempt, retryAfter) {
  const retrySeconds = Number(retryAfter);
  if (Number.isFinite(retrySeconds) && retrySeconds > 0) return retrySeconds * 1000;
  return Math.min(60_000, 1_500 * 2 ** (attempt - 1)) + Math.floor(Math.random() * 750);
}

async function sleep(ms) {
  await new Promise((resolve) => setTimeout(resolve, ms));
}

async function callTranslationApi(batch, options) {
  const payload = apiPayload(batch, options);
  for (let attempt = 1; attempt <= options.maxAttempts; attempt += 1) {
    let response;
    try {
      response = await fetch("https://api.openai.com/v1/responses", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(options.timeoutMs),
      });
    } catch (error) {
      if (attempt === options.maxAttempts) throw error;
      const delay = retryDelayMs(attempt, null);
      process.stderr.write(`network error; retrying batch in ${Math.ceil(delay / 1000)}s (${attempt}/${options.maxAttempts})\n`);
      await sleep(delay);
      continue;
    }

    const raw = await response.text();
    let json;
    try {
      json = JSON.parse(raw);
    } catch {
      throw new Error(`OpenAI HTTP ${response.status}: non-JSON response`);
    }
    if (!response.ok) {
      const message = json?.error?.message ?? `HTTP ${response.status}`;
      const retryable = response.status === 408 || response.status === 409 || response.status === 429 || response.status >= 500;
      if (!retryable || attempt === options.maxAttempts) {
        throw new Error(`OpenAI HTTP ${response.status}: ${message}`);
      }
      const delay = retryDelayMs(attempt, response.headers.get("retry-after"));
      process.stderr.write(`OpenAI HTTP ${response.status}; retrying in ${Math.ceil(delay / 1000)}s (${attempt}/${options.maxAttempts})\n`);
      await sleep(delay);
      continue;
    }
    if (json.status === "incomplete") {
      const reason = json.incomplete_details?.reason ?? "unknown";
      throw new Error(`incomplete OpenAI response: ${reason}`);
    }

    const parsed = JSON.parse(responseText(json));
    const translations = parsed.translations;
    if (!Array.isArray(translations)) throw new Error("structured response lacks translations array");
    const byId = new Map();
    for (const item of translations) {
      if (byId.has(item.id)) throw new Error(`duplicate translated id: ${item.id}`);
      validateTranslatedText(item.translated, item.id);
      byId.set(item.id, item.translated);
    }
    const expectedIds = new Set(batch.map((job) => job.id));
    if (byId.size !== expectedIds.size || [...expectedIds].some((id) => !byId.has(id))) {
      throw new Error("structured response ids do not exactly match the requested batch");
    }
    return {
      responseId: json.id,
      translations: byId,
      usage: {
        input_tokens: json.usage?.input_tokens ?? null,
        output_tokens: json.usage?.output_tokens ?? null,
        total_tokens: json.usage?.total_tokens ?? null,
      },
    };
  }
  throw new Error("unreachable retry state");
}

async function translateBatches(batches, cache, cachePath, options) {
  let completed = 0;
  let nextBatch = 0;
  let commitChain = Promise.resolve();

  const commitResult = (batch, result) => {
    commitChain = commitChain.then(async () => {
      for (const job of batch.jobs) {
        cache.translations[job.key] = {
          id: job.id,
          kind: job.kind,
          style: job.style,
          source_sha256: job.source_sha256,
          translated: result.translations.get(job.id),
        };
      }
      cache.batches.push({
        response_id: result.responseId,
        completed_at: new Date().toISOString(),
        item_count: batch.jobs.length,
        ...result.usage,
      });
      completed += 1;
      await saveJsonAtomic(cachePath, cache);
      process.stdout.write(`completed ${completed}/${batches.length} API batches; cache saved\n`);
    });
    return commitChain;
  };

  const worker = async () => {
    while (true) {
      const batchIndex = nextBatch;
      nextBatch += 1;
      if (batchIndex >= batches.length) return;
      const batch = batches[batchIndex];
      const result = await callTranslationApi(batch.jobs, options);
      await commitResult(batch, result);
    }
  };

  const workerCount = Math.min(options.concurrency, batches.length);
  await Promise.all(Array.from({ length: workerCount }, () => worker()));
}

function usageTotals(cache) {
  const sum = (field) => {
    const values = cache.batches.map((batch) => batch[field]).filter(Number.isFinite);
    return values.length === cache.batches.length ? values.reduce((total, value) => total + value, 0) : null;
  };
  return {
    response_count: cache.batches.length,
    input_tokens: sum("input_tokens"),
    output_tokens: sum("output_tokens"),
    total_tokens: sum("total_tokens"),
  };
}

function outputRows(handoffHeader, sourceRows, rowKeys, cache) {
  const records = sourceRows.map((row, index) => {
    const keys = rowKeys[index];
    const question = cache.translations[keys.questionKey]?.translated;
    const answer = cache.translations[keys.answerKey]?.translated;
    validateTranslatedText(question, `${row.row_id}/Question_translated`);
    validateTranslatedText(answer, `${row.row_id}/Answer_translated`);
    const result = { ...row, Question_translated: question, Answer_translated: answer };
    for (const field of IMMUTABLE_FIELDS) {
      if (result[field] !== row[field]) throw new Error(`${row.row_id}: changed immutable ${field}`);
    }
    return result;
  });
  return [handoffHeader, ...records.map((record) => handoffHeader.map((field) => record[field]))];
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const inputPath = path.resolve(options.input);
  const outputPath = path.resolve(options.output);
  if (!options.analyzeOnly && inputPath === outputPath) {
    throw new Error("--output must differ from --input; validate the generated file before replacing the template");
  }

  const [handoff, train, val] = await Promise.all([
    readCsv(inputPath),
    readCsv(path.resolve(options.trainSource)),
    readCsv(path.resolve(options.valSource)),
  ]);
  const { rows, styleCounts } = validateAndRecoverStyles(handoff, train, val);
  const roundTrip = stringifyCsv(parseCsv(handoff.text, options.input));
  if (!handoff.raw.equals(Buffer.from(roundTrip, "utf8"))) {
    throw new Error("CSV parser/writer round trip is not byte-identical to the blank template");
  }
  const { jobs, rowKeys } = buildJobs(rows);
  const questionJobs = jobs.filter((job) => job.kind === "question");
  const answerJobs = jobs.filter((job) => job.kind === "answer");
  process.stdout.write(
    `${rows.length} rows; ${questionJobs.length} unique register-aware questions; ` +
      `${answerJobs.length} unique answers; styles ${JSON.stringify(Object.fromEntries(styleCounts))}\n`,
  );
  if (options.analyzeOnly) return;
  if (!process.env.OPENAI_API_KEY) throw new Error("OPENAI_API_KEY is not set");

  const cachePath = path.resolve(options.cache);
  const cache = await loadCache(cachePath, options);
  validateCacheEntries(cache, jobs);
  let requestedJobs = jobs.filter((job) => !cache.translations[job.key]);
  if (options.smoke) {
    requestedJobs = smokeJobs(jobs).filter((job) => !cache.translations[job.key]);
  }
  const batches = options.smoke && requestedJobs.length
    ? [{ groupKey: "smoke", jobs: requestedJobs }]
    : makeBatches(requestedJobs, options);
  if (batches.length) await translateBatches(batches, cache, cachePath, options);
  else process.stdout.write("all requested translations were already present in the cache\n");

  if (options.smoke) {
    process.stdout.write("smoke translation complete; no handoff file written\n");
    return;
  }

  const missing = jobs.filter((job) => !cache.translations[job.key]);
  if (missing.length) throw new Error(`${missing.length} translation jobs remain missing from the cache`);
  const generatedRows = outputRows(handoff.header, rows, rowKeys, cache);
  const generatedText = stringifyCsv(generatedRows);
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.writeFile(outputPath, generatedText, "utf8");

  const provenance = {
    schema_version: 1,
    artifact: path.relative(process.cwd(), outputPath).replaceAll("\\", "/"),
    language: "fr",
    corpus_role: "silver multilingual fine-tuning data",
    generated_at: new Date().toISOString(),
    generation: {
      provider: "OpenAI",
      api: "Responses API",
      model: options.model,
      reasoning_effort: options.reasoningEffort,
      structured_outputs: true,
      store: false,
      prompt_sha256: sha256(SYSTEM_PROMPT),
      usage: usageTotals(cache),
    },
    method: {
      style_recovered_from_stable_row_id: true,
      question_registers: [...ALLOWED_STYLES],
      identical_english_answers_translated_once_and_reused: true,
      only_translation_columns_filled: true,
    },
    counts: {
      rows: rows.length,
      train: rows.filter((row) => row.split === "train").length,
      validation: rows.filter((row) => row.split === "val").length,
      unique_question_jobs: questionJobs.length,
      unique_answer_jobs: answerJobs.length,
      style_rows: Object.fromEntries(styleCounts),
    },
    integrity: {
      blank_template_sha256: sha256(handoff.raw),
      train_styled_source_sha256: sha256(train.raw),
      validation_styled_source_sha256: sha256(val.raw),
      generated_csv_sha256: sha256(generatedText),
    },
    review: {
      automated_structural_validation: "pending",
      native_french_human_review: "pending",
      benchmark_quality_claim: false,
    },
  };
  await saveJsonAtomic(path.resolve(options.provenance), provenance);
  process.stdout.write(`wrote ${rows.length} completed rows -> ${outputPath}\n`);
  process.stdout.write(`wrote provenance -> ${path.resolve(options.provenance)}\n`);
}

main().catch((error) => {
  process.stderr.write(`ERROR: ${error.stack ?? error.message}\n`);
  process.exitCode = 1;
});
