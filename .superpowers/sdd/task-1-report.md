# Task 1 report: `run_local_inference.py` — runner + unit tests

## Scope executed

Steps 1–5 and Step 7 (commit) per the controller's instructions. **Step 6 (GPU
smoke run) was explicitly skipped** — left for the controller to run on the
WSL GPU box.

## Files created

- `scripts/run_local_inference.py` — local (no-server) benchmark inference
  runner for M2/M3. Written verbatim from the brief (Step 3 code block):
  - `iter_benchmark(path) -> list[dict]`
  - `done_item_ids(path) -> set[str]`
  - `record_for_row(row, raw_response, model, label, row_number) -> dict`
    (delegates to `run_inference.parse_model_content` /
    `build_output_record` / `select_input_text` — no reimplementation)
  - `LocalGenerator` class: lazily imports `unsloth`/`torch` inside
    `__init__`/`__call__` only, so the module itself imports cleanly with
    no heavy deps present (verified: Windows `.venv` has neither torch nor
    unsloth installed, and the test module still imports/collects/runs).
    Uses `FastLanguageModel.from_pretrained(..., load_in_4bit=True,
    dtype=None)`, `enable_thinking=False`, `do_sample=False`, greedy decode.
  - `main()` CLI: `--label`, `--model`, `--adapter`, `--benchmark`,
    `--output`, `--limit`, `--max-new-tokens` (default 512). Appends to
    output JSONL, skips already-done `item_id`s (resume-safe), flushes each
    line, catches per-row generation exceptions and records them as
    `generation_error` rather than aborting the run.
- `tests/test_run_local_inference.py` — the exact 3-test file from the brief
  (Step 1 code block), unmodified.

## Files removed

- `scripts/run_local_raw_baseline.py` (superseded single-purpose Qwen3.5 raw
  baseline runner — its functionality is now covered by
  `run_local_inference.py --label M2` with no adapter).
- `tests/test_run_local_raw_baseline.py` (its test suite, 6 tests).
- Confirmed via `grep -rn "run_local_raw_baseline" scripts/ tests/` → no
  matches in either directory after removal.

## Test execution (exact commands + output)

Command specified by the controller:
```
.venv/Scripts/python.exe -m pytest tests/test_run_local_inference.py -v
```

**Note on environment:** on this machine, plain `pytest` runs (both before
and unrelated to my changes) hit a pre-existing Windows permission error
against the shared `C:\Users\SW2\AppData\Local\Temp\pytest-of-SW2` directory
used by pytest's `tmp_path` fixture (`PermissionError: [WinError 5] Access is
denied`) — almost certainly a directory ACL left behind by a concurrent/
previous process on this multi-agent machine, not something specific to this
task. It also manifested as a benign `PytestCacheWarning` on `.pytest_cache`.
I worked around both by pointing `--basetemp` at the writable scratchpad
directory and disabling cacheprovider; no test code or fixture behavior was
changed. This is an environment quirk, not a code defect — flagging for
visibility.

Step 2 (pre-implementation, confirming the intended failure):
```
$ .venv/Scripts/python.exe -m pytest tests/test_run_local_inference.py -v
...
ModuleNotFoundError: No module named 'run_local_inference'
=========================== 2 warnings, 1 error in 0.17s ===========================
```
Matches the brief's expected failure exactly.

Step 4 (post-implementation):
```
$ .venv/Scripts/python.exe -m pytest tests/test_run_local_inference.py -v \
    --basetemp="<scratchpad>/pytest-tmp"
tests/test_run_local_inference.py::test_record_for_row_is_scorable_by_evaluate PASSED [ 33%]
tests/test_run_local_inference.py::test_record_for_row_marks_bad_json_unparsed PASSED [ 66%]
tests/test_run_local_inference.py::test_done_item_ids_reads_existing_output PASSED [100%]
============================== 3 passed in 0.07s ==============================
```
3/3 pass — matches the brief's expected "3 passed".

Full repo test suite, before removal (sanity check nothing else broke):
```
75 passed in 0.48s
```

Full repo test suite, after removing the superseded raw-baseline
script+test (Step 5):
```
69 passed in 0.23s
```
75 − 6 (the removed `test_run_local_raw_baseline.py` tests) = 69. Confirms
the removal didn't silently break any other test file (nothing else imported
`run_local_raw_baseline`).

## Self-review notes

- The runner file and test file were written byte-for-byte from the brief's
  code blocks — no deviations. I did not invent any additional logic beyond
  what's specified.
- Verified `run_inference.py` actually exposes `FIXED_PROMPT_TEMPLATE`,
  `parse_model_content`, `build_output_record`, `select_input_text`,
  `build_item_id` with the signatures `record_for_row`/`main()` rely on —
  read the full file before trusting the brief's DRY claim; signatures match.
- Verified `evaluate.py`'s `score_record()` consumes the record shape
  `record_for_row` produces (`item_id`, `model_label`, `gold_category`,
  `gold_risk_level`, `requires_clarification`, `_parse_error`,
  `predicted_category`/`predicted_risk`/`asks_clarification` via
  `validate_prediction_object`) — confirmed by the passing
  `test_record_for_row_is_scorable_by_evaluate` test, which round-trips a
  record through `ev.score_record` and asserts `parse_ok`,
  `category_correct`, `risk_correct`, `clarification_correct` are all
  `True`.
- Confirmed `LocalGenerator` does not import `unsloth`/`torch` at module
  scope (only inside `__init__`/`__call__`), so the module imports fine on
  the CPU-only Windows venv, which lacks those heavy deps — this was
  load-bearing for the unit tests to run at all in this environment.
- Git line-ending warnings ("LF will be replaced by CRLF") appeared on `git
  add` for both new files — this is the repo's existing `core.autocrlf`
  behavior (not something I configured) and does not affect file content on
  disk; ignorable.
- `git status` is clean after commit; working tree ahead of
  `origin/feat/qwen35-en-finetune` by 3 commits (2 pre-existing + this
  one, `42c11c6`). Not pushed, per instructions.

## Concerns

- **Step 6 (GPU smoke run) was not executed**, per explicit scope
  instructions — the controller will run it on WSL. This means the
  `LocalGenerator` class (the only piece with real Unsloth/torch/CUDA
  interaction) has **not been exercised end-to-end**; only its pure-Python
  siblings (`iter_benchmark`, `done_item_ids`, `record_for_row`, and the CLI
  argument wiring in `main()`) are covered by the unit tests. Any bug in the
  chat-template/generation/decode path (e.g. `apply_chat_template` kwargs,
  `.to("cuda")`, tokenizer decode slicing) would only surface during the
  skipped smoke step.
- The pytest `tmp_path`/`--basetemp` permission issue described above is an
  environment artifact on this Windows machine (shared temp dir ACL), not
  introduced by this task — worth a one-time look if it recurs across
  sessions, but out of scope here.

## Follow-up fix (post-GPU-smoke): `apply_chat_template` return_dict

The controller's GPU smoke confirmed the concern above: the untested
`LocalGenerator.__call__` path had a real bug. Every `generate()` raised
`TypeError: string indices must be integers, not 'str'`.

Root cause: in transformers 5.5.0, `tokenizer.apply_chat_template(...,
tokenize=True, return_tensors="pt")` returns a **BatchEncoding dict**, not a
bare tensor. So the original `generate(input_ids=inputs)` and `inputs.shape[1]`
were operating on a dict, not a tensor.

Fix (only `__call__` changed): request `return_dict=True`, unpack with
`generate(**inputs, ...)`, and slice using `inputs["input_ids"].shape[1]` for
the prompt length. Everything else in the module is unchanged.

The three pure-function tests still pass after the fix (heavy deps still load
lazily, so this GPU path remains unit-uncovered — the controller re-runs the
GPU smoke to validate real generation):
```
$ .venv/Scripts/python.exe -m pytest tests/test_run_local_inference.py -v --basetemp=<scratch>
tests/test_run_local_inference.py::test_record_for_row_is_scorable_by_evaluate PASSED [ 33%]
tests/test_run_local_inference.py::test_record_for_row_marks_bad_json_unparsed PASSED [ 66%]
tests/test_run_local_inference.py::test_done_item_ids_reads_existing_output PASSED [100%]
============================== 3 passed in 0.07s ==============================
```
Committed as `fix: apply_chat_template return_dict for transformers 5.5 generate`.

## Follow-up fix 2 (confirmed root cause): typed content parts for Qwen3-VL

The controller's direct probe confirmed the true primary root cause: the
model's processor is a **`Qwen3VLProcessor`** — Qwen3.5-9B is a
vision-language model. Passing the chat message `content` as a **bare string**
raises `string indices must be integers`; passing it as a **typed text part**
(`content=[{"type": "text", "text": prompt}]`) works and returns
`input_ids` / `attention_mask` / `mm_token_type_ids`.

The earlier `return_dict=True` change remains correct and stays. This fix is
the additional, primary correction: in `LocalGenerator.__call__`, the message
content changed from a bare string to a typed text part. Nothing else changed.

The three pure-function tests still pass after this fix (GPU path still
unit-uncovered; controller re-runs the GPU smoke to validate real generation):
```
$ .venv/Scripts/python.exe -m pytest tests/test_run_local_inference.py -v --basetemp=<scratch>
tests/test_run_local_inference.py::test_record_for_row_is_scorable_by_evaluate PASSED [ 33%]
tests/test_run_local_inference.py::test_record_for_row_marks_bad_json_unparsed PASSED [ 66%]
tests/test_run_local_inference.py::test_done_item_ids_reads_existing_output PASSED [100%]
============================== 3 passed in 0.07s ==============================
```
Committed as `fix: pass typed content parts to Qwen3-VL chat template`.
