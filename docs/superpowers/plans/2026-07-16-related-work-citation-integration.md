# Related-Work Citation Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete and verify the approved related-work integration in the current HerHealthEval manuscript without duplicating citations already added by commit `b9a4028`.

**Architecture:** Keep journal and non-ACL records in `Overleaf_Template/custom.bib` and resolve LUHME records through the existing ACL Anthology shards. Integrate citations narratively at the claims they support, add one compact positioning table, and validate citation keys and LaTeX syntax statically before attempting compilation.

**Tech Stack:** ACL LaTeX, BibTeX/Natbib, PowerShell, ripgrep, Git

## Global Constraints

- Preserve the current HerHealthEval framing and all citation work already present in commit `b9a4028`.
- Cite only essential domain papers and the approved methodological papers; do not cite unrelated PDFs merely because they are locally available.
- Do not strengthen claims about human validation, multilingual results, clinical readiness, or fine-tuning results beyond repository evidence.
- Keep the novelty claim narrow: matched symptom cases with controlled register variation, triage targets, and clarification behavior.
- Do not modify benchmark data, training code, evaluation code, Methods, Results, Discussion, or Limitations.
- Leave the unrelated untracked `~/` path untouched.
- Execute inline unless the user explicitly asks for sub-agent delegation.

---

### Task 1: Complete and Correct Bibliography Metadata

**Files:**
- Modify: `Overleaf_Template/custom.bib:123`
- Modify: `Overleaf_Template/custom.bib:170`

**Interfaces:**
- Consumes: DOI and publisher metadata for the PCOS systematic review and the existing `graca-etal-2026-assessing` record.
- Produces: Correct BibTeX records keyed `graca-etal-2026-assessing` and `ghaderzadeh-etal-2025-artificial`.

- [ ] **Step 1: Confirm the existing incorrect author fields**

Run:

```powershell
rg -n -A 10 "graca-etal-2026-assessing" Overleaf_Template/custom.bib
```

Expected: the current entry incorrectly lists `Sofia`, `Alice`, and `Crystal` instead of Sandro Graça, Alexander Dallaway, and Chris Kite.

- [ ] **Step 2: Correct the PCOS survey entry and add the systematic review**

Replace the author line in `graca-etal-2026-assessing` with:

```bibtex
  author  = {Gra{\c{c}}a, Sandro and Dallaway, Alexander and Alloh, Folashade and Randeva, Harpal S. and Kite, Chris and Kyrou, Ioannis},
```

Append this verified record:

```bibtex
@article{ghaderzadeh-etal-2025-artificial,
  author  = {Ghaderzadeh, Mustafa and Garavand, Ali and Salehnasab, Cirruse},
  title   = {Artificial Intelligence in Polycystic Ovary Syndrome: A Systematic Review of Diagnostic and Predictive Applications},
  journal = {BMC Medical Informatics and Decision Making},
  volume  = {25},
  pages   = {427},
  year    = {2025},
  doi     = {10.1186/s12911-025-03255-6}
}
```

- [ ] **Step 3: Verify unique keys and corrected metadata**

Run:

```powershell
rg -n "^@(article|misc|inproceedings)\{(graca-etal-2026-assessing|ghaderzadeh-etal-2025-artificial),|Gra\{|Dallaway, Alexander|Kite, Chris|10.1186/s12911-025-03255-6" Overleaf_Template/custom.bib
```

Expected: one entry for each key, the three corrected names, and the systematic-review DOI.

- [ ] **Step 4: Commit the bibliography unit**

```powershell
git add -- Overleaf_Template/custom.bib
git commit -m "docs: complete related-work bibliography metadata"
```

### Task 2: Integrate the Remaining Claims and Positioning Table

**Files:**
- Modify: `Overleaf_Template/overleaf_intro_related.tex:60`
- Modify: `Overleaf_Template/overleaf_intro_related.tex:72`
- Modify: `Overleaf_Template/overleaf_intro_related.tex:87`

**Interfaces:**
- Consumes: `ghaderzadeh-etal-2025-artificial` from Task 1 and ACL-shard key `heinisch-2025-terminologists`.
- Produces: Claim-level citations and a four-study positioning table.

- [ ] **Step 1: Add PCOS systematic-review context**

After the two PCOS response-evaluation studies, insert:

```latex
A systematic review of 80 PCOS AI studies shows that the broader field remains dominated by imaging and structured predictive modeling, while external validation, interpretability, and clinical integration remain recurring limitations \citep{ghaderzadeh-etal-2025-artificial}.
```

Expected: the text distinguishes conversational LLM evaluation from the wider structured-PCOS AI literature.

- [ ] **Step 2: Add terminology stewardship to multilingual methodology**

After the paragraph discussing cultural adaptation and common ground, insert:

```latex
Specialized multilingual communication also requires concept-level terminological precision rather than surface fluency alone; terminological stewardship is particularly important when domain mismatches can alter meaning in sensitive settings such as healthcare \citep{heinisch-2025-terminologists}.
```

Expected: `heinisch-2025-terminologists` supports the translation/terminology claim without being presented as an empirical medical study.

- [ ] **Step 3: Add the approved positioning table**

Insert before the final paragraph of `Gap and Evaluation Requirements`:

```latex
\begin{table*}[t]
\centering
\small
\begin{tabular}{p{0.16\textwidth}p{0.20\textwidth}p{0.14\textwidth}p{0.18\textwidth}p{0.20\textwidth}}
\hline
\textbf{Study} & \textbf{Primary target} & \textbf{Languages} & \textbf{Matched register variants} & \textbf{Explicit triage / clarification targets} \\
\hline
MenstLLaMA \citep{adhikary-etal-2025-menstrual} & Educational answer generation & English & No & No \\
Mai \citep{mughal-etal-2025-mai} & Menstrual-health dialogue generation & English, Roman Urdu & No & No \\
WHBench \citep{maurya-etal-2026-whbench} & Expert-scored clinical responses & English & No & Safety rubric; no clarification target \\
HerHealthEval & Symptom-language understanding & English, Arabic, French & Yes & Yes \\
\hline
\end{tabular}
\caption{Positioning against the closest women's-health LLM systems and benchmarks. HerHealthEval isolates expression form using matched cases and scores interpretation targets separately from response quality.}
\label{tab:positioning}
\end{table*}
```

Expected: the table makes the narrower novelty claim visible without asserting that prior women's-health benchmarks or multilingual chatbots do not exist.

- [ ] **Step 4: Confirm all newly required keys are cited**

Run:

```powershell
rg -n "ghaderzadeh-etal-2025-artificial|heinisch-2025-terminologists|tab:positioning" Overleaf_Template/overleaf_intro_related.tex Overleaf_Template/custom.bib
```

Expected: the systematic-review key appears in both files, the Heinisch key appears in the manuscript, and the table label appears once.

- [ ] **Step 5: Commit the narrative integration**

```powershell
git add -- Overleaf_Template/overleaf_intro_related.tex
git commit -m "docs: finish related-work citation integration"
```

### Task 3: Verify Citation Resolution and LaTeX Readiness

**Files:**
- Verify: `Overleaf_Template/acl_latex.tex`
- Verify: `Overleaf_Template/overleaf_intro_related.tex`
- Verify: `Overleaf_Template/custom.bib`

**Interfaces:**
- Consumes: the bibliography and manuscript edits from Tasks 1 and 2.
- Produces: evidence that local BibTeX keys are unique, expected ACL-shard keys are identified, and the manuscript is ready for Overleaf compilation.

- [ ] **Step 1: Run a static citation-key audit**

Run this PowerShell block from the repository root:

```powershell
$tex = Get-Content -Raw Overleaf_Template/overleaf_intro_related.tex
$cited = [regex]::Matches($tex, '\\cite[tp]?\{([^}]+)\}') |
  ForEach-Object { $_.Groups[1].Value -split ',' } |
  ForEach-Object { $_.Trim() } |
  Sort-Object -Unique
$bib = Get-Content -Raw Overleaf_Template/custom.bib
$local = [regex]::Matches($bib, '@\w+\{([^,]+),') |
  ForEach-Object { $_.Groups[1].Value } |
  Sort-Object -Unique
$expectedAcl = @(
  'anikina-etal-2025-building',
  'freitas-cardoso-2025-nightmare',
  'heinisch-2025-terminologists',
  'mohammadi-etal-2025-large',
  'nicholls-alperin-2025-cross',
  'donati-etal-2025-large',
  'hershcovich-etal-2022-challenges',
  'adilazuarda-etal-2024-towards',
  'testoni-fernandez-2024-asking',
  'zhang-choi-2025-clarify'
)
$unresolved = $cited | Where-Object { $_ -notin $local -and $_ -notin $expectedAcl }
if ($unresolved) { $unresolved; exit 1 }
Write-Output "Citation audit passed: $($cited.Count) unique keys"
```

Expected: `Citation audit passed: 24 unique keys` and exit code 0. If a key is reported, verify whether it belongs to an ACL shard or needs a local BibTeX record instead of simply adding it to the allowlist.

- [ ] **Step 2: Check for duplicate local BibTeX keys**

Run:

```powershell
$keys = [regex]::Matches((Get-Content -Raw Overleaf_Template/custom.bib), '@\w+\{([^,]+),') | ForEach-Object { $_.Groups[1].Value }
$duplicates = $keys | Group-Object | Where-Object Count -gt 1
if ($duplicates) { $duplicates | Format-Table Name, Count; exit 1 }
Write-Output "BibTeX key audit passed: $($keys.Count) unique local entries"
```

Expected: `BibTeX key audit passed: 17 unique local entries` and exit code 0.

- [ ] **Step 3: Check compiler availability and compile when possible**

Run:

```powershell
Get-Command latexmk -ErrorAction SilentlyContinue
```

If `latexmk` is found, run from `Overleaf_Template`:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error acl_latex.tex
```

Expected: exit code 0 and an updated `acl_latex.pdf`. If `latexmk` is absent, record that compilation must be performed in Overleaf and rely on the static audits above.

- [ ] **Step 4: Inspect the final diff and worktree**

Run:

```powershell
git diff HEAD~2 -- Overleaf_Template/custom.bib Overleaf_Template/overleaf_intro_related.tex
git status --short
```

Expected: only the approved bibliography, citation, and table changes are present; `~/` remains untracked and untouched.

- [ ] **Step 5: Record verification state**

If verification required any correction, amend the relevant focused commit or create a final correction commit:

```powershell
git add -- Overleaf_Template/custom.bib Overleaf_Template/overleaf_intro_related.tex
git commit -m "docs: verify related-work references"
```

Skip this commit when verification produces no additional file changes.
