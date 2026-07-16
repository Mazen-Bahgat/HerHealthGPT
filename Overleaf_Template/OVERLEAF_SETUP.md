# HerHealthGPT-LU — Overleaf setup

## Recommended (default)

| Setting | Value |
|---------|--------|
| **Main document** | `acl_latex.tex` |
| **Compiler** | **pdfLaTeX** |

Open **Menu** (gear icon) → **Settings** → set **Main document** to `acl_latex.tex` and **Compiler** to pdfLaTeX.

Or: right-click `acl_latex.tex` in the file tree → **Set as main document**.

## Alternative (Arabic script in the paper body)

Use this only when you need Arabic (or other Unicode) text rendered in the manuscript with `babel`/`fontspec`.

| Setting | Value |
|---------|--------|
| **Main document** | `acl_lualatex.tex` |
| **Compiler** | **LuaLaTeX** |

**Do not** compile `acl_lualatex.tex` with pdfLaTeX — it will fail on `\babelfont` and Unicode fonts.

## Bibliography files

Add these two shards to the project root (Overleaf: **New File** → **From External URL**):

- https://aclanthology.org/anthology-1.bib
- https://aclanthology.org/anthology-2.bib

Journal and non-ACL entries live in `custom.bib`. Both main files use:

```latex
\bibliography{custom,anthology-1,anthology-2}
```

## Project layout

| File | Role |
|------|------|
| `acl_latex.tex` | Main paper (pdfLaTeX) |
| `acl_lualatex.tex` | Same paper, LuaLaTeX preamble |
| `overleaf_titleauthor.tex` | Title + author — shared by both main files |
| `overleaf_abstract.tex` | Abstract — shared by both main files |
| `overleaf_intro_related.tex` | Introduction + Related Work |
| `overleaf_methods.tex` | Methods |
| `overleaf_results_discussion.tex` | Results + Discussion + Limitations — shared by both main files |
| `figures/*.tex` | Native TikZ figures, `\input` from the sections above (no image files needed) |
| `figures/*.mmd` | Mermaid source for the same figures, kept for reference/future editing (not compiled) |
| `custom.bib` | Non-anthology references |

`overleaf_titleauthor.tex`, `overleaf_abstract.tex`, and
`overleaf_results_discussion.tex` are the single source of truth for that
content — `acl_latex.tex` and `acl_lualatex.tex` both `\input` them rather
than duplicating the text, specifically so the two entry points cannot
silently drift apart (a prior edit once overwrote the Results/Discussion
section because it lived inline in only one of the two files). When
uploading to Overleaf, make sure every file in this table is present in the
project, including the `figures/` subfolder.

## Note on `overleaf.json`

Overleaf does **not** read `overleaf.json` for the main file. Set the main document and compiler in the project menu as above.

## Local compilation (VSCode / LaTeX Workshop)

Every `\input`-ed file carries a `% !TEX root = acl_latex.tex` (or
`../acl_latex.tex` inside `figures/`) magic comment on its first line, so
LaTeX Workshop's "Build LaTeX project" command works no matter which file is
focused — it always builds `acl_latex.tex`. Requires a local LaTeX
distribution (e.g. MiKTeX) with `latexmk` on `PATH`, and `anthology-1.bib` /
`anthology-2.bib` downloaded into this folder (same two files as the
Bibliography section above — gitignored, ~70MB combined, not part of the
repo).
