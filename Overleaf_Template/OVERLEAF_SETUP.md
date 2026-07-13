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
| `overleaf_intro_related.tex` | Introduction + Related Work |
| `overleaf_methods.tex` | Methods (stub) |
| `custom.bib` | Non-anthology references |

## Note on `overleaf.json`

Overleaf does **not** read `overleaf.json` for the main file. Set the main document and compiler in the project menu as above.
