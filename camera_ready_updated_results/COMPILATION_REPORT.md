# Compilation Report

## Intended build (deliverable)
The project is configured for **LuaLaTeX** (per `% !TEX root = acl_lualatex.tex`
and the `polyglossia` + `\newfontfamily\arabicfont{Noto Naskh Arabic}` setup for
Arabic). On Overleaf, set the compiler to **LuaLaTeX**; the anthology bib shards are
included (`anthology-1.bib`, `anthology-2.bib`), so it compiles as-is. The preamble,
packages, and formatting configuration are **unchanged** from the supervisor's
version.

## Local verification
No LuaLaTeX is available in the audit environment, so verification used **Tectonic
(XeTeX engine)** on a *content-identical throwaway copy* with one preamble
reordering — the `polyglossia` block moved after the graphics packages — required
only because XeTeX's `bidi` demands graphics packages be loaded first (LuaLaTeX's
`luabidi` has no such constraint). **This reordering was NOT applied to the
deliverable**; it existed only in the discarded verification file. The Arabic font
(Noto Naskh Arabic) was installed locally to satisfy `fontspec`.

**Result: success — 12 pages, 0 unresolved references, 0 unresolved citations, 0
errors** (with all content edits present, bibliography resolved via the shards).

## Remaining warnings (all benign; none content-related)
| Warning | Cause | Impact |
|---|---|---|
| `Font shape TU/NotoNaskhArabic(0)/b/n` and `/m/it` undefined | Only the regular Arabic weight was installed locally; bold/italic Arabic fell back | Cosmetic; the document uses no bold/italic Arabic. On Overleaf the full family is present. |
| `inputenc package ignored with utf8 based engines` | XeTeX/LuaLaTeX are natively UTF-8 | Harmless; expected. |
| `microtype ... Unable to apply patch 'footnote'` | Engine-specific microtype patch | Cosmetic. |
| 1 × `Overfull \hbox (5.4pt too wide)` at the `tab:main` caption region | Minor line-width overflow | Cosmetic; ~5pt. |
| `accessing absolute path /usr/share/fonts/...` | Tectonic reproducibility notice for system fonts | Local-only; irrelevant on Overleaf. |

**No LaTeX errors. No undefined references or citations. No missing figures**
(after correcting the pre-existing figure path — see `CHANGELOG.md`).

## Notes
- A visual preview built during verification is included as
  `PREVIEW_xetex_12pp.pdf` (XeTeX; content-identical to the LuaLaTeX deliverable,
  preamble order differs). The authoritative PDF should be produced on Overleaf with
  LuaLaTeX.
- Both compile entry points (`acl_latex.tex`, `acl_lualatex.tex`) `\input` the same
  shared section files, so edits appear identically in both.
