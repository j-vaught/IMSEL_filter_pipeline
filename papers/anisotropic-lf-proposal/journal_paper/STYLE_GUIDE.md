# Typst Section Writing Style Guide

## Format
- Each section is a standalone .typ file in sections/ that will be #include'd into the main document
- Do NOT include document-level settings (#set page, #set text, etc.) — those are in main.typ
- Do NOT include #import for IEEE template — that's in main.typ
- DO use #import "../colors.typ": * at the top if you need brand colors
- Use = for section headings, == for subsections, === for subsubsections
- Heading numbering is handled by the IEEE template, don't add manual numbers
- Use standard Typst math: $hat(f)_x$, $bold(A)$, $theta_k$, etc.
- Use @label for cross-references (define labels with <label>)
- Use @bibkey for citations (refs.bib is loaded in main.typ)
- Use #figure() with #table() or #image() for all figures and tables
- Tables use #table() inside #figure() with caption and label
- Images reference PDFs or PNGs in ../figures/ via #image("../figures/filename.ext")

## Writing Style
- IEEE journal style: formal but clear, third person
- Use \textit equivalent: _italics_ for emphasis on key terms
- No bullet point lists in the body text — write as narrative paragraphs
- Equations in numbered blocks using $ ... $ (display) or $...$ (inline)
- No colons before statements; use periods
- Define acronyms on first use
- Do not mention Claude, Anthropic, or any AI assistant

## Colors (from colors.typ)
- garnet (#73000A), atlantic (#466A9F), congaree (#1F414D), rose (#CC2E40)
- black90, black70, black50, black30, black10
- Use these for any CeTZ figures

## Figures
- CeTZ figures: create standalone .typ files in figures/cetz_src/, compile to PDF in figures/
- Existing PNG figures: reference from ../../figures/ (relative to journal_paper/)
- All figures must have captions and labels
