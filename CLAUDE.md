# Project Rules

## Authorship & Attribution
- Author name for all GitHub commits, pull requests, and code contributions: `j-vaught`
- Email for all GitHub code work: `jvaught@sc.edu`
- Written authorship (readmes, documentation, articles): `J.C. Vaught`

## Restrictions
- Never mention Claude Code, Anthropic, opus, haiku, or sonnet in:
  - Code comments
  - GitHub commit messages
  - Pull request descriptions
  - READMEs or documentation
  - Code authorship/co-author fields

## GitHub Management
- If a github repository exists, commit and push after every single file change. Do not batch changes or wait for features to be completed. Push immediately after each file is written or edited.
- If a repo does not exist, ask the user after each prompt if they want to create a github repo
- Recommend branches and PRs when they make sense.
- Always add a .gitignore file to the repo to ignore Virtual environments, build files, and compilation artifacts.
- Do not add output files like png, mp4, tex, pdf, and similar to the .gitignore

## Compute
- If you have a large Job that needs a GPU, check SLRUM to see if an A100 is avialable, if not, check for H200. If both are unavailable, just schedule it.
- If you have a smaller job, prioritize using SLURM, but if unavialble, or if the wait is more than a few hours for the next timeslot, consider using the comech-2080 and comech-2422 servers on tailscale ssh.
- If the job ONLY needs CPU, use a cpu or defq node; thses should always be available.
- Check both .err and .out files when running a job, and for most jobs, check the results every few minutes. 
- Make sure to kill jobs if they have issues or if you want to edit a script/make a change to the job.

## Plotting & Visualization
Use brand colors for all plots and visualizations:

**Primary:**
| Color | Hex | RGB |
|-------|-----|-----|
| Garnet | #73000A | 115, 0, 10 |
| Black | #000000 | 0, 0, 0 |
| White | #FFFFFF | 255, 255, 255 |

**Neutral:**
| Color | Hex | RGB |
|-------|-----|-----|
| 90% Black | #363636 | 54, 54, 54 |
| 70% Black | #5C5C5C | 92, 92, 92 |
| 50% Black | #A2A2A2 | 162, 162, 162 |
| 30% Black | #C7C7C7 | 199, 199, 199 |
| 10% Black | #ECECEC | 235, 235, 235 |
| Warm Grey | #676156 | 103, 97, 86 |
| Sandstorm | #FFF2E3 | 255, 242, 227 |

**Accent:**
| Color | Hex | RGB |
|-------|-----|-----|
| Rose | #CC2E40 | 204, 46, 64 |
| Atlantic | #466A9F | 70, 106, 159 |
| Congaree | #1F414D | 31, 65, 77 |
| Horseshoe | #65780B | 101, 120, 11 |
| Grass | #CED318 | 206, 211, 24 |
| Honeycomb | #A49137 | 164, 145, 55 |

- Avoid rounded edges on all plots, figures, and graphical elements
- Always use high contrast color schemes
- Avoid putting textboxxes, or titles in figures and plots
- include Axes information in figures and include legneds where it makes senss.

## Compiling LaTeX Documents
- If asked to write a .tex file, always compile it twice after writing.
- Always delete compilation files, except the pdf, after the second compilation.

## Technical Proposal Writing Style
For technical proposals (LaTeX or Typst), follow these conventions:

Punctuation and Prose:
- No colons before statements. Use periods to end sentences instead.
  - Incorrect: "The approach consists of: First, we..."
  - Correct: "The approach consists of three components. First, we..."
- Avoid bullet point lists. Write as narrative paragraphs instead.
- Do not use dashes where a comma or other punctuation would be more suited.
  - Em dashes should be used, but only sparingly and in more narrative or less technical settings.
- Use consistent periods at the end of all sentences and technical definitions.

Formatting and Emphasis:
- Use italics for emphasis - \textit{sample efficiency}, \textit{off-policy learning}; do not use \emph{}
- Do not emphasize often. A word defintion or a colloquial phrase may be emphasized, but text should avoid emphasis in general.
- Reserve bold for section titles (handled automatically by Typst and Latex)
- Place equations in numbered \begin{align} blocks, inline math in $...$ delimiters
- If using an equation, alwaya put on a new line; if defining a variable or a short representation like (n+1), this can be inline. 
- Use math mode for variable names and symbols: $\alpha$, $Q_{\min}$, not plain text

Structure and Organization:
- Use \chapter{}, \subsection{}, and subsubsection{} to organize content hierarchically
- Separate subsections with blank lines in source code for readability
- Present equations in narrative context before displaying them
- Use formal \begin{table} environment with \caption{} and \label{} for comparisons
- Define acronyms on first use (e.g., ASV, SAC, SoC)

Bibliography:
- Compile with: pdflatex -> bibtex -> pdflatex -> pdflatex
- Store bibliography in centralized references.bib file
- Use BibTeX key convention: LastName2024Topic
