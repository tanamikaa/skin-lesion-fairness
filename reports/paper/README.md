# ELCVIA-format paper

`main.tex` + `elcvia.sty` — upload both to Overleaf (or any LaTeX2e
install with `times` available) and compile with pdfLaTeX.

Content mapped from the project README / TEAM_ROLES:
- Abstract, Intro, Related Work, Datasets, Method, Experimental Setup — filled in.
- Results, Discussion, Conclusion — placeholders (`\emph{[Placeholder ...]}`),
  to be filled in once `scripts/train.py` / `scripts/evaluate.py` produce real
  numbers (see `results/EXPERIMENTS.md`).

To update camera-ready info (volume/number/pages/year, Associate Editor,
dates) edit the block near the top of `main.tex` marked
`% INFORMATION TO FILL IN CAMERA READY VERSION`, per `CREADY.TXT` in the
original template zip.
