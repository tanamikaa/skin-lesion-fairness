# ELCVIA-format paper

`main.tex` + `elcvia.sty` + `figures/` — upload all to Overleaf (or any
LaTeX2e install with `times` available) and compile with pdfLaTeX.

All sections (Abstract through Conclusion) are filled in with the real
results from `reports/REPORT.md` and `results/*/*/metrics.json` /
`reports/figures/fairness_comparison.json` (one run per setting, seed
42, focal loss, ResNet-50 vs.\ ViT-B/16, with/without oversampling).
Numbers here should stay in sync with `reports/REPORT.md`, which is the
source of truth for the project's informal write-up; re-check both if
new experiment runs land in `results/`.

To update camera-ready info (volume/number/pages/year, Associate Editor,
dates) edit the block near the top of `main.tex` marked
`% INFORMATION TO FILL IN CAMERA READY VERSION`, per `CREADY.TXT` in the
original template zip.
