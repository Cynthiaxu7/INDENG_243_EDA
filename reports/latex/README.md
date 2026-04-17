# Local build

To produce the PDF:

```
cd reports/latex
pdflatex synthetic_texture_paper.tex
pdflatex synthetic_texture_paper.tex  # second pass for refs
```

Requires: TeX Live with `amsmath`, `graphicx`, `booktabs`, `hyperref`, `caption`,
`natbib` (or `cite`), `microtype`, `geometry`. Figure files must exist at
`../figures/<stem>.png`.
