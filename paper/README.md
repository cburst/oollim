# Manuscript editions — four ridge approaches

- `oollim-arxiv.tex/pdf`: single-column paper with all appendices embedded.
- `oollim-interspeech.tex/pdf`: two-column paper; main text and disclosure fit within four pages, with references continuing on page five.
- `oollim-interspeech-supplementary.tex/pdf`: complete family comparisons, paired differences, score distributions, public-development results, Korean rubric, representation ablation, environment and rater details.
- `main.tex/pdf` and `supplementary.tex/pdf`: working mirrors of the Interspeech files.

The four approaches are 1-grid construct-wise correlation (1G-S), 1-grid pronunciation-accuracy error (1G-AMAE), 5-grid pronunciation-accuracy error (5G-AMAE), and three-ratio banded ridge. Banded ridge retains both correlation-selected (3RB-C) and pronunciation-accuracy-error-selected (3RB-A) variants: five conditions across seven families, or 35 configurations. Correlation means Spearman; pronunciation-accuracy error means accuracy-construct MAE on public development data. Banded accuracy is the corresponding refined scalar-ridge head, not a banded fit.

The main table reports every configuration within .001 of either unrounded construct maximum, with marginal 95% bootstrap intervals. These external leaders are descriptive selections. All 35 configurations have 305 completed evaluations; no private data were used for fitting, hyperparameter selection, or calibration. Exact evaluated heads are included in the software package; the default remains matched SpeechOcean with Spearman selection.

Marginal intervals retain the supplied 2,000-draw export. Paired comparisons use 1,200,000 draws for each of seeds 42 and 2026, with one Bonferroni correction across 510 contrasts. Seven adjusted decisions differ between seeds and are marked borderline. The reported main-leader conclusions are stable. Full-precision results are in `../results/paired-four-ridges-v1/`; the earlier 258-contrast archive remains separate.

Compile each named source with pdfLaTeX twice or latexmk. Interspeech uses the included unmodified class; arXiv uses standard article packages. Sources embed their tables and references. Public aggregate tables are in `../results/four-ridges-v1/`; private filenames, identifiers, recording-level scores and audio are excluded. Recomputing the paired statistics requires the private row-level inputs, which are not released.

The original .01 representation-promotion rule and public-development construct selections remain unchanged. These named-author drafts are not anonymized submissions. The comprehensive appendix table arranges 1G-S/1G-AMAE/5G-AMAE/3RB-C/3RB-A horizontally for each family, stacking the three constructs; paired comparisons and representation matrices retain compact grouped layouts.
