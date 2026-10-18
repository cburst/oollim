# Manuscript editions — 2026-10-19

- `oollim-arxiv.tex/pdf`: current single-column paper with all supplementary sections embedded as appendices.
- `oollim-interspeech.tex/pdf`: current two-column paper; main text within four pages, references through page five.
- `oollim-interspeech-supplementary.tex/pdf`: complete external tables, 200,000-draw paired comparisons, score diagnostics, public-development results, Korean rubric, all 315 representation configurations, selection decisions, environment and rater details.
- `main.tex/pdf`: working copy of the Interspeech paper; `supplementary.tex/pdf` mirrors its supplement.
- `ablation-corpus-results.csv`: all 945 corpus-level ablation comparisons (Spearman and MAE).

Compile each named source twice with pdfLaTeX. Interspeech requires the included unmodified `Interspeech.cls`; arXiv uses standard article packages. No external images or bibliography files are required. The source ZIPs include the ablation CSV.

The 54 marginal external rows retain the supplied 2,000-draw export. Paired intervals were recomputed from the complete evaluation scores with 200,000 draws, seeds 42 and 2026; all 90 interval-exclusion decisions agree. Only aggregate outputs and generic analysis code are released. See `../results/external-paired-200k-seed42.json` and its second-seed counterpart. No private scores were used for training or calibration.

The 0.01 public-development promotion rule retains the overall configurations. Two AESOP-only alternatives passed within-family gates but did not replace the externally evaluated models. See `../results/ablation-mac-v1/`.

Named-author editions are publication drafts, not anonymized submissions or evidence of venue acceptance. See `TEMPLATE.md` and `../docs/RIGHTS.md` for submission and release constraints.
