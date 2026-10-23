# Manuscript editions — selection-comparison revision

- `oollim-arxiv.tex/pdf`: single-column manuscript with all supplementary sections embedded.
- `oollim-interspeech.tex/pdf`: two-column manuscript.
- `oollim-interspeech-supplementary.tex/pdf`: full matched tables, three-condition sensitivity comparison, paired differences, score diagnostics, public-development results, Korean rubric, representation ablation, environment and rater details.
- `main.tex/pdf` and `supplementary.tex/pdf`: working mirrors of the Interspeech files.

The main table reports construct-specific leaders across all three selection rules, including every configuration within .001 of the highest unrounded Spearman or QWK. These are descriptive external leaders; complete common-condition comparisons remain in the supplement. The main table includes marginal 95% bootstrap intervals for both metrics; those intervals are not adjusted for selecting the displayed leaders. Both DAMAE conditions and the historical deployed SpeechOcean results remain disclosed in `../results/selection-comparison-v1/`. All 21 combinations have 305 completed evaluations; no private data were used for fitting or calibration. Existing application defaults remain unchanged.

Marginal intervals use the supplied 2,000-draw export. Matched-Spearman paired differences were recomputed with 200,000 draws each for seeds 42 and 2026. Adjusted interval decisions agree for 89/90 contrasts; matched SpeechOcean prosody correlation versus Azure is borderline. The 90-contrast adjustment applies to that subset, not all displayed leaders or the selection-condition search.

Compile each named source twice with pdfLaTeX. Interspeech uses the included unmodified class; arXiv uses standard article packages. Sources embed their tables and references. The reproduction release includes all aggregate results, exact evaluated heads, and the existing 945-row public representation-ablation CSV. No private filenames, identifiers, or recording-level scores are included.

The original .01 representation-promotion rule and public-development construct selections remain unchanged. These named-author drafts are not anonymized conference submissions. See `TEMPLATE.md` and `../docs/RIGHTS.md` for release constraints.
