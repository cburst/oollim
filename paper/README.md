# Manuscript editions — selection-comparison revision

- `oollim-arxiv.tex/pdf`: single-column manuscript with all supplementary sections embedded.
- `oollim-interspeech.tex/pdf`: two-column manuscript.
- `oollim-interspeech-supplementary.tex/pdf`: full matched tables, three-condition sensitivity comparison, paired differences, score diagnostics, public-development results, Korean rubric, representation ablation, environment and rater details.
- `main.tex/pdf` and `supplementary.tex/pdf`: working mirrors of the Interspeech files.

The main table reports construct-specific leaders across all three selection rules, including every configuration within .001 of the highest unrounded Spearman or QWK. These are descriptive external leaders; complete common-condition comparisons remain in the supplement. The main table includes marginal 95% bootstrap intervals for both metrics; those intervals are not adjusted for selecting the displayed leaders. Both DAMAE conditions and the historical deployed SpeechOcean results remain disclosed in `../results/selection-comparison-v1/`. All 21 combinations have 305 completed evaluations; no private data were used for fitting or calibration. Existing application defaults remain unchanged.

Marginal intervals retain the supplied 2,000-draw export. Paired comparisons cover all 21 S/D/H models, with 600,000 draws each for seeds 42 and 2026 and a common Bonferroni correction across 258 contrasts. Both seeds agree on all adjusted interval-exclusion decisions. The correction covers the declared comparisons, not the preceding model search. Full-precision aggregates are in `../results/paired-all-selections-v1/`.

Compile each named source twice with pdfLaTeX. Interspeech uses the included unmodified class; arXiv uses standard article packages. Sources embed their tables and references. The reproduction release includes all aggregate results, exact evaluated heads, and the existing 945-row public representation-ablation CSV. No private filenames, identifiers, or recording-level scores are included.

The original .01 representation-promotion rule and public-development construct selections remain unchanged. These named-author drafts are not anonymized conference submissions. See `TEMPLATE.md` and `../docs/RIGHTS.md` for release constraints.

The compact appendix stacks all three constructs in one family/selection table, with S/D/H grouped horizontally and human/Azure benchmarks at the top of each panel. Paired differences share one landscape table; the three representation matrices share one vertically stacked table. These replace the duplicated full-agreement and selection-rule tables. Two- and three-rater aggregate results remain in results/selection-comparison-v1/external-all-conditions.csv. No new selection conditions or results are incorporated in this revision.
