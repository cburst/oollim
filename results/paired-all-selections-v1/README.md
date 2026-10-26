# Paired comparison across all S/D/H configurations

Extends the preceding 90-contrast S-only analysis to 258 comparisons across all 21 S/D/H configurations. The historical files remain unchanged in `../selection-comparison-v1/`; these new files supersede their inferential results in the manuscript.

- `PROTOCOL.md`: comparison scope and resampling/adjustment rules.
- `paired-seed42.json`, `paired-seed2026.json`: full-precision aggregate differences and intervals, 600,000 draws per seed.
- `paired-all-conditions.csv`: compact seed-42 aggregate table and cross-seed decision checks.
- `summary.json`: counts by rule, reference, and metric.
- `paired-table.tex`: reproducible appendix table.

All 258 adjusted interval-exclusion decisions agree across seeds. Against Azure, 61/63 correlation contrasts and 57/63 QWK contrasts have positive adjusted intervals. Against human consensus, koreanocean accuracy QWK has positive intervals under S, D, and H. H-koreanocean fluency QWK has a positive point difference but an interval containing zero; H-koreanocean prosody QWK has a negative adjusted interval.

Run `evaluation/paired_all_selections.py` on the authorized private input ZIP for each seed, then `reproduce/paired_selection_tables.py` on this directory. The repository contains aggregates only; the private input is not distributed. No models were fitted or selected in this analysis. Correction covers the 258 declared contrasts, not prior search or all possible comparisons between conditions.
