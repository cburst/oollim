# Paired S/D/H evaluation — v1

Scope fixed before this analysis: all 21 family/selection configurations, three constructs, and two metrics (Spearman and QWK). Each configuration is compared with human–other-rater consensus and Azure on identical two-rater references. Azure is also compared with human consensus, once per construct/metric. Total: (21*2+1)*3*2 = 258 contrasts.

Use all 305 unique recording IDs, treated as one speaker each as in the preceding analysis. Sample recordings jointly with replacement; recompute ranks including ties in each replicate. Human agreement averages each rater against the other two; machine agreement averages the same three pair-specific references. Raw predictions are retained for Spearman; QWK clips to 0–100 and rounds half-up to fixed integer categories.

Use 600,000 resamples for each seed, 42 and 2026. The increased count keeps about 58 draws per Bonferroni tail, comparable with the preceding 200,000-draw/90-contrast analysis. Report percentile 95% intervals and Bonferroni familywise 95% intervals with endpoints .025/258 and 1-.025/258. Report seed-42 intervals; use seed 2026 to check interval-exclusion stability. Disagreements are reported as borderline, not resolved by choosing a favorable seed. Monte Carlo tail uncertainty remains.

Correction covers these 258 reported comparisons, not model selection or all historical experiments. Do not infer condition-versus-condition differences from these intervals. The analysis conditions on the observed rater panel, trained models, and single passage. No fitting, calibration, new inference, or API requests. No private input rows or identifiers may be exported to public artifacts.

The optimized frequency-weighted midrank calculation is verified against explicit repeated-row ranking on synthetic data, including ties and clipped scores. All 132 system point estimates must match the existing aggregate export. Preserve the preceding 90-contrast files as historical evidence; use a separate versioned results directory.
