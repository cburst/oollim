# Full cached-feature ablation, Mac v1

Recorded before fitting, 2026-10-18. Public train/development data only.
Existing speaker-disjoint partitions and mapped 0–10 targets are preserved.
Korean fluency and prosody share a prosody-derived supervision target; they are not independent human construct labels.

Seven dataset families, every nonempty subset of XEUS/XLS-R/PhoneticXeus/IPA, three constructs. Corpus-balanced training, train-only scaling and original block gains; six alpha values from the matched family experiment.

Alpha selection: equal-weight mean Spearman across the three public development corpora, then lower macro MAE, then larger alpha. Unlike the prior included-corpus selection, all three development sets inform selection. These are development results, not untouched cross-corpus tests.

For each family/construct, rank feature sets by the same macro Spearman and MAE rule. Compare against the fixed-feature configuration refitted under this same selection protocol. Keep existing deployed heads unchanged.

Practical promotion gate: macro Spearman improvement >=0.01, no corpus Spearman deterioration >0.02, no corpus MAE deterioration >0.10 on the 0–10 scale; paired speaker bootstrap (2000 draws, seed 42) on shortlisted comparisons, requiring positive lower 95% macro-difference bound. These are prospective screening gates, not a correction for all sweep comparisons. If no candidate qualifies, retain the fixed-feature baseline. Shortlist candidates within 0.01 of the best macro Spearman; favor fewer encoder blocks when benefits are inconclusive. No automatic deployment. Report all candidate results.

Private recordings, ratings and evaluation outputs are excluded entirely. Final tests remain untouched. Public development selection does not establish private-data improvement.

Execution: Apple M1, two numerical CPU threads, serial fits, no GPU. Snapshot runner and checksummed manifests stored with run. No new extraction or changes to WSL API.
