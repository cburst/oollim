# Public-data grid options v1

Locked before fitting, 2026-10-25. No private recordings, ratings or predictions enter training or selection. Existing deployments remain unchanged.

All seven corpus families, original speaker-disjoint training/development partitions, weighted training-only scaling, equal corpus weights, fixed representation combinations and normalized ridge loss are retained. Selection metrics average only included-corpus public development sets. Spearman selection uses each construct, then MAE, then stronger penalty; accuracy-MAE selection uses accuracy for the same representation combination, then stronger penalty. Each construct has independently fitted coefficients.

Baselines: **1-grid** = [0.001, 0.01, 0.1, 1, 10, 100]; **5-grid** = [0.00005, 0.0005, 0.005, 0.05, 0.5, 5]. Their prior names were standard and historical. Existing API identifiers are retained for compatibility.

New conditions:
1. **Expanded grid:** 19 half-order settings from 1e-6 to 1e3, plus all 1-grid/5-grid values. This intentionally tests both range and intermediate spacing; it does not isolate those two effects.
2. **Refined grid:** expanded candidates plus nine log-spaced points from winner/sqrt(10) through winner*sqrt(10), for each expanded public-development winner. The spacing is one eighth of an order of magnitude. The procedure is adaptive only to public development data and is specified before fitting.
3. **Banded ridge:** fluency/prosody X+R uses penalty ratios alpha_XLSR/alpha_XEUS of 1/4, 1 and 4. Search the expanded scalar grid at each ratio, plus the corresponding refined equal-penalty solution. Relative block gains are ratio**(1/4) for XEUS and ratio**(-1/4) for XLS-R, yielding penalties alpha/sqrt(ratio) and alpha*sqrt(ratio) in the original block-scaled coordinates. Accuracy is carried unchanged from the refined-grid condition, explicitly marked. Exact ties prefer equal penalties, then stronger scalar penalty.

All three profiles retain two selection rules: construct Spearman and accuracy MAE. Thus each profile exposes 7 families x 2 rules, each with three scores; banded accuracy is intentionally shared with refined, and equal solutions can legitimately recur. No promise of unique scores is made.

Report rho, MAE, and fixed 0..100 half-up rounded/clipped QWK on each public development corpus. Development maxima are tuning evidence, not independent generalization tests. Do not promote based on these scores alone or replace existing production defaults. All candidate selection scores, exact penalties, and checksums are saved. Fit on CPU with two BLAS threads, with durable per-group completion records and no GPU use.
