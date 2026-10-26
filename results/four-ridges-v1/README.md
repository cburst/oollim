# Four ridge approaches

S: 1-grid construct-specific Spearman; D: 1-grid pronunciation-accuracy error; H: 5-grid pronunciation-accuracy error; B: three-ratio banded ridge with both Spearman (B-S) and accuracy-MAE (B-D) selection. Seven families give 35 configurations. The original 21 bundles are unchanged; 14 banded bundles retain the exact evaluated head bytes.

`external-all-conditions.csv` contains aggregate statistics only. Private recording identifiers, scores and audio are excluded. `leaders/` lists all two-rater leaders within .001 of each unrounded maximum. All hyperparameter fitting used public development partitions; external maxima are descriptive, not prospectively selected deployment choices.

Banded accuracy is copied from the corresponding refined scalar-ridge condition. Fluency/prosody search XLS-R/XEUS penalty ratios 1/4, 1, 4 and the expanded scalar grid, retaining the refined equal-penalty candidate. Equal scores across some choices are expected. See the versioned grid protocol and checkpoint metadata for exact penalties.

Paired comparisons are in `../paired-four-ridges-v1/`; prior 258-contrast results remain archived separately and are not used as the expanded analysis.
