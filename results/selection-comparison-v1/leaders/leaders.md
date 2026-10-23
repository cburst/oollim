# Observed leaders and descriptive ties

Human agreement averages each rater versus the other two; model agreement averages the same three two-rater references. S: construct-specific development Spearman; D: development accuracy MAE on the same standardized grid; H: development accuracy MAE on the historical grid converted to normalized loss. Every configuration within .001 of the highest observed Spearman or QWK for its construct among 21 combinations is shown, using unrounded estimates. Bold marks qualifying metrics and human baselines. Ties are descriptive, not tests of equivalence or superiority. Rows were chosen after external evaluation; all fitting and hyperparameter selection used public training/development data. Both metrics belong to each displayed configuration. No displayed prediction required clipping; QWK uses 0–100 integer categories with half-up rounding, and Spearman uses raw scores. Intervals are marginal 95% percentile-bootstrap intervals (2,000 draws, seed 42), not selection-adjusted. Full results are in the supplement. Its paired tests concern the matched-Spearman subset, not all displayed leaders.

| Construct | Family | Rule | Spearman | 95% CI | QWK | 95% CI |
|---|---|---|---:|---|---:|---|
| Accuracy | Human consensus | -- | **0.809** | [0.769, 0.842] | **0.704** | [0.663, 0.743] |
|  | Azure | -- | 0.761 | [0.709, 0.804] | 0.161 | [0.129, 0.196] |
|  | aesopocean | S | **0.872** | [0.842, 0.894] | 0.598 | [0.563, 0.631] |
|  | aesopocean | H | **0.872** | [0.841, 0.894] | 0.607 | [0.572, 0.640] |
|  | koreanocean | S | 0.853 | [0.816, 0.882] | **0.821** | [0.789, 0.847] |
| Fluency | Human consensus | -- | **0.807** | [0.768, 0.840] | **0.739** | [0.695, 0.777] |
|  | Azure | -- | 0.633 | [0.556, 0.698] | 0.078 | [0.056, 0.107] |
|  | aesopkoreanocean | D | **0.877** | [0.846, 0.902] | 0.577 | [0.539, 0.612] |
|  | koreanaesop | H | **0.878** | [0.848, 0.901] | 0.469 | [0.431, 0.502] |
|  | koreanocean | H | 0.838 | [0.799, 0.870] | **0.744** | [0.702, 0.779] |
| Prosody | Human consensus | -- | **0.807** | [0.768, 0.842] | **0.742** | [0.701, 0.778] |
|  | Azure | -- | 0.737 | [0.672, 0.792] | 0.370 | [0.318, 0.421] |
|  | koreanaesop | S | **0.883** | [0.852, 0.906] | 0.431 | [0.394, 0.465] |
|  | koreanaesop | D | **0.883** | [0.852, 0.906] | 0.431 | [0.394, 0.465] |
|  | koreanaesop | H | **0.883** | [0.853, 0.906] | 0.433 | [0.397, 0.468] |
|  | koreanocean | H | 0.840 | [0.802, 0.872] | **0.675** | [0.632, 0.712] |
